use crate::bigram_data::{BigramData, BigramId};
use crate::interner::Interner;
use crate::lexeme_data::LexemeData;
use crate::lexeme_store::{Lexeme, LexemeStore};
use crate::py_bindings::{RunOutcome, StepResult};
use crate::scoring::{compute_scores, scores_close, CandidateScore, CandidateStats};
use crate::types::{
    LexemeId, Location, RunStatus, SearchStrategy, SelectionMethod, StopwordPolicy, TokenId,
};
use pyo3::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

#[derive(Clone)]
pub(crate) struct WinnerInfo {
    pub(crate) bigram: BigramId,
    pub(crate) merged_lexeme: LexemeId,
    pub(crate) bigram_locations: Vec<Location>,
}

#[derive(Clone)]
pub(crate) struct StepData {
    pub(crate) score: f64,
    pub(crate) winner: WinnerInfo,
}

pub(crate) enum StepStatus {
    Winner(StepData),
    NoCandidate,
    BelowMinScore(f64),
}

#[derive(Clone)]
struct BeamState {
    engine: Engine,
    winners: Vec<StepResult>,
    cumulative_score: f64,
    insertion_order: usize,
}

pub(crate) fn apply_range_penalty(base_score: f64, range_factor: f64, range_alpha: f64) -> f64 {
    if range_alpha <= 0.0 {
        return base_score;
    }

    let multiplier = range_factor.powf(range_alpha).max(f64::MIN_POSITIVE);
    if base_score >= 0.0 {
        base_score * multiplier
    } else {
        base_score / multiplier
    }
}

#[derive(Clone, Debug)]
pub(crate) struct HeapEntry {
    pub(crate) score: f64,
    pub(crate) frequency: i64,
    pub(crate) merged_word: SmallVec<[TokenId; 6]>,
    pub(crate) bigram: BigramId,
    // Not part of ordering/equality semantics; used as a stale-entry guard.
    pub(crate) generation: u64,
}

impl PartialEq for HeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.bigram == other.bigram
            && self.frequency == other.frequency
            && self.score.to_bits() == other.score.to_bits()
            && self.merged_word == other.merged_word
    }
}

impl Eq for HeapEntry {}

impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score
            .total_cmp(&other.score)
            .then_with(|| self.frequency.cmp(&other.frequency))
            .then_with(|| other.merged_word.cmp(&self.merged_word))
            .then_with(|| self.bigram.cmp(&other.bigram))
    }
}

pub(crate) fn merged_word_ids(
    lexeme_store: &LexemeStore,
    bigram: BigramId,
) -> SmallVec<[TokenId; 6]> {
    let left = lexeme_store.get(bigram.left);
    let right = lexeme_store.get(bigram.right);

    let mut merged = SmallVec::<[TokenId; 6]>::with_capacity(left.word.len() + right.word.len());
    merged.extend_from_slice(&left.word);
    merged.extend_from_slice(&right.word);
    merged
}

pub(crate) fn merged_lexeme_ids_for_bigram(
    bigram: BigramId,
    lexeme_store: &mut LexemeStore,
) -> Vec<LexemeId> {
    let merged_word: SmallVec<[TokenId; 3]> = merged_word_ids(lexeme_store, bigram)
        .iter()
        .copied()
        .collect();
    let mut merged_ids = Vec::with_capacity(merged_word.len());

    if let Some(existing_root) = lexeme_store.id_for_lexeme(&Lexeme {
        word: merged_word.clone(),
        ix: 0,
    }) {
        merged_ids.push(existing_root);
        for lexeme_ix in 1..merged_word.len() {
            let lexeme = Lexeme {
                word: merged_word.clone(),
                ix: lexeme_ix,
            };
            if let Some(existing_id) = lexeme_store.id_for_lexeme(&lexeme) {
                merged_ids.push(existing_id);
            } else {
                merged_ids.clear();
                break;
            }
        }
        if merged_ids.len() == merged_word.len() {
            return merged_ids;
        }
    }

    for lexeme_ix in 0..merged_word.len() {
        merged_ids.push(lexeme_store.intern(Lexeme {
            word: merged_word.clone(),
            ix: lexeme_ix,
        }));
    }

    merged_ids
}

fn root_items_from_old_with_merges(
    old_root_items: &[(usize, LexemeId)],
    starts: &[usize],
    merged_root: LexemeId,
    merged_width: usize,
) -> Vec<(usize, LexemeId)> {
    if starts.is_empty() {
        return old_root_items.to_vec();
    }

    let start_set = starts.iter().copied().collect::<FxHashSet<_>>();
    let mut new_root_items = Vec::with_capacity(old_root_items.len());

    for (ix, lexeme) in old_root_items {
        let insertion_ix = starts.partition_point(|s| *s <= *ix);
        let in_merged_span = insertion_ix > 0 && *ix < starts[insertion_ix - 1] + merged_width;
        if !in_merged_span {
            new_root_items.push((*ix, *lexeme));
            continue;
        }

        if start_set.contains(ix) {
            new_root_items.push((*ix, merged_root));
        }
    }

    new_root_items
}

pub(crate) fn clean_bigram_locations(
    mut locations: Vec<Location>,
    merged_width: usize,
) -> Vec<Location> {
    locations.sort_unstable();

    let mut clean_locations = Vec::new();
    let mut ix = 0;

    while ix < locations.len() {
        let line = locations[ix].0;
        let mut token_ix = Vec::new();
        while ix < locations.len() && locations[ix].0 == line {
            token_ix.push(locations[ix].1);
            ix += 1;
        }

        let mut next_valid = 0usize;
        for token in token_ix.iter().copied() {
            if token >= next_valid {
                clean_locations.push((line, token));
                next_valid = token + merged_width;
            }
        }
    }

    clean_locations
}

pub(crate) fn winner_from_bigram_with_data(
    bigram: BigramId,
    merged_lexeme: LexemeId,
    merged_width: usize,
    bigram_data: &BigramData,
) -> WinnerInfo {
    let locations = bigram_data.locations_for_bigram(bigram);

    WinnerInfo {
        bigram,
        merged_lexeme,
        bigram_locations: clean_bigram_locations(locations, merged_width),
    }
}

pub(crate) fn merge_winner(
    winner: &WinnerInfo,
    merged_lexeme_ids: &[LexemeId],
    lexeme_data: &mut LexemeData,
    lexeme_store: &LexemeStore,
    bigram_data: &mut BigramData,
) -> FxHashSet<BigramId> {
    let mut bigram_lines = FxHashSet::default();
    let mut merge_starts_by_line: FxHashMap<usize, Vec<usize>> = FxHashMap::default();

    for (line_ix, word_ix) in &winner.bigram_locations {
        bigram_lines.insert(*line_ix);
        merge_starts_by_line
            .entry(*line_ix)
            .or_default()
            .push(*word_ix);
    }

    let mut touched_bigrams = FxHashSet::default();
    touched_bigrams.insert(winner.bigram);
    let mut old_location_removals: FxHashMap<BigramId, FxHashSet<Location>> = FxHashMap::default();
    let mut new_location_additions: FxHashMap<BigramId, Vec<Location>> = FxHashMap::default();
    let mut decremented_bigrams = FxHashSet::default();

    let mut old_bigrams_lookup: FxHashMap<usize, Vec<(usize, LexemeId)>> = FxHashMap::default();
    for line_ix in bigram_lines {
        old_bigrams_lookup.insert(
            line_ix,
            lexeme_data.root_items_for_line(line_ix, lexeme_store),
        );
    }

    for (line_ix, word_ix) in winner.bigram_locations.iter().copied() {
        for (lexeme_index, merged_lexeme) in merged_lexeme_ids.iter().enumerate() {
            let pos = word_ix + lexeme_index;
            let old_lexeme = lexeme_data.locations_to_lexemes[line_ix][pos];
            lexeme_data.locations_to_lexemes[line_ix][pos] = *merged_lexeme;

            if let Some(locations) = lexeme_data.lexemes_to_locations.get_mut(&old_lexeme) {
                locations.remove(&(line_ix, pos));
                if locations.is_empty() {
                    lexeme_data.lexemes_to_locations.remove(&old_lexeme);
                }
            }

            lexeme_data
                .lexemes_to_locations
                .entry(*merged_lexeme)
                .or_default()
                .insert((line_ix, pos));
        }
    }

    for (line_ix, old_root_items) in old_bigrams_lookup {
        let line_merge_starts = merge_starts_by_line
            .get_mut(&line_ix)
            .map(|starts| {
                starts.sort_unstable();
                starts.as_slice()
            })
            .unwrap_or(&[]);

        let old_bigrams = old_root_items
            .windows(2)
            .map(|pair| {
                (
                    BigramId {
                        left: pair[0].1,
                        right: pair[1].1,
                    },
                    (line_ix, pair[0].0),
                )
            })
            .collect::<Vec<_>>();

        let new_root_items = root_items_from_old_with_merges(
            &old_root_items,
            line_merge_starts,
            winner.merged_lexeme,
            merged_lexeme_ids.len(),
        );

        let new_bigrams = new_root_items
            .windows(2)
            .map(|pair| {
                (
                    BigramId {
                        left: pair[0].1,
                        right: pair[1].1,
                    },
                    (line_ix, pair[0].0),
                )
            })
            .collect::<Vec<_>>();

        for (bigram, _) in &old_bigrams {
            touched_bigrams.insert(*bigram);
        }
        for (bigram, _) in &new_bigrams {
            touched_bigrams.insert(*bigram);
        }

        for (bigram, location) in &new_bigrams {
            bigram_data.add_bigram(*bigram, 1);
            new_location_additions
                .entry(*bigram)
                .or_default()
                .push(*location);
        }

        for (bigram, location) in &old_bigrams {
            bigram_data.add_bigram(*bigram, -1);
            old_location_removals
                .entry(*bigram)
                .or_default()
                .insert(*location);
            decremented_bigrams.insert(*bigram);
            bigram_data.maybe_remove_lr_lexemes(*bigram);
        }
    }

    bigram_data.remove_locations_batch(&old_location_removals);
    bigram_data.add_locations_batch(&new_location_additions);
    for bigram in decremented_bigrams {
        bigram_data.maybe_remove_bigram(bigram);
    }

    touched_bigrams
}

#[pyclass]
#[derive(Clone)]
pub struct Engine {
    pub(crate) interner: Interner,
    pub(crate) lexeme_store: LexemeStore,
    pub(crate) lexemes: LexemeData,
    pub(crate) bigrams: BigramData,
    pub(crate) segment_delimiter: String,
    pub(crate) method: SelectionMethod,
    pub(crate) min_count: i64,
    pub(crate) rescore_interval: usize,
    pub(crate) stopword_token_ids: FxHashSet<TokenId>,
    pub(crate) stopword_policy: StopwordPolicy,
    pub(crate) block_punct_only: bool,
    pub(crate) min_range: usize,
    pub(crate) range_alpha: f64,
    pub(crate) min_p_ab: Option<f64>,
    pub(crate) min_p_ba: Option<f64>,
    pub(crate) min_merge_count: usize,
    pub(crate) search_strategy: SearchStrategy,
    pub(crate) beam_width: usize,
    pub(crate) beam_top_m: usize,
    pub(crate) candidate_scores: FxHashMap<BigramId, CandidateScore>,
    pub(crate) candidate_heap: BinaryHeap<HeapEntry>,
    pub(crate) bigram_generation: FxHashMap<BigramId, u64>,
    pub(crate) generation_counter: u64,
    pub(crate) dirty_bigrams: FxHashSet<BigramId>,
    pub(crate) iteration_counter: usize,
}

impl Engine {
    fn token_is_punct(token: &str) -> bool {
        !token.is_empty() && token.chars().all(|ch| !ch.is_alphanumeric())
    }

    fn lexeme_is_stopword_unigram(&self, lexeme_id: LexemeId) -> bool {
        let lexeme = self.lexeme_store.get(lexeme_id);
        if lexeme.word.len() != 1 {
            return false;
        }
        self.stopword_token_ids.contains(&lexeme.word[0])
    }

    fn lexeme_is_punct_only(&self, lexeme_id: LexemeId) -> bool {
        let lexeme = self.lexeme_store.get(lexeme_id);
        !lexeme.word.is_empty()
            && lexeme
                .word
                .iter()
                .all(|token_id| Self::token_is_punct(self.interner.token(*token_id)))
    }

    fn candidate_is_structurally_filtered(&self, bigram: BigramId) -> bool {
        if self.block_punct_only
            && (self.lexeme_is_punct_only(bigram.left) || self.lexeme_is_punct_only(bigram.right))
        {
            return true;
        }

        match self.stopword_policy {
            StopwordPolicy::None => false,
            StopwordPolicy::BlockStopwordStopword => {
                self.lexeme_is_stopword_unigram(bigram.left)
                    && self.lexeme_is_stopword_unigram(bigram.right)
            }
            StopwordPolicy::BlockAnyStopword => {
                self.lexeme_is_stopword_unigram(bigram.left)
                    || self.lexeme_is_stopword_unigram(bigram.right)
            }
        }
    }

    fn candidate_probabilities(&self, bigram: BigramId, freq: i64) -> Option<(f64, f64)> {
        let left_freq = *self.bigrams.left_lex_freqs.get(&bigram.left)?;
        let right_freq = *self.bigrams.right_lex_freqs.get(&bigram.right)?;
        if left_freq <= 0 || right_freq <= 0 {
            return None;
        }
        let p_b_given_a = freq as f64 / left_freq as f64;
        let p_a_given_b = freq as f64 / right_freq as f64;
        Some((p_b_given_a, p_a_given_b))
    }

    fn candidate_passes_probability_gates(&self, bigram: BigramId, freq: i64) -> bool {
        if self.min_p_ab.is_none() && self.min_p_ba.is_none() {
            return true;
        }
        let Some((p_ab, p_ba)) = self.candidate_probabilities(bigram, freq) else {
            return false;
        };
        if let Some(threshold) = self.min_p_ab {
            if p_ab < threshold {
                return false;
            }
        }
        if let Some(threshold) = self.min_p_ba {
            if p_ba < threshold {
                return false;
            }
        }
        true
    }

    fn bigram_merge_count(&self, bigram: BigramId) -> usize {
        let merged_width = self.lexeme_store.get(bigram.left).word.len()
            + self.lexeme_store.get(bigram.right).word.len();
        let locations = self.bigrams.locations_for_bigram(bigram);
        clean_bigram_locations(locations, merged_width).len()
    }

    fn candidate_passes_merge_count(&self, bigram: BigramId) -> bool {
        if self.min_merge_count <= 1 {
            return true;
        }
        self.bigram_merge_count(bigram) >= self.min_merge_count
    }

    fn candidate_is_selection_filtered(&self, bigram: BigramId, freq: i64) -> bool {
        self.candidate_is_structurally_filtered(bigram)
            || !self.candidate_passes_probability_gates(bigram, freq)
            || !self.candidate_passes_merge_count(bigram)
    }

    fn range_enabled(&self) -> bool {
        self.min_range > 1 || self.range_alpha > 0.0
    }

    fn segment_range_for_bigram(&self, bigram: BigramId) -> usize {
        self.bigrams.segment_range_for_bigram(bigram)
    }

    fn range_factor_for_bigram(&self, range: usize) -> f64 {
        let max_range = self.lexemes.corpus_length();
        if max_range <= 1 || range == 0 {
            return 1.0;
        }
        (1.0 + range as f64).ln() / (1.0 + max_range as f64).ln()
    }

    fn range_adjusted_score(&self, base_score: f64, range: usize) -> f64 {
        let range_factor = self.range_factor_for_bigram(range);
        apply_range_penalty(base_score, range_factor, self.range_alpha)
    }

    fn candidate_selection_score(&self, bigram: BigramId, base_score: f64) -> Option<f64> {
        if !self.range_enabled() {
            return Some(base_score);
        }

        let range = self.segment_range_for_bigram(bigram);
        if self.min_range > 1 && range < self.min_range {
            return None;
        }

        Some(self.range_adjusted_score(base_score, range))
    }

    pub(crate) fn token_ids_to_strings(&self, token_ids: &[TokenId]) -> Vec<String> {
        self.interner.ids_to_strings(token_ids)
    }

    fn heap_entry(
        &self,
        bigram: BigramId,
        score: f64,
        frequency: i64,
        generation: u64,
    ) -> HeapEntry {
        HeapEntry {
            score,
            frequency,
            merged_word: merged_word_ids(&self.lexeme_store, bigram),
            bigram,
            generation,
        }
    }

    fn bump_generation(&mut self, bigram: BigramId) -> u64 {
        self.generation_counter = self.generation_counter.saturating_add(1);
        self.bigram_generation
            .insert(bigram, self.generation_counter);
        self.generation_counter
    }

    fn push_heap_entry(&mut self, bigram: BigramId, score: f64, frequency: i64) {
        let generation = self.bump_generation(bigram);
        self.candidate_heap
            .push(self.heap_entry(bigram, score, frequency, generation));
    }

    fn remove_candidate_generation(&mut self, bigram: BigramId) {
        self.bigram_generation.remove(&bigram);
    }

    fn maybe_compact_candidate_heap(&mut self) {
        let live_candidates = match self.method {
            SelectionMethod::Frequency => self
                .bigrams
                .bigrams_to_freqs
                .iter()
                .filter(|(bigram, freq)| {
                    **freq >= self.min_count
                        && !self.candidate_is_selection_filtered(**bigram, **freq)
                        && self
                            .candidate_selection_score(**bigram, **freq as f64)
                            .is_some()
                })
                .count(),
            SelectionMethod::LogLikelihood
            | SelectionMethod::Npmi
            | SelectionMethod::LogDice
            | SelectionMethod::TScore
            | SelectionMethod::DeltaP => self
                .candidate_scores
                .iter()
                .filter(|(bigram, score)| {
                    !self.candidate_is_selection_filtered(**bigram, score.frequency)
                        && self
                            .candidate_selection_score(**bigram, score.score)
                            .is_some()
                })
                .count(),
        };
        if live_candidates == 0 {
            self.candidate_heap.clear();
            self.bigram_generation.clear();
            return;
        }

        let heap_len = self.candidate_heap.len();
        let heap_growth = heap_len.saturating_sub(live_candidates);
        if heap_len <= live_candidates.saturating_mul(4) || heap_growth < 2048 {
            return;
        }

        self.candidate_heap.clear();
        self.bigram_generation.clear();

        match self.method {
            SelectionMethod::Frequency => {
                let entries = self
                    .bigrams
                    .bigrams_to_freqs
                    .iter()
                    .filter_map(|(bigram, freq)| {
                        if *freq < self.min_count
                            || self.candidate_is_selection_filtered(*bigram, *freq)
                        {
                            None
                        } else {
                            self.candidate_selection_score(*bigram, *freq as f64)
                                .map(|selection_score| (*bigram, *freq, selection_score))
                        }
                    })
                    .collect::<Vec<_>>();

                for (bigram, freq, selection_score) in entries {
                    self.push_heap_entry(bigram, selection_score, freq);
                }
            }
            SelectionMethod::LogLikelihood
            | SelectionMethod::Npmi
            | SelectionMethod::LogDice
            | SelectionMethod::TScore
            | SelectionMethod::DeltaP => {
                let entries = self
                    .candidate_scores
                    .iter()
                    .filter_map(|(bigram, score)| {
                        if self.candidate_is_selection_filtered(*bigram, score.frequency) {
                            None
                        } else {
                            self.candidate_selection_score(*bigram, score.score)
                                .map(|selection_score| (*bigram, score.frequency, selection_score))
                        }
                    })
                    .collect::<Vec<_>>();

                for (bigram, frequency, selection_score) in entries {
                    self.push_heap_entry(bigram, selection_score, frequency);
                }
            }
        }
    }

    pub(crate) fn refresh_candidate_state(&mut self, force_full: bool) {
        if self.method == SelectionMethod::Frequency {
            if force_full || self.candidate_heap.is_empty() {
                self.candidate_heap.clear();
                self.bigram_generation.clear();
                let entries = self
                    .bigrams
                    .bigrams_to_freqs
                    .iter()
                    .filter_map(|(bigram, freq)| {
                        if *freq < self.min_count
                            || self.candidate_is_selection_filtered(*bigram, *freq)
                        {
                            None
                        } else {
                            self.candidate_selection_score(*bigram, *freq as f64)
                                .map(|selection_score| (*bigram, *freq, selection_score))
                        }
                    })
                    .collect::<Vec<_>>();
                for (bigram, freq, selection_score) in entries {
                    self.push_heap_entry(bigram, selection_score, freq);
                }
                self.dirty_bigrams.clear();
                return;
            }

            let dirty = self.dirty_bigrams.drain().collect::<Vec<_>>();
            for bigram in dirty {
                let Some(freq) = self.bigrams.bigrams_to_freqs.get(&bigram).copied() else {
                    self.remove_candidate_generation(bigram);
                    continue;
                };
                if freq < self.min_count || self.candidate_is_selection_filtered(bigram, freq) {
                    self.remove_candidate_generation(bigram);
                    continue;
                }
                let Some(selection_score) = self.candidate_selection_score(bigram, freq as f64)
                else {
                    self.remove_candidate_generation(bigram);
                    continue;
                };
                self.push_heap_entry(bigram, selection_score, freq);
            }
            self.maybe_compact_candidate_heap();
            return;
        }

        #[allow(clippy::manual_is_multiple_of)]
        let full = force_full
            || self.candidate_scores.is_empty()
            || self.iteration_counter % self.rescore_interval == 0;

        let total_bigram_count = self.bigrams.total_bigram_count;

        if full {
            let full_stats = self
                .bigrams
                .bigrams_to_freqs
                .iter()
                .map(|(bigram, freq)| CandidateStats {
                    bigram: *bigram,
                    freq: Some(*freq),
                    left_freq: *self.bigrams.left_lex_freqs.get(&bigram.left).unwrap_or(&0),
                    right_freq: *self
                        .bigrams
                        .right_lex_freqs
                        .get(&bigram.right)
                        .unwrap_or(&0),
                })
                .collect::<Vec<_>>();

            let rescored =
                compute_scores(self.method, self.min_count, total_bigram_count, &full_stats);

            self.candidate_scores.clear();
            self.candidate_heap.clear();
            self.bigram_generation.clear();
            for (bigram, maybe_score) in rescored {
                if let Some(score) = maybe_score {
                    if self.candidate_is_selection_filtered(bigram, score.frequency) {
                        self.remove_candidate_generation(bigram);
                        continue;
                    }
                    let Some(selection_score) = self.candidate_selection_score(bigram, score.score)
                    else {
                        self.remove_candidate_generation(bigram);
                        continue;
                    };
                    self.candidate_scores.insert(bigram, score);
                    self.push_heap_entry(bigram, selection_score, score.frequency);
                }
            }

            self.dirty_bigrams.clear();
            return;
        }

        let dirty = self.dirty_bigrams.drain().collect::<Vec<_>>();
        if dirty.is_empty() {
            return;
        }

        let dirty_stats = dirty
            .iter()
            .map(|bigram| CandidateStats {
                bigram: *bigram,
                freq: self.bigrams.bigrams_to_freqs.get(bigram).copied(),
                left_freq: self
                    .bigrams
                    .left_lex_freqs
                    .get(&bigram.left)
                    .copied()
                    .unwrap_or(0),
                right_freq: self
                    .bigrams
                    .right_lex_freqs
                    .get(&bigram.right)
                    .copied()
                    .unwrap_or(0),
            })
            .collect::<Vec<_>>();

        let rescored = compute_scores(
            self.method,
            self.min_count,
            total_bigram_count,
            &dirty_stats,
        );

        for (bigram, maybe_score) in rescored {
            if let Some(score) = maybe_score {
                if self.candidate_is_selection_filtered(bigram, score.frequency) {
                    self.candidate_scores.remove(&bigram);
                    self.remove_candidate_generation(bigram);
                    continue;
                }
                let Some(selection_score) = self.candidate_selection_score(bigram, score.score)
                else {
                    self.candidate_scores.remove(&bigram);
                    self.remove_candidate_generation(bigram);
                    continue;
                };
                self.candidate_scores.insert(bigram, score);
                self.push_heap_entry(bigram, selection_score, score.frequency);
            } else {
                self.candidate_scores.remove(&bigram);
                self.remove_candidate_generation(bigram);
            }
        }
        self.maybe_compact_candidate_heap();
    }

    pub(crate) fn select_candidate(&mut self) -> Option<(BigramId, CandidateScore)> {
        while let Some(entry) = self.candidate_heap.pop() {
            let current_generation = self
                .bigram_generation
                .get(&entry.bigram)
                .copied()
                .unwrap_or(0);
            if entry.generation != current_generation {
                continue;
            }

            let current = match self.method {
                SelectionMethod::Frequency => {
                    let Some(freq) = self.bigrams.bigrams_to_freqs.get(&entry.bigram).copied()
                    else {
                        continue;
                    };
                    if freq < self.min_count
                        || self.candidate_is_selection_filtered(entry.bigram, freq)
                    {
                        continue;
                    }
                    let Some(selection_score) =
                        self.candidate_selection_score(entry.bigram, freq as f64)
                    else {
                        continue;
                    };
                    CandidateScore {
                        score: selection_score,
                        frequency: freq,
                    }
                }
                SelectionMethod::LogLikelihood
                | SelectionMethod::Npmi
                | SelectionMethod::LogDice
                | SelectionMethod::TScore
                | SelectionMethod::DeltaP => {
                    let Some(candidate) = self.candidate_scores.get(&entry.bigram).copied() else {
                        continue;
                    };
                    if candidate.frequency < self.min_count
                        || self.candidate_is_selection_filtered(entry.bigram, candidate.frequency)
                    {
                        continue;
                    }
                    let Some(selection_score) =
                        self.candidate_selection_score(entry.bigram, candidate.score)
                    else {
                        continue;
                    };
                    CandidateScore {
                        score: selection_score,
                        frequency: candidate.frequency,
                    }
                }
            };

            if entry.frequency != current.frequency || !scores_close(entry.score, current.score) {
                continue;
            }

            return Some((entry.bigram, current));
        }

        None
    }

    pub(crate) fn step_result(&self, step_data: StepData) -> StepResult {
        let merge_token_count = step_data.winner.bigram_locations.len();
        let merge_segment_range = step_data
            .winner
            .bigram_locations
            .iter()
            .map(|(line_ix, _)| *line_ix)
            .collect::<FxHashSet<_>>()
            .len();
        let left = self.lexeme_store.get(step_data.winner.bigram.left);
        let right = self.lexeme_store.get(step_data.winner.bigram.right);
        let merged = self.lexeme_store.get(step_data.winner.merged_lexeme);

        StepResult {
            score: step_data.score,
            left_word: self.token_ids_to_strings(&left.word),
            left_ix: left.ix,
            right_word: self.token_ids_to_strings(&right.word),
            right_ix: right.ix,
            merged_word: self.token_ids_to_strings(&merged.word),
            merged_ix: merged.ix,
            merge_token_count,
            merge_segment_range,
        }
    }

    pub(crate) fn annotate_corpus_internal(
        &self,
        mwe_prefix: &str,
        mwe_suffix: &str,
        token_separator: &str,
    ) -> (Vec<String>, Vec<String>) {
        let mut annotated_documents =
            Vec::with_capacity(self.lexemes.doc_boundaries.len().saturating_sub(1));
        let mut mwe_labels = FxHashSet::default();

        for boundary in self.lexemes.doc_boundaries.windows(2) {
            let start = boundary[0];
            let end = boundary[1];
            let mut annotated_segments = Vec::with_capacity(end.saturating_sub(start));

            for line_ix in start..end {
                let mut annotated_tokens = Vec::new();

                for lexeme_id in &self.lexemes.locations_to_lexemes[line_ix] {
                    let lexeme = self.lexeme_store.get(*lexeme_id);
                    if lexeme.ix > 0 {
                        continue;
                    }

                    if lexeme.word.len() > 1 {
                        let lexeme_tokens = self.token_ids_to_strings(&lexeme.word);
                        let label = format!(
                            "{mwe_prefix}{}{mwe_suffix}",
                            lexeme_tokens.join(token_separator)
                        );
                        mwe_labels.insert(label.clone());
                        annotated_tokens.push(label);
                    } else {
                        let token = self.token_ids_to_strings(&lexeme.word).join("");
                        annotated_tokens.push(token);
                    }
                }

                annotated_segments.push(annotated_tokens.join(" "));
            }

            annotated_documents.push(annotated_segments.join(&self.segment_delimiter));
        }

        let mut sorted_labels = mwe_labels.into_iter().collect::<Vec<_>>();
        sorted_labels.sort_unstable();
        (annotated_documents, sorted_labels)
    }

    fn apply_selected_candidate(
        &mut self,
        bigram: BigramId,
        candidate: CandidateScore,
    ) -> Option<StepData> {
        let freq = match self.method {
            SelectionMethod::Frequency => self.bigrams.bigrams_to_freqs.get(&bigram).copied()?,
            _ => self
                .candidate_scores
                .get(&bigram)
                .copied()
                .map(|score| score.frequency)?,
        };

        if freq < self.min_count || self.candidate_is_selection_filtered(bigram, freq) {
            return None;
        }

        let merged_lexeme_ids = merged_lexeme_ids_for_bigram(bigram, &mut self.lexeme_store);
        let winner = winner_from_bigram_with_data(
            bigram,
            merged_lexeme_ids[0],
            merged_lexeme_ids.len(),
            &self.bigrams,
        );

        let touched = merge_winner(
            &winner,
            &merged_lexeme_ids,
            &mut self.lexemes,
            &self.lexeme_store,
            &mut self.bigrams,
        );
        self.dirty_bigrams = touched;
        self.iteration_counter += 1;

        Some(StepData {
            score: candidate.score,
            winner,
        })
    }

    fn top_candidates(
        &self,
        min_score: Option<f64>,
        limit: usize,
    ) -> (Vec<(BigramId, CandidateScore)>, Option<f64>) {
        let mut probe = self.clone();
        probe.refresh_candidate_state(false);

        let mut out = Vec::new();
        let mut best_below_min = None;
        while out.len() < limit {
            let Some((bigram, candidate)) = probe.select_candidate() else {
                break;
            };
            if let Some(threshold) = min_score {
                if candidate.score < threshold {
                    best_below_min = Some(candidate.score);
                    break;
                }
            }
            out.push((bigram, candidate));
        }
        (out, best_below_min)
    }

    fn sort_beam_states(states: &mut [BeamState]) {
        states.sort_by(|left, right| {
            right
                .cumulative_score
                .total_cmp(&left.cumulative_score)
                .then_with(|| right.winners.len().cmp(&left.winners.len()))
                .then_with(|| left.insertion_order.cmp(&right.insertion_order))
        });
    }

    fn run_beam_internal(&mut self, iterations: usize, min_score: Option<f64>) -> RunOutcome {
        let corpus_length = self.lexemes.corpus_length();

        let mut states = vec![BeamState {
            engine: self.clone(),
            winners: Vec::new(),
            cumulative_score: 0.0,
            insertion_order: 0,
        }];
        let mut insertion_counter = 1usize;

        for _ in 0..iterations {
            let mut next_states = Vec::new();
            let mut best_below_min: Option<f64> = None;

            for state in &states {
                let (candidates, below_min) =
                    state.engine.top_candidates(min_score, self.beam_top_m);
                if let Some(score) = below_min {
                    best_below_min = match best_below_min {
                        Some(current) => Some(current.max(score)),
                        None => Some(score),
                    };
                }
                if candidates.is_empty() {
                    continue;
                }

                for (bigram, candidate) in candidates {
                    let mut branch_engine = state.engine.clone();
                    let Some(step_data) = branch_engine.apply_selected_candidate(bigram, candidate)
                    else {
                        continue;
                    };
                    let step_result = branch_engine.step_result(step_data);
                    let mut branch_winners = state.winners.clone();
                    branch_winners.push(step_result);
                    next_states.push(BeamState {
                        engine: branch_engine,
                        winners: branch_winners,
                        cumulative_score: state.cumulative_score + candidate.score,
                        insertion_order: insertion_counter,
                    });
                    insertion_counter = insertion_counter.saturating_add(1);
                }
            }

            if next_states.is_empty() {
                let mut terminal_states = states;
                Self::sort_beam_states(&mut terminal_states);
                let best = terminal_states.into_iter().next().unwrap_or(BeamState {
                    engine: self.clone(),
                    winners: Vec::new(),
                    cumulative_score: 0.0,
                    insertion_order: 0,
                });
                *self = best.engine;
                if let Some(score) = best_below_min {
                    return (
                        RunStatus::BelowMinScore.code(),
                        best.winners,
                        Some(score),
                        corpus_length,
                    );
                }
                return (
                    RunStatus::NoCandidate.code(),
                    best.winners,
                    None,
                    corpus_length,
                );
            }

            Self::sort_beam_states(&mut next_states);
            if next_states.len() > self.beam_width {
                next_states.truncate(self.beam_width);
            }
            states = next_states;
        }

        Self::sort_beam_states(&mut states);
        let best = states.into_iter().next().unwrap_or(BeamState {
            engine: self.clone(),
            winners: Vec::new(),
            cumulative_score: 0.0,
            insertion_order: 0,
        });
        *self = best.engine;
        (
            RunStatus::Completed.code(),
            best.winners,
            None,
            corpus_length,
        )
    }

    pub(crate) fn step_internal(&mut self, min_score: Option<f64>) -> StepStatus {
        self.refresh_candidate_state(false);

        let Some((bigram, candidate)) = self.select_candidate() else {
            return StepStatus::NoCandidate;
        };

        if let Some(score_threshold) = min_score {
            if candidate.score < score_threshold {
                return StepStatus::BelowMinScore(candidate.score);
            }
        }
        let Some(step_data) = self.apply_selected_candidate(bigram, candidate) else {
            return StepStatus::NoCandidate;
        };
        StepStatus::Winner(step_data)
    }

    pub(crate) fn run_internal(&mut self, iterations: usize, min_score: Option<f64>) -> RunOutcome {
        if self.search_strategy == SearchStrategy::Beam && self.beam_width > 1 {
            return self.run_beam_internal(iterations, min_score);
        }

        let mut winners = Vec::new();
        let corpus_length = self.lexemes.corpus_length();

        for _ in 0..iterations {
            match self.step_internal(min_score) {
                StepStatus::NoCandidate => {
                    return (RunStatus::NoCandidate.code(), winners, None, corpus_length)
                }
                StepStatus::BelowMinScore(score) => {
                    return (
                        RunStatus::BelowMinScore.code(),
                        winners,
                        Some(score),
                        corpus_length,
                    )
                }
                StepStatus::Winner(step_data) => {
                    winners.push(self.step_result(step_data));
                }
            }
        }

        (RunStatus::Completed.code(), winners, None, corpus_length)
    }
}
