use crate::bigram_data::{BigramData, BigramId};
use crate::interner::Interner;
use crate::lexeme_data::LexemeData;
use crate::lexeme_store::{Lexeme, LexemeStore};
use crate::py_bindings::{RunOutcome, StepResult};
use crate::scoring::{compute_scores, scores_close, CandidateScore, CandidateStats};
use crate::types::{LexemeId, Location, RunStatus, SelectionMethod, TokenId};
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
pub struct Engine {
    pub(crate) interner: Interner,
    pub(crate) lexeme_store: LexemeStore,
    pub(crate) lexemes: LexemeData,
    pub(crate) bigrams: BigramData,
    pub(crate) segment_delimiter: String,
    pub(crate) method: SelectionMethod,
    pub(crate) min_count: i64,
    pub(crate) rescore_interval: usize,
    pub(crate) candidate_scores: FxHashMap<BigramId, CandidateScore>,
    pub(crate) candidate_heap: BinaryHeap<HeapEntry>,
    pub(crate) bigram_generation: FxHashMap<BigramId, u64>,
    pub(crate) generation_counter: u64,
    pub(crate) dirty_bigrams: FxHashSet<BigramId>,
    pub(crate) iteration_counter: usize,
}

impl Engine {
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
                .values()
                .filter(|freq| **freq >= self.min_count)
                .count(),
            SelectionMethod::LogLikelihood | SelectionMethod::Npmi => self.candidate_scores.len(),
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
                        if *freq < self.min_count {
                            None
                        } else {
                            Some((*bigram, *freq))
                        }
                    })
                    .collect::<Vec<_>>();

                for (bigram, freq) in entries {
                    self.push_heap_entry(bigram, freq as f64, freq);
                }
            }
            SelectionMethod::LogLikelihood | SelectionMethod::Npmi => {
                let entries = self
                    .candidate_scores
                    .iter()
                    .map(|(bigram, score)| (*bigram, *score))
                    .collect::<Vec<_>>();

                for (bigram, score) in entries {
                    self.push_heap_entry(bigram, score.score, score.frequency);
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
                        if *freq < self.min_count {
                            None
                        } else {
                            Some((*bigram, *freq))
                        }
                    })
                    .collect::<Vec<_>>();
                for (bigram, freq) in entries {
                    self.push_heap_entry(bigram, freq as f64, freq);
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
                if freq < self.min_count {
                    self.remove_candidate_generation(bigram);
                    continue;
                }
                self.push_heap_entry(bigram, freq as f64, freq);
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
                    self.candidate_scores.insert(bigram, score);
                    self.push_heap_entry(bigram, score.score, score.frequency);
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
                self.candidate_scores.insert(bigram, score);
                self.push_heap_entry(bigram, score.score, score.frequency);
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
                    if freq < self.min_count {
                        continue;
                    }
                    CandidateScore {
                        score: freq as f64,
                        frequency: freq,
                    }
                }
                SelectionMethod::LogLikelihood | SelectionMethod::Npmi => {
                    let Some(candidate) = self.candidate_scores.get(&entry.bigram).copied() else {
                        continue;
                    };
                    if candidate.frequency < self.min_count {
                        continue;
                    }
                    candidate
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

        StepStatus::Winner(StepData {
            score: candidate.score,
            winner,
        })
    }

    pub(crate) fn run_internal(&mut self, iterations: usize, min_score: Option<f64>) -> RunOutcome {
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
