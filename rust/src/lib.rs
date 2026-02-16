use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};
use sentencex::segment;
use smallvec::{smallvec, SmallVec};
use std::cmp::Ordering;
use std::collections::BinaryHeap;

const SMALL: f64 = 1e-10;
const SCORE_ATOL: f64 = 1e-12;
const SCORE_RTOL: f64 = 1e-12;
const DEFAULT_RESCORE_INTERVAL: usize = 25;
const PARALLEL_SCORE_THRESHOLD: usize = 500;

type TokenId = u32;
type LexemeId = u32;
type Location = (usize, usize);
#[cfg(feature = "location-hashset")]
type BigramLocations = FxHashSet<Location>;
#[cfg(not(feature = "location-hashset"))]
type BigramLocations = Vec<Location>;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SelectionMethod {
    Frequency,
    LogLikelihood,
    Npmi,
}

impl SelectionMethod {
    fn parse(value: &str) -> PyResult<Self> {
        match value {
            "frequency" => Ok(Self::Frequency),
            "log_likelihood" => Ok(Self::LogLikelihood),
            "npmi" => Ok(Self::Npmi),
            _ => Err(PyValueError::new_err(format!(
                "Invalid method {value:?}. Expected one of: 'frequency', 'log_likelihood', 'npmi'."
            ))),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Splitter<'a> {
    Delimiter(Option<&'a str>),
    Sentencex { language: &'a str },
}

impl<'a> Splitter<'a> {
    fn parse(
        splitter: &'a str,
        line_delimiter: Option<&'a str>,
        sentencex_language: &'a str,
    ) -> PyResult<Self> {
        match splitter {
            "delimiter" => Ok(Self::Delimiter(line_delimiter)),
            "sentencex" => {
                if sentencex_language.trim().is_empty() {
                    return Err(PyValueError::new_err(
                        "sentencex_language must be a non-empty language code.",
                    ));
                }
                Ok(Self::Sentencex {
                    language: sentencex_language,
                })
            }
            _ => Err(PyValueError::new_err(format!(
                "Invalid splitter {splitter:?}. Expected one of: 'delimiter', 'sentencex'."
            ))),
        }
    }
}

#[derive(Default)]
struct Interner {
    str_to_id: FxHashMap<String, TokenId>,
    id_to_str: Vec<String>,
}

impl Interner {
    fn from_documents(
        documents: &[String],
        splitter: Splitter<'_>,
    ) -> (Self, Vec<Vec<TokenId>>, Vec<usize>) {
        let mut uniq = FxHashSet::default();
        let mut tokenized_segments = Vec::new();
        let mut doc_boundaries = Vec::with_capacity(documents.len() + 1);

        for document in documents {
            doc_boundaries.push(tokenized_segments.len());

            let mut ingest_segment = |segment: &str| {
                let tokens = segment
                    .split_whitespace()
                    .map(str::to_string)
                    .collect::<Vec<_>>();
                if tokens.is_empty() {
                    return;
                }
                uniq.extend(tokens.iter().cloned());
                tokenized_segments.push(tokens);
            };

            match splitter {
                Splitter::Delimiter(Some(delim)) => {
                    for segment in document.split(delim) {
                        ingest_segment(segment);
                    }
                }
                Splitter::Delimiter(None) => ingest_segment(document),
                Splitter::Sentencex { language } => {
                    for sentence in segment(language, document) {
                        ingest_segment(sentence);
                    }
                }
            }
        }
        doc_boundaries.push(tokenized_segments.len());

        let mut sorted = uniq.into_iter().collect::<Vec<_>>();
        sorted.sort_unstable();

        let mut interner = Self::default();
        interner.str_to_id.reserve(sorted.len());
        interner.id_to_str.reserve(sorted.len());
        for token in sorted {
            let id = TokenId::try_from(interner.id_to_str.len())
                .expect("token vocabulary exceeded TokenId capacity (u32)");
            interner.str_to_id.insert(token.clone(), id);
            interner.id_to_str.push(token);
        }

        let corpus_ids = tokenized_segments
            .into_iter()
            .map(|tokens| {
                tokens
                    .into_iter()
                    .map(|token| interner.id_for(&token))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        (interner, corpus_ids, doc_boundaries)
    }

    fn id_for(&self, value: &str) -> TokenId {
        *self
            .str_to_id
            .get(value)
            .expect("token missing in interner while converting corpus")
    }

    fn ids_to_strings(&self, ids: &[TokenId]) -> Vec<String> {
        ids.iter()
            .map(|id| self.id_to_str[*id as usize].clone())
            .collect()
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
struct Lexeme {
    word: SmallVec<[TokenId; 3]>,
    ix: usize,
}

#[derive(Default)]
struct LexemeStore {
    lexeme_to_id: FxHashMap<Lexeme, LexemeId>,
    id_to_lexeme: Vec<Lexeme>,
}

impl LexemeStore {
    fn intern(&mut self, lexeme: Lexeme) -> LexemeId {
        if let Some(id) = self.lexeme_to_id.get(&lexeme) {
            return *id;
        }

        let id = LexemeId::try_from(self.id_to_lexeme.len())
            .expect("lexeme vocabulary exceeded LexemeId capacity (u32)");
        self.id_to_lexeme.push(lexeme.clone());
        self.lexeme_to_id.insert(lexeme, id);
        id
    }

    fn get(&self, id: LexemeId) -> &Lexeme {
        &self.id_to_lexeme[id as usize]
    }

    fn id_for_lexeme(&self, lexeme: &Lexeme) -> Option<LexemeId> {
        self.lexeme_to_id.get(lexeme).copied()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
struct BigramId {
    left: LexemeId,
    right: LexemeId,
}

#[derive(Default)]
struct LexemeData {
    lexemes_to_locations: FxHashMap<LexemeId, FxHashSet<Location>>,
    locations_to_lexemes: Vec<Vec<LexemeId>>,
    doc_boundaries: Vec<usize>,
}

impl LexemeData {
    fn from_corpus(
        corpus: &[Vec<TokenId>],
        doc_boundaries: Vec<usize>,
        lexeme_store: &mut LexemeStore,
    ) -> Self {
        let mut lexeme_data = Self {
            doc_boundaries,
            ..Self::default()
        };

        for (line_ix, tokens) in corpus.iter().enumerate() {
            let mut line_lexemes = Vec::with_capacity(tokens.len());
            for (word_ix, word) in tokens.iter().enumerate() {
                let lexeme_id = lexeme_store.intern(Lexeme {
                    word: smallvec![*word],
                    ix: 0,
                });
                let loc = (line_ix, word_ix);
                lexeme_data
                    .lexemes_to_locations
                    .entry(lexeme_id)
                    .or_default()
                    .insert(loc);
                line_lexemes.push(lexeme_id);
            }
            lexeme_data.locations_to_lexemes.push(line_lexemes);
        }

        lexeme_data
    }

    fn corpus_length(&self) -> usize {
        self.locations_to_lexemes.len()
    }

    fn root_items_for_line(
        &self,
        line_ix: usize,
        lexeme_store: &LexemeStore,
    ) -> Vec<(usize, LexemeId)> {
        self.locations_to_lexemes[line_ix]
            .iter()
            .enumerate()
            .filter_map(|(ix, lexeme)| {
                if lexeme_store.get(*lexeme).ix == 0 {
                    Some((ix, *lexeme))
                } else {
                    None
                }
            })
            .collect()
    }
}

#[derive(Default)]
struct BigramData {
    bigrams_to_freqs: FxHashMap<BigramId, i64>,
    total_bigram_count: i64,
    bigrams_to_locations: FxHashMap<BigramId, BigramLocations>,
    left_lex_freqs: FxHashMap<LexemeId, i64>,
    right_lex_freqs: FxHashMap<LexemeId, i64>,
}

impl BigramData {
    fn from_lexemes(lexeme_data: &LexemeData, lexeme_store: &LexemeStore) -> Self {
        let mut bigram_data = Self::default();

        for line_ix in 0..lexeme_data.corpus_length() {
            let root_items = lexeme_data.root_items_for_line(line_ix, lexeme_store);
            for pair in root_items.windows(2) {
                let bigram = BigramId {
                    left: pair[0].1,
                    right: pair[1].1,
                };
                let location = (line_ix, pair[0].0);
                bigram_data.add_location(bigram, location);
                bigram_data.add_bigram(bigram, 1);
            }
        }

        bigram_data
    }

    fn add_bigram(&mut self, bigram: BigramId, delta: i64) {
        *self.left_lex_freqs.entry(bigram.left).or_insert(0) += delta;
        *self.right_lex_freqs.entry(bigram.right).or_insert(0) += delta;
        *self.bigrams_to_freqs.entry(bigram).or_insert(0) += delta;
        self.total_bigram_count += delta;
    }

    fn add_location(&mut self, bigram: BigramId, location: Location) {
        #[cfg(feature = "location-hashset")]
        {
            self.bigrams_to_locations
                .entry(bigram)
                .or_default()
                .insert(location);
        }

        #[cfg(not(feature = "location-hashset"))]
        {
            self.bigrams_to_locations
                .entry(bigram)
                .or_default()
                .push(location);
        }
    }

    fn remove_locations_batch(&mut self, removals: &FxHashMap<BigramId, FxHashSet<Location>>) {
        for (bigram, to_remove) in removals {
            let mut should_remove = false;
            if let Some(locations) = self.bigrams_to_locations.get_mut(bigram) {
                #[cfg(feature = "location-hashset")]
                {
                    for location in to_remove {
                        locations.remove(location);
                    }
                }

                #[cfg(not(feature = "location-hashset"))]
                {
                    locations.retain(|location| !to_remove.contains(location));
                }

                should_remove = locations.is_empty();
            }

            if should_remove {
                self.bigrams_to_locations.remove(bigram);
            }
        }
    }

    fn add_locations_batch(&mut self, additions: &FxHashMap<BigramId, Vec<Location>>) {
        for (bigram, locations) in additions {
            #[cfg(feature = "location-hashset")]
            {
                let target = self.bigrams_to_locations.entry(*bigram).or_default();
                for location in locations {
                    target.insert(*location);
                }
            }

            #[cfg(not(feature = "location-hashset"))]
            {
                self.bigrams_to_locations
                    .entry(*bigram)
                    .or_default()
                    .extend(locations.iter().copied());
            }
        }
    }

    fn locations_for_bigram(&self, bigram: BigramId) -> Vec<Location> {
        #[cfg(feature = "location-hashset")]
        {
            self.bigrams_to_locations
                .get(&bigram)
                .map(|locations| locations.iter().copied().collect())
                .unwrap_or_default()
        }

        #[cfg(not(feature = "location-hashset"))]
        {
            self.bigrams_to_locations
                .get(&bigram)
                .cloned()
                .unwrap_or_default()
        }
    }

    fn maybe_remove_bigram(&mut self, bigram: BigramId) {
        let should_remove = self
            .bigrams_to_freqs
            .get(&bigram)
            .map(|freq| *freq <= 0)
            .unwrap_or(false);
        if should_remove {
            self.bigrams_to_freqs.remove(&bigram);
            self.bigrams_to_locations.remove(&bigram);
        }
    }

    fn maybe_remove_lr_lexemes(&mut self, bigram: BigramId) {
        let remove_left = self
            .left_lex_freqs
            .get(&bigram.left)
            .map(|freq| *freq <= 0)
            .unwrap_or(false);
        if remove_left {
            self.left_lex_freqs.remove(&bigram.left);
        }

        let remove_right = self
            .right_lex_freqs
            .get(&bigram.right)
            .map(|freq| *freq <= 0)
            .unwrap_or(false);
        if remove_right {
            self.right_lex_freqs.remove(&bigram.right);
        }
    }
}

#[derive(Clone)]
struct WinnerInfo {
    bigram: BigramId,
    merged_lexeme: LexemeId,
    bigram_locations: Vec<Location>,
}

#[derive(Clone, Copy)]
struct CandidateScore {
    score: f64,
    frequency: i64,
}

#[derive(Clone)]
struct StepData {
    score: f64,
    winner: WinnerInfo,
}

enum StepStatus {
    Winner(StepData),
    NoCandidate,
    BelowMinScore(f64),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
enum RunStatus {
    Completed = 0,
    NoCandidate = 1,
    BelowMinScore = 2,
}

impl RunStatus {
    fn code(self) -> u8 {
        self as u8
    }
}

#[derive(Clone, Copy)]
struct CandidateStats {
    bigram: BigramId,
    freq: Option<i64>,
    left_freq: i64,
    right_freq: i64,
}

#[pyclass(frozen)]
#[derive(Clone, Debug, PartialEq)]
struct StepResult {
    #[pyo3(get)]
    score: f64,
    #[pyo3(get)]
    left_word: Vec<String>,
    #[pyo3(get)]
    left_ix: usize,
    #[pyo3(get)]
    right_word: Vec<String>,
    #[pyo3(get)]
    right_ix: usize,
    #[pyo3(get)]
    merged_word: Vec<String>,
    #[pyo3(get)]
    merged_ix: usize,
    #[pyo3(get)]
    bigram_locations: Vec<Location>,
}

fn safe_ll_term(observed: f64, expected: f64) -> f64 {
    if observed > 0.0 {
        observed * (((observed / (expected + SMALL)) + SMALL).ln())
    } else {
        0.0
    }
}

fn scores_close(a: f64, b: f64) -> bool {
    (a - b).abs() <= (SCORE_ATOL + SCORE_RTOL * b.abs())
}

fn is_close_default(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-8 + (1e-5 * b.abs())
}

fn coerce_score(score: f64) -> f64 {
    if score.is_finite() {
        score
    } else {
        f64::NEG_INFINITY
    }
}

fn score_ll_npmi(
    method: SelectionMethod,
    bigram_freq: i64,
    left_freq: i64,
    right_freq: i64,
    total_bigram_count: i64,
) -> f64 {
    if total_bigram_count == 0 {
        return f64::NEG_INFINITY;
    }

    let total = total_bigram_count as f64;

    match method {
        SelectionMethod::LogLikelihood => {
            let obs_a = bigram_freq as f64;
            let obs_b = left_freq as f64 - obs_a;
            let obs_c = right_freq as f64 - obs_a;
            let mut obs_d = total - obs_a - obs_b - obs_c;
            if obs_d < 0.0 {
                obs_d = 0.0;
            }

            let exp_a = ((obs_a + obs_b) * (obs_a + obs_c)) / total;
            let exp_b = ((obs_a + obs_b) * (obs_b + obs_d)) / total;
            let exp_c = ((obs_c + obs_d) * (obs_a + obs_c)) / total;
            let exp_d = ((obs_c + obs_d) * (obs_b + obs_d)) / total;

            let ll_a = safe_ll_term(obs_a, exp_a);
            let ll_b = safe_ll_term(obs_b, exp_b);
            let ll_c = safe_ll_term(obs_c, exp_c);
            let ll_d = safe_ll_term(obs_d, exp_d);
            let ll = 2.0 * (ll_a + ll_b + ll_c + ll_d);
            coerce_score(if obs_a > exp_a { ll } else { -ll })
        }
        SelectionMethod::Npmi => {
            let prob_ab = bigram_freq as f64 / total;
            let prob_a = left_freq as f64 / total;
            let prob_b = right_freq as f64 / total;
            let numerator = (prob_ab / (prob_a * prob_b)).ln();
            let denominator = -(prob_ab.ln());
            let npmi = if denominator > 0.0 {
                numerator / denominator
            } else {
                f64::NAN
            };
            let perfect_association =
                is_close_default(denominator, 0.0) && is_close_default(numerator, 0.0);
            coerce_score(if perfect_association { 1.0 } else { npmi })
        }
        SelectionMethod::Frequency => bigram_freq as f64,
    }
}

fn merged_word_ids(lexeme_store: &LexemeStore, bigram: BigramId) -> SmallVec<[TokenId; 6]> {
    let left = lexeme_store.get(bigram.left);
    let right = lexeme_store.get(bigram.right);

    let mut merged = SmallVec::<[TokenId; 6]>::with_capacity(left.word.len() + right.word.len());
    merged.extend_from_slice(&left.word);
    merged.extend_from_slice(&right.word);
    merged
}

fn merged_lexeme_ids_for_bigram(bigram: BigramId, lexeme_store: &mut LexemeStore) -> Vec<LexemeId> {
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

fn clean_bigram_locations(mut locations: Vec<Location>, merged_width: usize) -> Vec<Location> {
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

fn winner_from_bigram_with_data(
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

fn merge_winner(
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

#[derive(Clone, Debug)]
struct HeapEntry {
    score: f64,
    frequency: i64,
    merged_word: SmallVec<[TokenId; 6]>,
    bigram: BigramId,
    generation: u64,
}

impl PartialEq for HeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.bigram == other.bigram
            && self.frequency == other.frequency
            && self.score.to_bits() == other.score.to_bits()
            && self.merged_word == other.merged_word
            && self.generation == other.generation
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

fn score_stats(
    method: SelectionMethod,
    min_count: i64,
    total_bigram_count: i64,
    stats: CandidateStats,
) -> (BigramId, Option<CandidateScore>) {
    let Some(freq) = stats.freq else {
        return (stats.bigram, None);
    };

    if freq < min_count {
        return (stats.bigram, None);
    }

    let score = match method {
        SelectionMethod::Frequency => freq as f64,
        _ => score_ll_npmi(
            method,
            freq,
            stats.left_freq,
            stats.right_freq,
            total_bigram_count,
        ),
    };

    if score == f64::NEG_INFINITY {
        (stats.bigram, None)
    } else {
        (
            stats.bigram,
            Some(CandidateScore {
                score,
                frequency: freq,
            }),
        )
    }
}

fn compute_scores(
    method: SelectionMethod,
    min_count: i64,
    total_bigram_count: i64,
    stats: &[CandidateStats],
) -> Vec<(BigramId, Option<CandidateScore>)> {
    if stats.len() >= PARALLEL_SCORE_THRESHOLD {
        stats
            .par_iter()
            .copied()
            .map(|item| score_stats(method, min_count, total_bigram_count, item))
            .collect()
    } else {
        stats
            .iter()
            .copied()
            .map(|item| score_stats(method, min_count, total_bigram_count, item))
            .collect()
    }
}

#[pyclass]
struct Engine {
    interner: Interner,
    lexeme_store: LexemeStore,
    lexemes: LexemeData,
    bigrams: BigramData,
    segment_delimiter: String,
    method: SelectionMethod,
    min_count: i64,
    rescore_interval: usize,
    candidate_scores: FxHashMap<BigramId, CandidateScore>,
    candidate_heap: BinaryHeap<HeapEntry>,
    bigram_generation: FxHashMap<BigramId, u64>,
    generation_counter: u64,
    dirty_bigrams: FxHashSet<BigramId>,
    iteration_counter: usize,
}

impl Engine {
    fn token_ids_to_strings(&self, token_ids: &[TokenId]) -> Vec<String> {
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

    fn refresh_candidate_state(&mut self, force_full: bool) {
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

    fn select_candidate(&mut self) -> Option<(BigramId, CandidateScore)> {
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

    fn step_result(&self, step_data: StepData) -> StepResult {
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
            bigram_locations: step_data.winner.bigram_locations,
        }
    }

    fn annotate_corpus_internal(
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

    fn step_internal(&mut self, min_score: Option<f64>) -> StepStatus {
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

    fn run_internal(&mut self, iterations: usize, min_score: Option<f64>) -> RunOutcome {
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

type RunOutcome = (u8, Vec<StepResult>, Option<f64>, usize);
type AnnotateRunOutcome = (
    u8,
    Vec<StepResult>,
    Option<f64>,
    usize,
    Vec<String>,
    Vec<String>,
);

#[pymethods]
impl Engine {
    #[new]
    #[pyo3(signature = (
        corpus,
        method,
        min_count,
        splitter="delimiter",
        line_delimiter=None,
        sentencex_language="en",
        rescore_interval=DEFAULT_RESCORE_INTERVAL,
    ))]
    fn new(
        corpus: Vec<String>,
        method: &str,
        min_count: usize,
        splitter: &str,
        line_delimiter: Option<&str>,
        sentencex_language: &str,
        rescore_interval: usize,
    ) -> PyResult<Self> {
        if rescore_interval == 0 {
            return Err(PyValueError::new_err(
                "rescore_interval must be greater than or equal to 1.",
            ));
        }

        let method = SelectionMethod::parse(method)?;
        let splitter = Splitter::parse(splitter, line_delimiter, sentencex_language)?;
        let segment_delimiter = match splitter {
            Splitter::Delimiter(Some(delim)) => delim.to_string(),
            Splitter::Delimiter(None) => String::new(),
            Splitter::Sentencex { .. } => " ".to_string(),
        };

        let (interner, corpus_ids, doc_boundaries) = Interner::from_documents(&corpus, splitter);

        let mut lexeme_store = LexemeStore::default();
        let lexemes = LexemeData::from_corpus(&corpus_ids, doc_boundaries, &mut lexeme_store);
        let bigrams = BigramData::from_lexemes(&lexemes, &lexeme_store);

        let mut engine = Self {
            interner,
            lexeme_store,
            lexemes,
            bigrams,
            segment_delimiter,
            method,
            min_count: min_count as i64,
            rescore_interval,
            candidate_scores: FxHashMap::default(),
            candidate_heap: BinaryHeap::new(),
            bigram_generation: FxHashMap::default(),
            generation_counter: 0,
            dirty_bigrams: FxHashSet::default(),
            iteration_counter: 0,
        };
        engine.refresh_candidate_state(true);

        Ok(engine)
    }

    fn corpus_length(&self) -> usize {
        self.lexemes.corpus_length()
    }

    #[pyo3(signature = (iterations, min_score=None))]
    fn run(&mut self, py: Python<'_>, iterations: usize, min_score: Option<f64>) -> RunOutcome {
        py.allow_threads(|| self.run_internal(iterations, min_score))
    }

    #[pyo3(signature = (
        iterations,
        min_score=None,
        mwe_prefix="<mwe:",
        mwe_suffix=">",
        token_separator="_",
    ))]
    fn run_and_annotate(
        &mut self,
        py: Python<'_>,
        iterations: usize,
        min_score: Option<f64>,
        mwe_prefix: &str,
        mwe_suffix: &str,
        token_separator: &str,
    ) -> AnnotateRunOutcome {
        py.allow_threads(|| {
            let (status, payloads, selected_score, corpus_length) =
                self.run_internal(iterations, min_score);
            let (annotated_docs, mwe_labels) =
                self.annotate_corpus_internal(mwe_prefix, mwe_suffix, token_separator);
            (
                status,
                payloads,
                selected_score,
                corpus_length,
                annotated_docs,
                mwe_labels,
            )
        })
    }
}

#[pymodule(gil_used = true)]
fn _core(_py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<Engine>()?;
    module.add_class::<StepResult>()?;
    module.add("STATUS_COMPLETED", RunStatus::Completed.code())?;
    module.add("STATUS_NO_CANDIDATE", RunStatus::NoCandidate.code())?;
    module.add("STATUS_BELOW_MIN_SCORE", RunStatus::BelowMinScore.code())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn build_engine(
        corpus: Vec<&str>,
        method: SelectionMethod,
        min_count: i64,
        rescore_interval: usize,
    ) -> Engine {
        let corpus = corpus.into_iter().map(str::to_string).collect::<Vec<_>>();
        let (interner, corpus_ids, doc_boundaries) =
            Interner::from_documents(&corpus, Splitter::Delimiter(Some("\n")));

        let mut lexeme_store = LexemeStore::default();
        let lexemes = LexemeData::from_corpus(&corpus_ids, doc_boundaries, &mut lexeme_store);
        let bigrams = BigramData::from_lexemes(&lexemes, &lexeme_store);

        let mut engine = Engine {
            interner,
            lexeme_store,
            lexemes,
            bigrams,
            segment_delimiter: "\n".to_string(),
            method,
            min_count,
            rescore_interval,
            candidate_scores: FxHashMap::default(),
            candidate_heap: BinaryHeap::new(),
            bigram_generation: FxHashMap::default(),
            generation_counter: 0,
            dirty_bigrams: FxHashSet::default(),
            iteration_counter: 0,
        };
        engine.refresh_candidate_state(true);
        engine
    }

    fn total_root_count(engine: &Engine) -> usize {
        (0..engine.lexemes.corpus_length())
            .map(|line_ix| {
                engine
                    .lexemes
                    .root_items_for_line(line_ix, &engine.lexeme_store)
                    .len()
            })
            .sum()
    }

    #[test]
    fn interner_roundtrip() {
        let corpus = vec!["b a b".to_string()];
        let (interner, corpus_ids, doc_boundaries) =
            Interner::from_documents(&corpus, Splitter::Delimiter(Some("\n")));
        let ids = vec![interner.id_for("a"), interner.id_for("b")];
        assert_eq!(interner.ids_to_strings(&ids), vec!["a", "b"]);
        assert!(interner.id_for("a") < interner.id_for("b"));
        assert_eq!(
            corpus_ids,
            vec![vec![
                interner.id_for("b"),
                interner.id_for("a"),
                interner.id_for("b")
            ]]
        );
        assert_eq!(doc_boundaries, vec![0, 1]);
    }

    #[test]
    fn deterministic_tie_break_prefers_frequency_then_lexicographic() {
        let mut engine = build_engine(
            vec!["a d", "a c", "a b", "a b"],
            SelectionMethod::Frequency,
            0,
            DEFAULT_RESCORE_INTERVAL,
        );

        let StepStatus::Winner(step) = engine.step_internal(None) else {
            panic!("expected winner");
        };
        assert_eq!(
            engine.token_ids_to_strings(&engine.lexeme_store.get(step.winner.merged_lexeme).word),
            vec!["a", "b"]
        );
    }

    #[test]
    fn engine_run_matches_repeated_step_for_frequency() {
        let corpus = vec!["a a a a"];
        let mut run_engine = build_engine(
            corpus.clone(),
            SelectionMethod::Frequency,
            0,
            DEFAULT_RESCORE_INTERVAL,
        );
        let mut step_engine = build_engine(
            corpus,
            SelectionMethod::Frequency,
            0,
            DEFAULT_RESCORE_INTERVAL,
        );

        let (_, run_payloads, _, _) = run_engine.run_internal(3, None);

        let mut step_payloads = Vec::new();
        for _ in 0..3 {
            match step_engine.step_internal(None) {
                StepStatus::Winner(step_data) => {
                    step_payloads.push(step_engine.step_result(step_data));
                }
                _ => break,
            }
        }

        assert_eq!(run_payloads, step_payloads);
    }

    #[test]
    fn merge_winner_updates_all_indexes_consistently() {
        let mut engine = build_engine(
            vec!["a b a b"],
            SelectionMethod::Frequency,
            0,
            DEFAULT_RESCORE_INTERVAL,
        );

        let left = engine.lexeme_store.intern(Lexeme {
            word: smallvec![engine.interner.id_for("a")],
            ix: 0,
        });
        let right = engine.lexeme_store.intern(Lexeme {
            word: smallvec![engine.interner.id_for("b")],
            ix: 0,
        });

        let winner_bigram = BigramId { left, right };
        let merged_ids = merged_lexeme_ids_for_bigram(winner_bigram, &mut engine.lexeme_store);
        let winner = winner_from_bigram_with_data(
            winner_bigram,
            merged_ids[0],
            merged_ids.len(),
            &engine.bigrams,
        );
        assert_eq!(winner.bigram_locations, vec![(0, 0), (0, 2)]);

        let touched = merge_winner(
            &winner,
            &merged_ids,
            &mut engine.lexemes,
            &engine.lexeme_store,
            &mut engine.bigrams,
        );

        let merged_bigram = BigramId {
            left: merged_ids[0],
            right: merged_ids[0],
        };
        let bridge_bigram = BigramId {
            left: right,
            right: left,
        };

        assert_eq!(
            touched,
            FxHashSet::from_iter([winner_bigram, bridge_bigram, merged_bigram])
        );
        assert_eq!(
            engine.lexemes.locations_to_lexemes[0],
            vec![merged_ids[0], merged_ids[1], merged_ids[0], merged_ids[1]]
        );
        assert_eq!(
            engine
                .lexemes
                .lexemes_to_locations
                .get(&merged_ids[0])
                .cloned(),
            Some(FxHashSet::from_iter([(0, 0), (0, 2)]))
        );
        assert_eq!(
            engine
                .lexemes
                .lexemes_to_locations
                .get(&merged_ids[1])
                .cloned(),
            Some(FxHashSet::from_iter([(0, 1), (0, 3)]))
        );

        assert_eq!(
            engine.bigrams.bigrams_to_freqs,
            FxHashMap::from_iter([(merged_bigram, 1)])
        );

        let mut locations = engine.bigrams.locations_for_bigram(merged_bigram);
        locations.sort_unstable();
        assert_eq!(locations, vec![(0, 0)]);
        assert_eq!(engine.bigrams.total_bigram_count, 1);
        assert_eq!(
            engine.bigrams.left_lex_freqs,
            FxHashMap::from_iter([(merged_ids[0], 1)])
        );
        assert_eq!(
            engine.bigrams.right_lex_freqs,
            FxHashMap::from_iter([(merged_ids[0], 1)])
        );
    }

    #[test]
    fn min_score_blocks_low_score_winner() {
        let mut engine = build_engine(
            vec!["a b c"],
            SelectionMethod::Frequency,
            0,
            DEFAULT_RESCORE_INTERVAL,
        );
        let status = engine.step_internal(Some(10.0));
        let StepStatus::BelowMinScore(score) = status else {
            panic!("expected below-min-score status");
        };
        assert_eq!(score, 1.0);
    }

    #[test]
    fn rescore_interval_one_forces_full_rescore_every_step() {
        let mut engine = build_engine(vec!["a b a c a b"], SelectionMethod::Npmi, 0, 1);

        let initial_count = engine.bigrams.bigrams_to_freqs.len();
        assert_eq!(engine.candidate_scores.len(), initial_count);

        let StepStatus::Winner(_) = engine.step_internal(None) else {
            panic!("expected winner");
        };

        engine.refresh_candidate_state(false);

        let eligible = engine
            .bigrams
            .bigrams_to_freqs
            .iter()
            .filter(|(_, freq)| **freq >= engine.min_count)
            .count();
        assert_eq!(engine.candidate_scores.len(), eligible);
    }

    #[test]
    fn candidate_heap_skips_stale_frequency_entries() {
        let mut engine = build_engine(
            vec!["a b a b"],
            SelectionMethod::Frequency,
            0,
            DEFAULT_RESCORE_INTERVAL,
        );

        let bigram = *engine
            .bigrams
            .bigrams_to_freqs
            .keys()
            .next()
            .expect("expected at least one bigram");

        engine.candidate_heap.push(HeapEntry {
            score: 9999.0,
            frequency: 9999,
            merged_word: merged_word_ids(&engine.lexeme_store, bigram),
            bigram,
            generation: 0,
        });

        let Some((_winner, candidate)) = engine.select_candidate() else {
            panic!("expected candidate")
        };
        assert!(candidate.frequency < 9999);
    }

    #[test]
    fn clean_bigram_locations_filters_overlaps_per_line() {
        let locations = vec![(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 4)];
        let cleaned = clean_bigram_locations(locations, 2);
        assert_eq!(cleaned, vec![(0, 0), (0, 2), (1, 0), (1, 4)]);
    }

    #[test]
    fn clean_bigram_locations_is_line_local() {
        let locations = vec![(0, 0), (0, 3), (1, 1), (1, 2)];
        let cleaned = clean_bigram_locations(locations, 3);
        assert_eq!(cleaned, vec![(0, 0), (0, 3), (1, 1)]);
    }

    #[test]
    fn merged_lexeme_ids_for_bigram_reuses_existing_ids() {
        let mut engine = build_engine(
            vec!["a b a b"],
            SelectionMethod::Frequency,
            0,
            DEFAULT_RESCORE_INTERVAL,
        );
        let left = engine.lexeme_store.intern(Lexeme {
            word: smallvec![engine.interner.id_for("a")],
            ix: 0,
        });
        let right = engine.lexeme_store.intern(Lexeme {
            word: smallvec![engine.interner.id_for("b")],
            ix: 0,
        });
        let bigram = BigramId { left, right };

        let pre_len = engine.lexeme_store.id_to_lexeme.len();
        let first_ids = merged_lexeme_ids_for_bigram(bigram, &mut engine.lexeme_store);
        let after_first_len = engine.lexeme_store.id_to_lexeme.len();
        let second_ids = merged_lexeme_ids_for_bigram(bigram, &mut engine.lexeme_store);
        let after_second_len = engine.lexeme_store.id_to_lexeme.len();

        assert_eq!(first_ids, second_ids);
        assert_eq!(after_first_len, pre_len + first_ids.len());
        assert_eq!(after_second_len, after_first_len);
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(48))]

        #[test]
        fn total_root_count_changes_only_on_winner(corpus in "[a-c ]{5,30}") {
            let mut engine = build_engine(
                vec![corpus.as_str()],
                SelectionMethod::Frequency,
                0,
                DEFAULT_RESCORE_INTERVAL,
            );

            let before = total_root_count(&engine);
            let status = engine.step_internal(None);
            let after = total_root_count(&engine);

            match status {
                StepStatus::Winner(_) => prop_assert!(after < before),
                StepStatus::NoCandidate | StepStatus::BelowMinScore(_) => {
                    prop_assert_eq!(after, before)
                }
            }
        }
    }
}
