use indexmap::IndexMap;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::cmp::Reverse;
use std::collections::{HashMap, HashSet};

const SMALL: f64 = 1e-10;
const SCORE_ATOL: f64 = 1e-12;
const SCORE_RTOL: f64 = 1e-12;

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
enum TieBreaker {
    Deterministic,
    LegacyFirstSeen,
}

impl TieBreaker {
    fn parse(value: &str) -> PyResult<Self> {
        match value {
            "deterministic" => Ok(Self::Deterministic),
            "legacy_first_seen" => Ok(Self::LegacyFirstSeen),
            _ => Err(PyValueError::new_err(format!(
                "Invalid tie_breaker {value:?}. Expected one of: 'deterministic', 'legacy_first_seen'."
            ))),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
struct Lexeme {
    word: Vec<String>,
    ix: usize,
}

#[derive(Clone, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
struct Bigram {
    left: Lexeme,
    right: Lexeme,
}

#[derive(Default)]
struct LexemeData {
    lexemes_to_locations: HashMap<Lexeme, HashSet<(usize, usize)>>,
    locations_to_lexemes: Vec<Vec<Lexeme>>,
    lexemes_to_freqs: HashMap<Lexeme, i64>,
}

impl LexemeData {
    fn from_corpus(corpus: &[Vec<String>]) -> Self {
        let mut lexeme_data = Self::default();
        for (line_ix, tokens) in corpus.iter().enumerate() {
            let mut line_lexemes = Vec::with_capacity(tokens.len());
            for (word_ix, word) in tokens.iter().enumerate() {
                let lexeme = Lexeme {
                    word: vec![word.clone()],
                    ix: 0,
                };
                let loc = (line_ix, word_ix);
                lexeme_data
                    .lexemes_to_locations
                    .entry(lexeme.clone())
                    .or_default()
                    .insert(loc);
                line_lexemes.push(lexeme);
            }
            lexeme_data.locations_to_lexemes.push(line_lexemes);
        }

        lexeme_data.lexemes_to_freqs = lexeme_data
            .lexemes_to_locations
            .iter()
            .filter(|(lexeme, _)| lexeme.ix == 0)
            .map(|(lexeme, locs)| (lexeme.clone(), locs.len() as i64))
            .collect();

        lexeme_data
    }

    fn corpus_length(&self) -> usize {
        self.locations_to_lexemes.len()
    }

    fn locations_to_root_lexemes(&self, line_ix: usize) -> Vec<(usize, Lexeme)> {
        self.locations_to_lexemes[line_ix]
            .iter()
            .enumerate()
            .filter(|(_, lexeme)| lexeme.ix == 0)
            .map(|(ix, lexeme)| (ix, lexeme.clone()))
            .collect()
    }
}

#[derive(Default)]
struct BigramData {
    bigrams_to_freqs: IndexMap<Bigram, i64>,
    bigrams_to_locations: HashMap<Bigram, HashSet<(usize, usize)>>,
    left_lex_freqs: HashMap<Lexeme, i64>,
    right_lex_freqs: HashMap<Lexeme, i64>,
}

impl BigramData {
    fn from_lexemes(lexeme_data: &LexemeData) -> Self {
        let mut bigram_data = Self::default();
        for line_ix in 0..lexeme_data.corpus_length() {
            let line_lexeme_data = lexeme_data.locations_to_root_lexemes(line_ix);
            let mut line_bigrams = Vec::new();
            for pair in line_lexeme_data.windows(2) {
                let left_ix = pair[0].0;
                let left = pair[0].1.clone();
                let right = pair[1].1.clone();
                let bigram = Bigram { left, right };
                let location = (line_ix, left_ix);
                bigram_data
                    .bigrams_to_locations
                    .entry(bigram.clone())
                    .or_default()
                    .insert(location);
                line_bigrams.push(bigram);
            }
            bigram_data.batch_add_bigrams(&line_bigrams);
        }
        bigram_data
    }

    fn batch_add_bigrams(&mut self, bigram_locations: &[Bigram]) {
        for bigram in bigram_locations {
            *self.left_lex_freqs.entry(bigram.left.clone()).or_insert(0) += 1;
            *self
                .right_lex_freqs
                .entry(bigram.right.clone())
                .or_insert(0) += 1;
            *self.bigrams_to_freqs.entry(bigram.clone()).or_insert(0) += 1;
        }
    }
}

#[derive(Clone)]
struct WinnerInfo {
    bigram: Bigram,
    merged_lexeme: Lexeme,
    bigram_locations: Vec<(usize, usize)>,
}

impl WinnerInfo {
    fn from_bigram_with_data(bigram: &Bigram, bigram_data: &BigramData) -> Self {
        let mut all_words = Vec::with_capacity(bigram.left.word.len() + bigram.right.word.len());
        all_words.extend_from_slice(&bigram.left.word);
        all_words.extend_from_slice(&bigram.right.word);

        let mut locations = bigram_data
            .bigrams_to_locations
            .get(bigram)
            .cloned()
            .unwrap_or_default()
            .into_iter()
            .collect::<Vec<_>>();
        locations.sort_unstable();

        Self {
            bigram: bigram.clone(),
            merged_lexeme: Lexeme {
                word: all_words,
                ix: 0,
            },
            bigram_locations: locations,
        }
    }

    fn n_lexemes(&self) -> usize {
        self.merged_lexeme.word.len()
    }

    fn cleaned_bigram_locations(&self) -> Vec<(usize, usize)> {
        let mut clean_locations = Vec::new();
        let mut ix = 0;
        while ix < self.bigram_locations.len() {
            let line = self.bigram_locations[ix].0;
            let mut token_ix = Vec::new();
            while ix < self.bigram_locations.len() && self.bigram_locations[ix].0 == line {
                token_ix.push(self.bigram_locations[ix].1);
                ix += 1;
            }

            let mut exclude_tokens: HashSet<usize> = HashSet::new();
            for token in token_ix.iter().copied() {
                if exclude_tokens.contains(&token) {
                    continue;
                }
                for candidate in token_ix.iter().copied() {
                    if token <= candidate && candidate < token + self.n_lexemes() {
                        exclude_tokens.insert(candidate);
                    }
                }
                clean_locations.push((line, token));
            }
        }

        clean_locations
    }
}

#[derive(Clone)]
struct ScoredBigram {
    bigram: Bigram,
    score: f64,
    frequency: i64,
}

impl ScoredBigram {
    fn merged_word(&self) -> Vec<String> {
        let mut merged =
            Vec::with_capacity(self.bigram.left.word.len() + self.bigram.right.word.len());
        merged.extend_from_slice(&self.bigram.left.word);
        merged.extend_from_slice(&self.bigram.right.word);
        merged
    }
}

struct BigramCandidateData {
    bigram_index: Vec<Bigram>,
    bigram_freq_array: Vec<i64>,
    el1_freq_array: Vec<i64>,
    el2_freq_array: Vec<i64>,
    total_bigram_count: i64,
}

impl BigramCandidateData {
    fn from_bigram_data(bigram_data: &BigramData, min_count: i64) -> Self {
        let total_bigram_count = bigram_data.bigrams_to_freqs.values().sum::<i64>();
        let candidate_items = bigram_data
            .bigrams_to_freqs
            .iter()
            .filter(|(_, freq)| **freq >= min_count)
            .collect::<Vec<_>>();

        if candidate_items.is_empty() {
            return Self {
                bigram_index: Vec::new(),
                bigram_freq_array: Vec::new(),
                el1_freq_array: Vec::new(),
                el2_freq_array: Vec::new(),
                total_bigram_count,
            };
        }

        let mut bigram_index = Vec::with_capacity(candidate_items.len());
        let mut bigram_freq_array = Vec::with_capacity(candidate_items.len());
        let mut el1_freq_array = Vec::with_capacity(candidate_items.len());
        let mut el2_freq_array = Vec::with_capacity(candidate_items.len());

        for (bigram, freq) in candidate_items {
            bigram_index.push((*bigram).clone());
            bigram_freq_array.push(*freq);
            el1_freq_array.push(*bigram_data.left_lex_freqs.get(&bigram.left).unwrap_or(&0));
            el2_freq_array.push(*bigram_data.right_lex_freqs.get(&bigram.right).unwrap_or(&0));
        }

        Self {
            bigram_index,
            bigram_freq_array,
            el1_freq_array,
            el2_freq_array,
            total_bigram_count,
        }
    }
}

fn safe_ll_term(observed: f64, expected: f64) -> f64 {
    if observed > 0.0 {
        observed * (((observed / (expected + SMALL)) + SMALL).ln())
    } else {
        0.0
    }
}

fn calculate_npmi(data: &BigramCandidateData) -> Vec<f64> {
    if data.total_bigram_count == 0 {
        return Vec::new();
    }

    let total = data.total_bigram_count as f64;
    let mut scores = Vec::with_capacity(data.bigram_freq_array.len());
    for i in 0..data.bigram_freq_array.len() {
        let prob_ab = data.bigram_freq_array[i] as f64 / total;
        let prob_a = data.el1_freq_array[i] as f64 / total;
        let prob_b = data.el2_freq_array[i] as f64 / total;

        let numerator = (prob_ab / (prob_a * prob_b)).ln();
        let denominator = -(prob_ab.ln());
        let npmi = if denominator > 0.0 {
            numerator / denominator
        } else {
            f64::NAN
        };

        let perfect_association =
            is_close_default(denominator, 0.0) && is_close_default(numerator, 0.0);
        if perfect_association {
            scores.push(1.0);
        } else {
            scores.push(npmi);
        }
    }

    scores
}

fn calculate_log_likelihood(data: &BigramCandidateData) -> Vec<f64> {
    if data.total_bigram_count == 0 {
        return Vec::new();
    }

    let total = data.total_bigram_count as f64;
    let mut scores = Vec::with_capacity(data.bigram_freq_array.len());

    for i in 0..data.bigram_freq_array.len() {
        let obs_a = data.bigram_freq_array[i] as f64;
        let obs_b = data.el1_freq_array[i] as f64 - obs_a;
        let obs_c = data.el2_freq_array[i] as f64 - obs_a;
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

        let log_likelihood = 2.0 * (ll_a + ll_b + ll_c + ll_d);
        if obs_a > exp_a {
            scores.push(log_likelihood);
        } else {
            scores.push(-log_likelihood);
        }
    }

    scores
}

fn coerce_scores(scores: Vec<f64>) -> Vec<f64> {
    scores
        .into_iter()
        .map(|score| {
            if score.is_finite() {
                score
            } else {
                f64::NEG_INFINITY
            }
        })
        .collect()
}

fn as_scored_bigrams(data: &BigramCandidateData, scores: Vec<f64>) -> Vec<ScoredBigram> {
    let safe_scores = coerce_scores(scores);
    if safe_scores.is_empty() || safe_scores.iter().all(|score| *score == f64::NEG_INFINITY) {
        return Vec::new();
    }

    data.bigram_index
        .iter()
        .zip(safe_scores.iter())
        .zip(data.bigram_freq_array.iter())
        .map(|((bigram, score), freq)| ScoredBigram {
            bigram: bigram.clone(),
            score: *score,
            frequency: *freq,
        })
        .collect()
}

fn calculate_candidates_log_likelihood(
    bigram_data: &BigramData,
    min_count: i64,
) -> Vec<ScoredBigram> {
    let data = BigramCandidateData::from_bigram_data(bigram_data, min_count);
    let scores = calculate_log_likelihood(&data);
    as_scored_bigrams(&data, scores)
}

fn calculate_candidates_npmi(bigram_data: &BigramData, min_count: i64) -> Vec<ScoredBigram> {
    let data = BigramCandidateData::from_bigram_data(bigram_data, min_count);
    let scores = calculate_npmi(&data);
    as_scored_bigrams(&data, scores)
}

fn calculate_candidates_frequency(bigram_data: &BigramData, min_count: i64) -> Vec<ScoredBigram> {
    bigram_data
        .bigrams_to_freqs
        .iter()
        .filter(|(_, freq)| **freq >= min_count)
        .map(|(bigram, freq)| ScoredBigram {
            bigram: bigram.clone(),
            score: *freq as f64,
            frequency: *freq,
        })
        .collect()
}

fn scores_close(a: f64, b: f64) -> bool {
    (a - b).abs() <= (SCORE_ATOL + SCORE_RTOL * b.abs())
}

fn is_close_default(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-8 + (1e-5 * b.abs())
}

fn select_candidate(candidates: &[ScoredBigram], tie_breaker: TieBreaker) -> Option<ScoredBigram> {
    if candidates.is_empty() {
        return None;
    }

    let max_score = candidates
        .iter()
        .map(|candidate| candidate.score)
        .fold(f64::NEG_INFINITY, f64::max);

    if tie_breaker == TieBreaker::LegacyFirstSeen {
        return candidates
            .iter()
            .find(|candidate| scores_close(candidate.score, max_score))
            .cloned();
    }

    let mut best_candidate: Option<ScoredBigram> = None;
    let mut best_key: Option<(Reverse<i64>, Vec<String>)> = None;

    for candidate in candidates {
        if !scores_close(candidate.score, max_score) {
            continue;
        }
        let candidate_key = (Reverse(candidate.frequency), candidate.merged_word());
        match (&best_candidate, &best_key) {
            (None, None) => {
                best_candidate = Some(candidate.clone());
                best_key = Some(candidate_key);
            }
            (Some(_), Some(key)) => {
                if candidate_key < *key {
                    best_candidate = Some(candidate.clone());
                    best_key = Some(candidate_key);
                }
            }
            _ => {}
        }
    }

    best_candidate
}

fn adjacent_bigrams(lexemes: &[Lexeme]) -> Vec<Bigram> {
    lexemes
        .windows(2)
        .map(|pair| Bigram {
            left: pair[0].clone(),
            right: pair[1].clone(),
        })
        .collect()
}

fn merge_winner(
    winner: &WinnerInfo,
    clean_locations: &[(usize, usize)],
    lexeme_data: &mut LexemeData,
    bigram_data: &mut BigramData,
) {
    let bigram_lines = clean_locations
        .iter()
        .map(|(line_ix, _)| *line_ix)
        .collect::<HashSet<_>>();

    let mut touched_lexemes = HashSet::new();
    touched_lexemes.insert(winner.merged_lexeme.clone());
    touched_lexemes.insert(winner.bigram.left.clone());
    touched_lexemes.insert(winner.bigram.right.clone());

    let mut touched_bigrams = HashSet::new();
    touched_bigrams.insert(winner.bigram.clone());

    let mut old_bigrams_lookup: HashMap<usize, Vec<(usize, Lexeme)>> = HashMap::new();
    for line_ix in bigram_lines {
        old_bigrams_lookup.insert(line_ix, lexeme_data.locations_to_root_lexemes(line_ix));
    }

    for (line_ix, word_ix) in clean_locations.iter().copied() {
        for lexeme_index in 0..winner.n_lexemes() {
            let pos = word_ix + lexeme_index;
            let old_lexeme = lexeme_data.locations_to_lexemes[line_ix][pos].clone();
            touched_lexemes.insert(old_lexeme.clone());

            let lexeme = Lexeme {
                word: winner.merged_lexeme.word.clone(),
                ix: lexeme_index,
            };
            lexeme_data.locations_to_lexemes[line_ix][pos] = lexeme.clone();

            if let Some(locations) = lexeme_data.lexemes_to_locations.get_mut(&old_lexeme) {
                locations.remove(&(line_ix, pos));
            }
            lexeme_data
                .lexemes_to_locations
                .entry(lexeme)
                .or_default()
                .insert((line_ix, pos));
        }
    }

    for (line_ix, old_root_items) in old_bigrams_lookup {
        let old_root_lexemes = old_root_items
            .iter()
            .map(|(_, lexeme)| lexeme.clone())
            .collect::<Vec<_>>();
        let old_bigrams = adjacent_bigrams(&old_root_lexemes);
        for bigram in old_bigrams.iter().cloned() {
            touched_bigrams.insert(bigram);
        }

        let new_root_items = lexeme_data.locations_to_root_lexemes(line_ix);
        let new_root_lexemes = new_root_items
            .iter()
            .map(|(_, lexeme)| lexeme.clone())
            .collect::<Vec<_>>();
        let new_bigrams = adjacent_bigrams(&new_root_lexemes);
        for bigram in new_bigrams.iter().cloned() {
            touched_bigrams.insert(bigram);
        }

        for bigram in &new_bigrams {
            *bigram_data
                .bigrams_to_freqs
                .entry(bigram.clone())
                .or_insert(0) += 1;
            *bigram_data
                .left_lex_freqs
                .entry(bigram.left.clone())
                .or_insert(0) += 1;
            *bigram_data
                .right_lex_freqs
                .entry(bigram.right.clone())
                .or_insert(0) += 1;
        }

        for bigram in &old_bigrams {
            *bigram_data
                .bigrams_to_freqs
                .entry(bigram.clone())
                .or_insert(0) -= 1;
            *bigram_data
                .left_lex_freqs
                .entry(bigram.left.clone())
                .or_insert(0) -= 1;
            *bigram_data
                .right_lex_freqs
                .entry(bigram.right.clone())
                .or_insert(0) -= 1;
        }

        for pair in old_root_items.windows(2) {
            let left_ix = pair[0].0;
            let left = pair[0].1.clone();
            let right = pair[1].1.clone();
            let bigram = Bigram { left, right };
            let location = (line_ix, left_ix);
            if let Some(locations) = bigram_data.bigrams_to_locations.get_mut(&bigram) {
                locations.remove(&location);
            }
        }

        for pair in new_root_items.windows(2) {
            let left_ix = pair[0].0;
            let left = pair[0].1.clone();
            let right = pair[1].1.clone();
            let bigram = Bigram { left, right };
            let location = (line_ix, left_ix);
            bigram_data
                .bigrams_to_locations
                .entry(bigram)
                .or_default()
                .insert(location);
        }
    }

    let merge_token_count = clean_locations.len() as i64;
    lexeme_data
        .lexemes_to_freqs
        .insert(winner.merged_lexeme.clone(), merge_token_count);

    if let Some(el1_freq) = lexeme_data.lexemes_to_freqs.get_mut(&winner.bigram.left) {
        *el1_freq -= merge_token_count;
    }
    if let Some(el2_freq) = lexeme_data.lexemes_to_freqs.get_mut(&winner.bigram.right) {
        *el2_freq -= merge_token_count;
    }

    for lexeme in touched_lexemes {
        let remove_freq = lexeme_data
            .lexemes_to_freqs
            .get(&lexeme)
            .map(|freq| *freq <= 0)
            .unwrap_or(false);
        if remove_freq {
            lexeme_data.lexemes_to_freqs.remove(&lexeme);
        }

        let remove_locations = lexeme_data
            .lexemes_to_locations
            .get(&lexeme)
            .map(|locations| locations.is_empty())
            .unwrap_or(true);
        if remove_locations {
            lexeme_data.lexemes_to_locations.remove(&lexeme);
        }
    }

    let mut touched_lr_lexemes = HashSet::new();
    for bigram in touched_bigrams {
        touched_lr_lexemes.insert(bigram.left.clone());
        touched_lr_lexemes.insert(bigram.right.clone());

        let remove_freq = bigram_data
            .bigrams_to_freqs
            .get(&bigram)
            .map(|freq| *freq <= 0)
            .unwrap_or(false);
        if remove_freq {
            bigram_data.bigrams_to_freqs.shift_remove(&bigram);
        }

        let remove_locations = bigram_data
            .bigrams_to_locations
            .get(&bigram)
            .map(|locations| locations.is_empty())
            .unwrap_or(true);
        if remove_locations {
            bigram_data.bigrams_to_locations.remove(&bigram);
        }
    }

    for lexeme in touched_lr_lexemes {
        let remove_left = bigram_data
            .left_lex_freqs
            .get(&lexeme)
            .map(|freq| *freq <= 0)
            .unwrap_or(false);
        if remove_left {
            bigram_data.left_lex_freqs.remove(&lexeme);
        }

        let remove_right = bigram_data
            .right_lex_freqs
            .get(&lexeme)
            .map(|freq| *freq <= 0)
            .unwrap_or(false);
        if remove_right {
            bigram_data.right_lex_freqs.remove(&lexeme);
        }
    }

    debug_assert!(!bigram_data.bigrams_to_freqs.contains_key(&winner.bigram));
}

#[derive(Clone)]
struct StepData {
    score: f64,
    winner: WinnerInfo,
    clean_locations: Vec<(usize, usize)>,
}

enum StepStatus {
    Winner(StepData),
    NoCandidate,
    BelowMinScore(f64),
}

#[pyclass]
struct Engine {
    lexemes: LexemeData,
    bigrams: BigramData,
    method: SelectionMethod,
    min_count: i64,
    tie_breaker: TieBreaker,
}

impl Engine {
    fn calculate_candidates(&self) -> Vec<ScoredBigram> {
        match self.method {
            SelectionMethod::Frequency => {
                calculate_candidates_frequency(&self.bigrams, self.min_count)
            }
            SelectionMethod::LogLikelihood => {
                calculate_candidates_log_likelihood(&self.bigrams, self.min_count)
            }
            SelectionMethod::Npmi => calculate_candidates_npmi(&self.bigrams, self.min_count),
        }
    }

    fn step_internal(&mut self, min_score: Option<f64>) -> StepStatus {
        let candidates = self.calculate_candidates();
        let Some(selected) = select_candidate(&candidates, self.tie_breaker) else {
            return StepStatus::NoCandidate;
        };

        if let Some(score_threshold) = min_score {
            if selected.score < score_threshold {
                return StepStatus::BelowMinScore(selected.score);
            }
        }

        let winner = WinnerInfo::from_bigram_with_data(&selected.bigram, &self.bigrams);
        let clean_locations = winner.cleaned_bigram_locations();
        merge_winner(
            &winner,
            &clean_locations,
            &mut self.lexemes,
            &mut self.bigrams,
        );

        StepStatus::Winner(StepData {
            score: selected.score,
            winner,
            clean_locations,
        })
    }
}

type StepPayload = (
    f64,
    Vec<String>,
    usize,
    Vec<String>,
    usize,
    Vec<String>,
    usize,
    Vec<(usize, usize)>,
    Vec<(usize, usize)>,
);

type StepOutcome = (String, Option<StepPayload>, Option<f64>);

#[pymethods]
impl Engine {
    #[new]
    fn new(
        corpus: Vec<Vec<String>>,
        method: &str,
        min_count: usize,
        tie_breaker: &str,
    ) -> PyResult<Self> {
        let method = SelectionMethod::parse(method)?;
        let tie_breaker = TieBreaker::parse(tie_breaker)?;
        let lexemes = LexemeData::from_corpus(&corpus);
        let bigrams = BigramData::from_lexemes(&lexemes);

        Ok(Self {
            lexemes,
            bigrams,
            method,
            min_count: min_count as i64,
            tie_breaker,
        })
    }

    fn corpus_length(&self) -> usize {
        self.lexemes.corpus_length()
    }

    #[pyo3(signature = (min_score=None))]
    fn step(&mut self, py: Python<'_>, min_score: Option<f64>) -> StepOutcome {
        let step_result = py.allow_threads(|| self.step_internal(min_score));
        match step_result {
            StepStatus::NoCandidate => ("no_candidate".to_string(), None, None),
            StepStatus::BelowMinScore(score) => ("below_min_score".to_string(), None, Some(score)),
            StepStatus::Winner(step_data) => {
                let payload = (
                    step_data.score,
                    step_data.winner.bigram.left.word,
                    step_data.winner.bigram.left.ix,
                    step_data.winner.bigram.right.word,
                    step_data.winner.bigram.right.ix,
                    step_data.winner.merged_lexeme.word,
                    step_data.winner.merged_lexeme.ix,
                    step_data.winner.bigram_locations,
                    step_data.clean_locations,
                );
                ("winner".to_string(), Some(payload), Some(step_data.score))
            }
        }
    }
}

// This module currently assumes GIL use for Python interaction.
#[pymodule(gil_used = true)]
fn _core(_py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<Engine>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_engine(corpus: Vec<Vec<&str>>, method: SelectionMethod, min_count: i64) -> Engine {
        let corpus = corpus
            .into_iter()
            .map(|line| line.into_iter().map(str::to_string).collect::<Vec<_>>())
            .collect::<Vec<_>>();

        let lexemes = LexemeData::from_corpus(&corpus);
        let bigrams = BigramData::from_lexemes(&lexemes);
        Engine {
            lexemes,
            bigrams,
            method,
            min_count,
            tie_breaker: TieBreaker::Deterministic,
        }
    }

    #[test]
    fn deterministic_tie_break_prefers_frequency_then_lexicographic() {
        let mut engine = build_engine(
            vec![
                vec!["a", "d"],
                vec!["a", "c"],
                vec!["a", "b"],
                vec!["a", "b"],
            ],
            SelectionMethod::Frequency,
            0,
        );
        let StepStatus::Winner(step) = engine.step_internal(None) else {
            panic!("expected winner");
        };
        assert_eq!(step.winner.merged_lexeme.word, vec!["a", "b"]);
    }

    #[test]
    fn legacy_tie_break_is_first_seen() {
        let corpus = vec![vec!["c", "d"], vec!["a", "b"]]
            .into_iter()
            .map(|line| line.into_iter().map(str::to_string).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        let lexemes = LexemeData::from_corpus(&corpus);
        let bigrams = BigramData::from_lexemes(&lexemes);
        let mut engine = Engine {
            lexemes,
            bigrams,
            method: SelectionMethod::Frequency,
            min_count: 0,
            tie_breaker: TieBreaker::LegacyFirstSeen,
        };

        let StepStatus::Winner(step) = engine.step_internal(None) else {
            panic!("expected winner");
        };
        assert_eq!(step.winner.merged_lexeme.word, vec!["c", "d"]);
    }

    #[test]
    fn min_score_blocks_low_score_winner() {
        let mut engine = build_engine(vec![vec!["a", "b", "c"]], SelectionMethod::Frequency, 0);
        let status = engine.step_internal(Some(10.0));
        let StepStatus::BelowMinScore(score) = status else {
            panic!("expected below-min-score status");
        };
        assert_eq!(score, 1.0);
    }

    #[test]
    fn merge_invariants_hold_on_small_corpus() {
        let mut engine = build_engine(
            vec![
                vec!["a", "a", "a", "a"],
                vec!["c", "a", "b", "a", "b", "a", "b", "d"],
            ],
            SelectionMethod::Frequency,
            0,
        );

        for _ in 0..8 {
            let status = engine.step_internal(None);
            let StepStatus::Winner(step) = status else {
                break;
            };
            assert!(!engine
                .bigrams
                .bigrams_to_freqs
                .contains_key(&step.winner.bigram));
            assert!(engine
                .lexemes
                .lexemes_to_freqs
                .values()
                .all(|freq| *freq > 0));
            assert!(engine
                .bigrams
                .bigrams_to_freqs
                .values()
                .all(|freq| *freq > 0));
            assert!(engine
                .lexemes
                .lexemes_to_locations
                .values()
                .all(|locations| !locations.is_empty()));
            assert!(engine
                .bigrams
                .bigrams_to_locations
                .values()
                .all(|locations| !locations.is_empty()));
        }
    }

    #[test]
    fn frequency_respects_min_count() {
        let mut engine = build_engine(
            vec![vec!["a", "b"], vec!["a", "c"]],
            SelectionMethod::Frequency,
            2,
        );
        let status = engine.step_internal(None);
        assert!(matches!(status, StepStatus::NoCandidate));
    }

    #[test]
    fn empty_or_single_token_corpus_has_no_candidate() {
        let mut empty_engine = build_engine(vec![], SelectionMethod::Frequency, 0);
        assert!(matches!(
            empty_engine.step_internal(None),
            StepStatus::NoCandidate
        ));

        let mut single_engine = build_engine(vec![vec!["a"]], SelectionMethod::Frequency, 0);
        assert!(matches!(
            single_engine.step_internal(None),
            StepStatus::NoCandidate
        ));
    }

    #[test]
    fn consecutive_merge_path_is_greedy() {
        let mut engine = build_engine(
            vec![vec!["a", "a", "a", "a"]],
            SelectionMethod::Frequency,
            0,
        );

        let StepStatus::Winner(first) = engine.step_internal(None) else {
            panic!("expected first winner");
        };
        assert_eq!(first.clean_locations.len(), 2);

        let StepStatus::Winner(second) = engine.step_internal(None) else {
            panic!("expected second winner");
        };
        assert_eq!(second.winner.merged_lexeme.word, vec!["a", "a", "a", "a"]);
    }

    #[test]
    fn scoring_methods_select_winners_on_small_corpus() {
        let corpus = vec![vec!["a", "b", "a", "b", "c"], vec!["a", "b", "d", "e"]];
        for method in [
            SelectionMethod::Frequency,
            SelectionMethod::LogLikelihood,
            SelectionMethod::Npmi,
        ] {
            let mut engine = build_engine(corpus.clone(), method, 0);
            assert!(matches!(engine.step_internal(None), StepStatus::Winner(_)));
        }
    }
}
