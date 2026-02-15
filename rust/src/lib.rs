use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;
use sentencex::segment;
use smallvec::{smallvec, SmallVec};
use std::cmp::Reverse;
use std::collections::{HashMap, HashSet};

const SMALL: f64 = 1e-10;
const SCORE_ATOL: f64 = 1e-12;
const SCORE_RTOL: f64 = 1e-12;
const FULL_RESCORE_INTERVAL: usize = 25;

type TokenId = u32;
type Location = (usize, usize);

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
    str_to_id: HashMap<String, TokenId>,
    id_to_str: Vec<String>,
}

impl Interner {
    fn from_documents(documents: &[String], splitter: Splitter<'_>) -> (Self, Vec<Vec<TokenId>>) {
        let mut uniq = HashSet::new();
        for document in documents {
            match splitter {
                Splitter::Delimiter(Some(delim)) => {
                    for segment in document.split(delim) {
                        for token in segment.split_whitespace() {
                            uniq.insert(token.to_string());
                        }
                    }
                }
                Splitter::Delimiter(None) => {
                    for token in document.split_whitespace() {
                        uniq.insert(token.to_string());
                    }
                }
                Splitter::Sentencex { language } => {
                    for sentence in segment(language, document) {
                        for token in sentence.split_whitespace() {
                            uniq.insert(token.to_string());
                        }
                    }
                }
            }
        }
        let mut sorted = uniq.into_iter().collect::<Vec<_>>();
        sorted.sort_unstable();

        let mut interner = Self::default();
        for token in sorted {
            let id = interner.id_to_str.len() as TokenId;
            interner.str_to_id.insert(token.clone(), id);
            interner.id_to_str.push(token);
        }

        let mut corpus_ids = Vec::new();
        for document in documents {
            match splitter {
                Splitter::Delimiter(Some(delim)) => {
                    for segment in document.split(delim) {
                        let tokens = segment
                            .split_whitespace()
                            .map(|token| interner.id_for(token))
                            .collect::<Vec<_>>();
                        if !tokens.is_empty() {
                            corpus_ids.push(tokens);
                        }
                    }
                }
                Splitter::Delimiter(None) => {
                    let tokens = document
                        .split_whitespace()
                        .map(|token| interner.id_for(token))
                        .collect::<Vec<_>>();
                    if !tokens.is_empty() {
                        corpus_ids.push(tokens);
                    }
                }
                Splitter::Sentencex { language } => {
                    for sentence in segment(language, document) {
                        let tokens = sentence
                            .split_whitespace()
                            .map(|token| interner.id_for(token))
                            .collect::<Vec<_>>();
                        if !tokens.is_empty() {
                            corpus_ids.push(tokens);
                        }
                    }
                }
            }
        }

        (interner, corpus_ids)
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

#[derive(Clone, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
struct Bigram {
    left: Lexeme,
    right: Lexeme,
}

#[derive(Default)]
struct LexemeData {
    lexemes_to_locations: HashMap<Lexeme, HashSet<Location>>,
    locations_to_lexemes: Vec<Vec<Lexeme>>,
    lexemes_to_freqs: HashMap<Lexeme, i64>,
}

impl LexemeData {
    fn from_corpus(corpus: &[Vec<TokenId>]) -> Self {
        let mut lexeme_data = Self::default();
        for (line_ix, tokens) in corpus.iter().enumerate() {
            let mut line_lexemes = Vec::with_capacity(tokens.len());
            for (word_ix, word) in tokens.iter().enumerate() {
                let lexeme = Lexeme {
                    word: smallvec![*word],
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

    fn root_items_for_line(&self, line_ix: usize) -> Vec<(usize, Lexeme)> {
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
    bigrams_to_freqs: HashMap<Bigram, i64>,
    bigrams_first_seen_order: Option<HashMap<Bigram, usize>>,
    next_bigram_order: usize,
    total_bigram_count: i64,
    bigrams_to_locations: HashMap<Bigram, HashSet<Location>>,
    left_lex_freqs: HashMap<Lexeme, i64>,
    right_lex_freqs: HashMap<Lexeme, i64>,
}

impl BigramData {
    fn from_lexemes(lexeme_data: &LexemeData, track_first_seen: bool) -> Self {
        let mut bigram_data = Self {
            bigrams_first_seen_order: track_first_seen.then(HashMap::new),
            ..Self::default()
        };
        for line_ix in 0..lexeme_data.corpus_length() {
            let root_items = lexeme_data.root_items_for_line(line_ix);
            for pair in root_items.windows(2) {
                let bigram = Bigram {
                    left: pair[0].1.clone(),
                    right: pair[1].1.clone(),
                };
                let location = (line_ix, pair[0].0);
                bigram_data
                    .bigrams_to_locations
                    .entry(bigram.clone())
                    .or_default()
                    .insert(location);
                bigram_data.add_bigram(&bigram, 1);
            }
        }
        bigram_data
    }

    fn add_bigram(&mut self, bigram: &Bigram, delta: i64) {
        *self.left_lex_freqs.entry(bigram.left.clone()).or_insert(0) += delta;
        *self
            .right_lex_freqs
            .entry(bigram.right.clone())
            .or_insert(0) += delta;
        self.total_bigram_count += delta;

        let was_present = self.bigrams_to_freqs.contains_key(bigram);
        *self.bigrams_to_freqs.entry(bigram.clone()).or_insert(0) += delta;
        if delta > 0 && !was_present {
            if let Some(first_seen) = self.bigrams_first_seen_order.as_mut() {
                first_seen.insert(bigram.clone(), self.next_bigram_order);
                self.next_bigram_order += 1;
            }
        }
    }

    fn maybe_remove_bigram(&mut self, bigram: &Bigram) {
        let should_remove = self
            .bigrams_to_freqs
            .get(bigram)
            .map(|freq| *freq <= 0)
            .unwrap_or(false);
        if should_remove {
            self.bigrams_to_freqs.remove(bigram);
            if let Some(first_seen) = self.bigrams_first_seen_order.as_mut() {
                first_seen.remove(bigram);
            }
        }
    }

    fn maybe_remove_lr_lexemes(&mut self, bigram: &Bigram) {
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
    bigram: Bigram,
    merged_lexeme: Lexeme,
    bigram_locations: Vec<Location>,
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
                word: all_words.into_iter().collect(),
                ix: 0,
            },
            bigram_locations: locations,
        }
    }

    fn n_lexemes(&self) -> usize {
        self.merged_lexeme.word.len()
    }

    fn cleaned_bigram_locations(&self) -> Vec<Location> {
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
struct CandidateScore {
    score: f64,
    frequency: i64,
}

#[derive(Clone)]
struct StepData {
    score: f64,
    winner: WinnerInfo,
    line_hits_count: usize,
}

enum StepStatus {
    Winner(StepData),
    NoCandidate,
    BelowMinScore(f64),
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

fn merged_word_ids(bigram: &Bigram) -> SmallVec<[TokenId; 6]> {
    let mut merged =
        SmallVec::<[TokenId; 6]>::with_capacity(bigram.left.word.len() + bigram.right.word.len());
    merged.extend_from_slice(&bigram.left.word);
    merged.extend_from_slice(&bigram.right.word);
    merged
}

fn root_items_from_old_with_merges(
    old_root_items: &[(usize, Lexeme)],
    starts: &[usize],
    merged_lexeme: &Lexeme,
    merged_width: usize,
) -> Vec<(usize, Lexeme)> {
    if starts.is_empty() {
        return old_root_items.to_vec();
    }

    let start_set = starts.iter().copied().collect::<HashSet<_>>();
    let mut new_root_items = Vec::with_capacity(old_root_items.len());

    for (ix, lexeme) in old_root_items {
        let insertion_ix = starts.partition_point(|s| *s <= *ix);
        let in_merged_span = insertion_ix > 0 && *ix < starts[insertion_ix - 1] + merged_width;
        if !in_merged_span {
            new_root_items.push((*ix, lexeme.clone()));
            continue;
        }

        if start_set.contains(ix) {
            new_root_items.push((*ix, merged_lexeme.clone()));
        }
    }

    new_root_items
}

fn select_candidate(
    tie_breaker: TieBreaker,
    method: SelectionMethod,
    bigrams_to_freqs: &HashMap<Bigram, i64>,
    bigrams_first_seen_order: Option<&HashMap<Bigram, usize>>,
    candidate_scores: &HashMap<Bigram, CandidateScore>,
    min_count: i64,
) -> Option<(Bigram, CandidateScore)> {
    let legacy = tie_breaker == TieBreaker::LegacyFirstSeen;

    if method == SelectionMethod::Frequency {
        let mut best: Option<(Bigram, CandidateScore)> = None;
        let mut best_key: Option<(Reverse<i64>, SmallVec<[TokenId; 6]>)> = None;
        let mut best_score = f64::NEG_INFINITY;
        let mut best_order = usize::MAX;

        for (bigram, freq) in bigrams_to_freqs {
            if *freq < min_count {
                continue;
            }
            let score = *freq as f64;
            let candidate = CandidateScore {
                score,
                frequency: *freq,
            };
            let order = if legacy {
                bigrams_first_seen_order
                    .and_then(|m| m.get(bigram))
                    .copied()
                    .unwrap_or(usize::MAX)
            } else {
                usize::MAX
            };

            if best.is_none() || score > best_score {
                best_score = score;
                best_order = order;
                best = Some((bigram.clone(), candidate.clone()));
                if !legacy {
                    best_key = Some((Reverse(*freq), merged_word_ids(bigram)));
                }
                continue;
            }

            if scores_close(score, best_score) {
                if legacy {
                    if order < best_order {
                        best_order = order;
                        best = Some((bigram.clone(), candidate));
                    }
                } else {
                    let candidate_key = (Reverse(*freq), merged_word_ids(bigram));
                    if let Some(key) = &best_key {
                        if candidate_key < *key {
                            best = Some((bigram.clone(), candidate));
                            best_key = Some(candidate_key);
                        }
                    }
                }
            }
        }

        return best;
    }

    if candidate_scores.is_empty() {
        return None;
    }

    let mut best: Option<(Bigram, CandidateScore)> = None;
    let mut best_key: Option<(Reverse<i64>, SmallVec<[TokenId; 6]>)> = None;
    let mut best_score = f64::NEG_INFINITY;
    let mut best_order = usize::MAX;

    for (bigram, candidate) in candidate_scores {
        let order = if legacy {
            bigrams_first_seen_order
                .and_then(|m| m.get(bigram))
                .copied()
                .unwrap_or(usize::MAX)
        } else {
            usize::MAX
        };

        if best.is_none() || candidate.score > best_score {
            best_score = candidate.score;
            best_order = order;
            best = Some((bigram.clone(), candidate.clone()));
            if !legacy {
                best_key = Some((Reverse(candidate.frequency), merged_word_ids(bigram)));
            }
            continue;
        }

        if scores_close(candidate.score, best_score) {
            if legacy {
                if order < best_order {
                    best_order = order;
                    best = Some((bigram.clone(), candidate.clone()));
                }
            } else {
                let candidate_key = (Reverse(candidate.frequency), merged_word_ids(bigram));
                if let Some(key) = &best_key {
                    if candidate_key < *key {
                        best = Some((bigram.clone(), candidate.clone()));
                        best_key = Some(candidate_key);
                    }
                }
            }
        }
    }

    best
}

fn merge_winner(
    winner: &WinnerInfo,
    clean_locations: &[Location],
    lexeme_data: &mut LexemeData,
    bigram_data: &mut BigramData,
) -> HashSet<Bigram> {
    let mut bigram_lines = HashSet::new();
    let mut merge_starts_by_line: HashMap<usize, Vec<usize>> = HashMap::new();
    for (line_ix, word_ix) in clean_locations {
        bigram_lines.insert(*line_ix);
        merge_starts_by_line
            .entry(*line_ix)
            .or_default()
            .push(*word_ix);
    }

    let mut touched_bigrams = HashSet::new();
    touched_bigrams.insert(winner.bigram.clone());

    let mut old_bigrams_lookup: HashMap<usize, Vec<(usize, Lexeme)>> = HashMap::new();
    for line_ix in bigram_lines {
        old_bigrams_lookup.insert(line_ix, lexeme_data.root_items_for_line(line_ix));
    }

    for (line_ix, word_ix) in clean_locations.iter().copied() {
        for lexeme_index in 0..winner.n_lexemes() {
            let pos = word_ix + lexeme_index;
            let old_lexeme = lexeme_data.locations_to_lexemes[line_ix][pos].clone();

            let lexeme = Lexeme {
                word: winner.merged_lexeme.word.clone(),
                ix: lexeme_index,
            };
            lexeme_data.locations_to_lexemes[line_ix][pos] = lexeme.clone();

            if let Some(locations) = lexeme_data.lexemes_to_locations.get_mut(&old_lexeme) {
                locations.remove(&(line_ix, pos));
                if locations.is_empty() {
                    lexeme_data.lexemes_to_locations.remove(&old_lexeme);
                }
            }
            lexeme_data
                .lexemes_to_locations
                .entry(lexeme)
                .or_default()
                .insert((line_ix, pos));
        }
    }

    for (line_ix, old_root_items) in old_bigrams_lookup {
        let line_merge_starts = merge_starts_by_line
            .get(&line_ix)
            .map(Vec::as_slice)
            .unwrap_or(&[]);
        let old_bigrams = old_root_items
            .windows(2)
            .map(|pair| {
                (
                    Bigram {
                        left: pair[0].1.clone(),
                        right: pair[1].1.clone(),
                    },
                    (line_ix, pair[0].0),
                )
            })
            .collect::<Vec<_>>();

        let new_root_items = root_items_from_old_with_merges(
            &old_root_items,
            line_merge_starts,
            &winner.merged_lexeme,
            winner.n_lexemes(),
        );
        let new_bigrams = new_root_items
            .windows(2)
            .map(|pair| {
                (
                    Bigram {
                        left: pair[0].1.clone(),
                        right: pair[1].1.clone(),
                    },
                    (line_ix, pair[0].0),
                )
            })
            .collect::<Vec<_>>();

        for (bigram, _) in &old_bigrams {
            touched_bigrams.insert(bigram.clone());
        }
        for (bigram, _) in &new_bigrams {
            touched_bigrams.insert(bigram.clone());
        }

        for (bigram, _) in &new_bigrams {
            bigram_data.add_bigram(bigram, 1);
            bigram_data.maybe_remove_bigram(bigram);
            bigram_data.maybe_remove_lr_lexemes(bigram);
        }
        for (bigram, _) in &old_bigrams {
            bigram_data.add_bigram(bigram, -1);
            bigram_data.maybe_remove_bigram(bigram);
            bigram_data.maybe_remove_lr_lexemes(bigram);
        }

        for (bigram, location) in old_bigrams {
            if let Some(locations) = bigram_data.bigrams_to_locations.get_mut(&bigram) {
                locations.remove(&location);
                if locations.is_empty() {
                    bigram_data.bigrams_to_locations.remove(&bigram);
                }
            }
        }
        for (bigram, location) in new_bigrams {
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
        if *el1_freq <= 0 {
            lexeme_data.lexemes_to_freqs.remove(&winner.bigram.left);
        }
    }
    if let Some(el2_freq) = lexeme_data.lexemes_to_freqs.get_mut(&winner.bigram.right) {
        *el2_freq -= merge_token_count;
        if *el2_freq <= 0 {
            lexeme_data.lexemes_to_freqs.remove(&winner.bigram.right);
        }
    }

    touched_bigrams
}

#[pyclass]
struct Engine {
    interner: Interner,
    lexemes: LexemeData,
    bigrams: BigramData,
    method: SelectionMethod,
    min_count: i64,
    tie_breaker: TieBreaker,
    candidate_scores: HashMap<Bigram, CandidateScore>,
    dirty_bigrams: HashSet<Bigram>,
    iteration_counter: usize,
}

impl Engine {
    fn refresh_candidate_scores(&mut self, force_full: bool) {
        if self.method == SelectionMethod::Frequency {
            return;
        }

        #[allow(clippy::manual_is_multiple_of)]
        let full = force_full
            || self.candidate_scores.is_empty()
            || self.iteration_counter % FULL_RESCORE_INTERVAL == 0;

        let total_bigram_count = self.bigrams.total_bigram_count;

        if full {
            let snapshot = self
                .bigrams
                .bigrams_to_freqs
                .iter()
                .filter(|(_, freq)| **freq >= self.min_count)
                .map(|(bigram, freq)| {
                    (
                        bigram.clone(),
                        *freq,
                        *self.bigrams.left_lex_freqs.get(&bigram.left).unwrap_or(&0),
                        *self
                            .bigrams
                            .right_lex_freqs
                            .get(&bigram.right)
                            .unwrap_or(&0),
                    )
                })
                .collect::<Vec<_>>();

            let scored = snapshot
                .par_iter()
                .map(|(bigram, freq, left_freq, right_freq)| {
                    (
                        bigram.clone(),
                        CandidateScore {
                            score: score_ll_npmi(
                                self.method,
                                *freq,
                                *left_freq,
                                *right_freq,
                                total_bigram_count,
                            ),
                            frequency: *freq,
                        },
                    )
                })
                .collect::<Vec<_>>();

            self.candidate_scores.clear();
            for (bigram, score) in scored {
                if score.score > f64::NEG_INFINITY {
                    self.candidate_scores.insert(bigram, score);
                }
            }
            self.dirty_bigrams.clear();
            return;
        }

        let dirty = self.dirty_bigrams.drain().collect::<Vec<_>>();
        let dirty_snapshot = dirty
            .into_iter()
            .map(|bigram| {
                (
                    bigram.clone(),
                    self.bigrams.bigrams_to_freqs.get(&bigram).copied(),
                    self.bigrams.left_lex_freqs.get(&bigram.left).copied(),
                    self.bigrams.right_lex_freqs.get(&bigram.right).copied(),
                )
            })
            .collect::<Vec<_>>();

        let rescored = dirty_snapshot
            .par_iter()
            .map(|(bigram, maybe_freq, maybe_left, maybe_right)| {
                let Some(freq) = maybe_freq else {
                    return (bigram.clone(), None);
                };
                if *freq < self.min_count {
                    return (bigram.clone(), None);
                }
                let left_freq = maybe_left.unwrap_or(0);
                let right_freq = maybe_right.unwrap_or(0);
                let score = score_ll_npmi(
                    self.method,
                    *freq,
                    left_freq,
                    right_freq,
                    total_bigram_count,
                );
                if score == f64::NEG_INFINITY {
                    (bigram.clone(), None)
                } else {
                    (
                        bigram.clone(),
                        Some(CandidateScore {
                            score,
                            frequency: *freq,
                        }),
                    )
                }
            })
            .collect::<Vec<_>>();

        for (bigram, maybe_score) in rescored {
            if let Some(score) = maybe_score {
                self.candidate_scores.insert(bigram, score);
            } else {
                self.candidate_scores.remove(&bigram);
            }
        }
    }

    fn token_ids_to_strings(&self, token_ids: &[TokenId]) -> Vec<String> {
        self.interner.ids_to_strings(token_ids)
    }

    fn step_payload(&self, step_data: StepData) -> StepPayload {
        (
            step_data.score,
            self.token_ids_to_strings(&step_data.winner.bigram.left.word),
            step_data.winner.bigram.left.ix,
            self.token_ids_to_strings(&step_data.winner.bigram.right.word),
            step_data.winner.bigram.right.ix,
            self.token_ids_to_strings(&step_data.winner.merged_lexeme.word),
            step_data.winner.merged_lexeme.ix,
            step_data.winner.bigram_locations,
        )
    }

    fn step_internal(&mut self, min_score: Option<f64>) -> StepStatus {
        self.refresh_candidate_scores(false);

        let Some((bigram, candidate)) = select_candidate(
            self.tie_breaker,
            self.method,
            &self.bigrams.bigrams_to_freqs,
            self.bigrams.bigrams_first_seen_order.as_ref(),
            &self.candidate_scores,
            self.min_count,
        ) else {
            return StepStatus::NoCandidate;
        };

        if let Some(score_threshold) = min_score {
            if candidate.score < score_threshold {
                return StepStatus::BelowMinScore(candidate.score);
            }
        }

        let winner = WinnerInfo::from_bigram_with_data(&bigram, &self.bigrams);
        let clean_locations = winner.cleaned_bigram_locations();
        let line_hits_count = if clean_locations.is_empty() {
            0
        } else {
            1 + clean_locations
                .windows(2)
                .filter(|pair| pair[0].0 != pair[1].0)
                .count()
        };

        let touched = merge_winner(
            &winner,
            &clean_locations,
            &mut self.lexemes,
            &mut self.bigrams,
        );
        self.dirty_bigrams = touched;
        self.iteration_counter += 1;

        StepStatus::Winner(StepData {
            score: candidate.score,
            winner,
            line_hits_count,
        })
    }

    fn run_internal(
        &mut self,
        iterations: usize,
        min_score: Option<f64>,
        return_progress: bool,
    ) -> RunOutcome {
        let mut winners = Vec::new();
        let mut progress = Vec::new();
        let corpus_length = self.lexemes.corpus_length();

        for _ in 0..iterations {
            match self.step_internal(min_score) {
                StepStatus::NoCandidate => {
                    return (
                        "no_candidate".to_string(),
                        winners,
                        None,
                        corpus_length,
                        progress,
                    )
                }
                StepStatus::BelowMinScore(score) => {
                    return (
                        "below_min_score".to_string(),
                        winners,
                        Some(score),
                        corpus_length,
                        progress,
                    )
                }
                StepStatus::Winner(step_data) => {
                    if return_progress {
                        progress.push((
                            step_data.line_hits_count,
                            step_data.score,
                            self.token_ids_to_strings(&step_data.winner.merged_lexeme.word),
                        ));
                    }
                    winners.push(self.step_payload(step_data));
                }
            }
        }

        (
            "completed".to_string(),
            winners,
            None,
            corpus_length,
            progress,
        )
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
);

type ProgressPayload = (usize, f64, Vec<String>);
type RunOutcome = (
    String,
    Vec<StepPayload>,
    Option<f64>,
    usize,
    Vec<ProgressPayload>,
);

#[pymethods]
impl Engine {
    #[new]
    #[pyo3(signature = (
        corpus,
        method,
        min_count,
        tie_breaker,
        splitter="delimiter",
        line_delimiter=None,
        sentencex_language="en"
    ))]
    fn new(
        corpus: Vec<String>,
        method: &str,
        min_count: usize,
        tie_breaker: &str,
        splitter: &str,
        line_delimiter: Option<&str>,
        sentencex_language: &str,
    ) -> PyResult<Self> {
        let method = SelectionMethod::parse(method)?;
        let tie_breaker = TieBreaker::parse(tie_breaker)?;
        let splitter = Splitter::parse(splitter, line_delimiter, sentencex_language)?;
        let (interner, corpus_ids) = Interner::from_documents(&corpus, splitter);

        let lexemes = LexemeData::from_corpus(&corpus_ids);
        let track_first_seen = tie_breaker == TieBreaker::LegacyFirstSeen;
        let bigrams = BigramData::from_lexemes(&lexemes, track_first_seen);

        Ok(Self {
            interner,
            lexemes,
            bigrams,
            method,
            min_count: min_count as i64,
            tie_breaker,
            candidate_scores: HashMap::new(),
            dirty_bigrams: HashSet::new(),
            iteration_counter: 0,
        })
    }

    fn corpus_length(&self) -> usize {
        self.lexemes.corpus_length()
    }

    #[pyo3(signature = (iterations, min_score=None, return_progress=false))]
    fn run(
        &mut self,
        py: Python<'_>,
        iterations: usize,
        min_score: Option<f64>,
        return_progress: bool,
    ) -> RunOutcome {
        py.allow_threads(|| self.run_internal(iterations, min_score, return_progress))
    }
}

#[pymodule(gil_used = true)]
fn _core(_py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<Engine>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_engine(corpus: Vec<&str>, method: SelectionMethod, min_count: i64) -> Engine {
        let corpus = corpus.into_iter().map(str::to_string).collect::<Vec<_>>();

        let (interner, corpus_ids) =
            Interner::from_documents(&corpus, Splitter::Delimiter(Some("\n")));
        let lexemes = LexemeData::from_corpus(&corpus_ids);
        let bigrams = BigramData::from_lexemes(&lexemes, false);

        Engine {
            interner,
            lexemes,
            bigrams,
            method,
            min_count,
            tie_breaker: TieBreaker::Deterministic,
            candidate_scores: HashMap::new(),
            dirty_bigrams: HashSet::new(),
            iteration_counter: 0,
        }
    }

    #[test]
    fn interner_roundtrip() {
        let corpus = vec!["b a b".to_string()];
        let (interner, corpus_ids) =
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
    }

    #[test]
    fn deterministic_tie_break_prefers_frequency_then_lexicographic() {
        let mut engine = build_engine(
            vec!["a d", "a c", "a b", "a b"],
            SelectionMethod::Frequency,
            0,
        );
        let StepStatus::Winner(step) = engine.step_internal(None) else {
            panic!("expected winner");
        };
        let merged = engine.token_ids_to_strings(&step.winner.merged_lexeme.word);
        assert_eq!(merged, vec!["a", "b"]);
    }

    #[test]
    fn engine_run_matches_repeated_step_for_frequency() {
        let corpus = vec!["a a a a"];
        let mut run_engine = build_engine(corpus.clone(), SelectionMethod::Frequency, 0);
        let mut step_engine = build_engine(corpus, SelectionMethod::Frequency, 0);

        let (_, run_payloads, _, _, _) = run_engine.run_internal(3, None, false);

        let mut step_payloads = Vec::new();
        for _ in 0..3 {
            match step_engine.step_internal(None) {
                StepStatus::Winner(step_data) => {
                    step_payloads.push(step_engine.step_payload(step_data))
                }
                _ => break,
            }
        }

        assert_eq!(run_payloads, step_payloads);
    }

    #[test]
    fn min_score_blocks_low_score_winner() {
        let mut engine = build_engine(vec!["a b c"], SelectionMethod::Frequency, 0);
        let status = engine.step_internal(Some(10.0));
        let StepStatus::BelowMinScore(score) = status else {
            panic!("expected below-min-score status");
        };
        assert_eq!(score, 1.0);
    }
}
