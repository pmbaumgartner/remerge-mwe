use super::annotations::{ActiveSpan, TaggedSentence};
use crate::py_bindings::{panic_payload_to_string, AnnotateRunOutcome, RunOutcome, StepResult};
use crate::scoring::score_ll_npmi;
use crate::types::{RunStatus, SelectionMethod};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use rustc_hash::FxHashMap;
use std::cmp::Ordering;
use std::panic::{catch_unwind, AssertUnwindSafe};

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
enum MatchKind {
    Support,
    Full,
}

#[derive(Clone, Debug)]
struct Pattern {
    positions: Vec<Vec<String>>,
}

impl Pattern {
    fn matches_at(&self, tags: &[String], offset: usize) -> bool {
        tags.iter().enumerate().all(|(index, tag)| {
            let alternatives = &self.positions[offset + index];
            alternatives
                .iter()
                .any(|candidate| candidate == "*" || candidate == tag)
        })
    }

    fn classifies(&self, tags: &[String]) -> Option<MatchKind> {
        if tags.len() == self.positions.len() && self.matches_at(tags, 0) {
            return Some(MatchKind::Full);
        }
        if tags.len() >= self.positions.len() {
            return None;
        }
        (0..=self.positions.len() - tags.len())
            .any(|offset| self.matches_at(tags, offset))
            .then_some(MatchKind::Support)
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct CandidateKey {
    left: Vec<String>,
    right: Vec<String>,
    kind: MatchKind,
}

impl CandidateKey {
    fn merged(&self) -> Vec<String> {
        self.left.iter().chain(&self.right).cloned().collect()
    }
}

#[derive(Clone, Debug)]
struct Occurrence {
    sentence: usize,
    left_index: usize,
    start: usize,
    end: usize,
}

#[derive(Clone, Debug)]
struct Candidate {
    key: CandidateKey,
    score: f64,
    frequency: i64,
    occurrences: Vec<Occurrence>,
}

impl Candidate {
    fn compare(&self, other: &Self) -> Ordering {
        self.score
            .total_cmp(&other.score)
            .then_with(|| self.frequency.cmp(&other.frequency))
            .then_with(|| other.key.merged().cmp(&self.key.merged()))
            .then_with(|| self.key.kind.cmp(&other.key.kind))
            .then_with(|| other.key.left.cmp(&self.key.left))
            .then_with(|| other.key.right.cmp(&self.key.right))
    }
}

#[pyclass]
pub(crate) struct PosEngine {
    sentences: Vec<TaggedSentence>,
    doc_boundaries: Vec<usize>,
    patterns: Vec<Pattern>,
    method: SelectionMethod,
    min_count: i64,
}

impl PosEngine {
    fn classify(&self, tags: &[String]) -> Option<MatchKind> {
        let mut result = None;
        for pattern in &self.patterns {
            match pattern.classifies(tags) {
                Some(MatchKind::Full) => return Some(MatchKind::Full),
                Some(MatchKind::Support) => result = Some(MatchKind::Support),
                None => {}
            }
        }
        result
    }

    fn collect_occurrences(&self) -> FxHashMap<CandidateKey, Vec<Occurrence>> {
        let mut raw: FxHashMap<CandidateKey, Vec<Occurrence>> = FxHashMap::default();
        for (sentence_index, sentence) in self.sentences.iter().enumerate() {
            for (left_index, pair) in sentence.active.windows(2).enumerate() {
                let start = pair[0].start;
                let end = pair[1].end;
                let Some(kind) = self.classify(&sentence.upos[start..end]) else {
                    continue;
                };
                let key = CandidateKey {
                    left: sentence.forms[pair[0].start..pair[0].end].to_vec(),
                    right: sentence.forms[pair[1].start..pair[1].end].to_vec(),
                    kind,
                };
                raw.entry(key).or_default().push(Occurrence {
                    sentence: sentence_index,
                    left_index,
                    start,
                    end,
                });
            }
        }

        for occurrences in raw.values_mut() {
            occurrences.sort_by_key(|item| (item.sentence, item.start, item.end));
            let mut clean = Vec::with_capacity(occurrences.len());
            let mut previous_sentence = usize::MAX;
            let mut next_valid = 0;
            for occurrence in occurrences.drain(..) {
                if occurrence.sentence != previous_sentence {
                    previous_sentence = occurrence.sentence;
                    next_valid = 0;
                }
                if occurrence.start >= next_valid {
                    next_valid = occurrence.end;
                    clean.push(occurrence);
                }
            }
            *occurrences = clean;
        }
        raw
    }

    fn select_candidate(&self) -> Option<Candidate> {
        let occurrences = self.collect_occurrences();
        let mut left_frequencies: FxHashMap<Vec<String>, i64> = FxHashMap::default();
        let mut right_frequencies: FxHashMap<Vec<String>, i64> = FxHashMap::default();
        let mut total = 0_i64;

        for (key, locations) in &occurrences {
            let frequency = locations.len() as i64;
            *left_frequencies.entry(key.left.clone()).or_default() += frequency;
            *right_frequencies.entry(key.right.clone()).or_default() += frequency;
            total += frequency;
        }

        occurrences
            .into_iter()
            .filter_map(|(key, locations)| {
                let frequency = locations.len() as i64;
                if frequency < self.min_count {
                    return None;
                }
                let score = score_ll_npmi(
                    self.method,
                    frequency,
                    *left_frequencies.get(&key.left).unwrap_or(&0),
                    *right_frequencies.get(&key.right).unwrap_or(&0),
                    total,
                );
                (score != f64::NEG_INFINITY).then_some(Candidate {
                    key,
                    score,
                    frequency,
                    occurrences: locations,
                })
            })
            .max_by(Candidate::compare)
    }

    fn merge_candidate(&mut self, candidate: &Candidate) {
        let mut by_sentence: FxHashMap<usize, Vec<&Occurrence>> = FxHashMap::default();
        for occurrence in &candidate.occurrences {
            by_sentence
                .entry(occurrence.sentence)
                .or_default()
                .push(occurrence);
        }

        for (sentence_index, mut occurrences) in by_sentence {
            occurrences.sort_by_key(|item| std::cmp::Reverse(item.left_index));
            let sentence = &mut self.sentences[sentence_index];
            for occurrence in occurrences {
                let left = &sentence.active[occurrence.left_index];
                let right = &sentence.active[occurrence.left_index + 1];
                debug_assert_eq!((left.start, right.end), (occurrence.start, occurrence.end));
                sentence.active[occurrence.left_index] = ActiveSpan {
                    start: occurrence.start,
                    end: occurrence.end,
                };
                sentence.active.remove(occurrence.left_index + 1);
                if candidate.key.kind == MatchKind::Full {
                    sentence.emit(occurrence.start, occurrence.end);
                }
            }
        }
    }

    fn occurrence_coordinate(&self, occurrence: &Occurrence) -> (usize, usize, usize, usize) {
        let document = self
            .doc_boundaries
            .partition_point(|boundary| *boundary <= occurrence.sentence)
            .saturating_sub(1);
        (
            document,
            occurrence.sentence - self.doc_boundaries[document],
            occurrence.start,
            occurrence.end,
        )
    }

    fn step_result(&self, candidate: &Candidate) -> StepResult {
        let coordinates = candidate
            .occurrences
            .iter()
            .map(|occurrence| self.occurrence_coordinate(occurrence))
            .collect::<Vec<_>>();
        StepResult {
            score: candidate.score,
            left_word: candidate.key.left.clone(),
            left_ix: 0,
            right_word: candidate.key.right.clone(),
            right_ix: 0,
            merged_word: candidate.key.merged(),
            merged_ix: 0,
            merge_token_count: candidate.occurrences.len(),
            occurrence_documents: coordinates.iter().map(|item| item.0).collect(),
            occurrence_sentences: coordinates.iter().map(|item| item.1).collect(),
            occurrence_starts: coordinates.iter().map(|item| item.2).collect(),
            occurrence_ends: coordinates.iter().map(|item| item.3).collect(),
        }
    }

    pub(crate) fn run_internal(&mut self, iterations: usize, min_score: Option<f64>) -> RunOutcome {
        let mut winners = Vec::new();
        let corpus_length = self.sentences.len();

        while winners.len() < iterations {
            let Some(candidate) = self.select_candidate() else {
                return (RunStatus::NoCandidate.code(), winners, None, corpus_length);
            };
            if let Some(threshold) = min_score {
                if candidate.score < threshold {
                    return (
                        RunStatus::BelowMinScore.code(),
                        winners,
                        Some(candidate.score),
                        corpus_length,
                    );
                }
            }
            self.merge_candidate(&candidate);
            if candidate.key.kind == MatchKind::Full {
                winners.push(self.step_result(&candidate));
            }
        }

        (RunStatus::Completed.code(), winners, None, corpus_length)
    }

    pub(crate) fn annotate_internal(
        &self,
        mwe_prefix: &str,
        mwe_suffix: &str,
        token_separator: &str,
    ) -> (Vec<String>, Vec<String>) {
        let mut documents = Vec::with_capacity(self.doc_boundaries.len().saturating_sub(1));
        let mut labels = std::collections::BTreeSet::new();

        for boundary in self.doc_boundaries.windows(2) {
            let mut rendered_sentences = Vec::with_capacity(boundary[1] - boundary[0]);
            for sentence in &self.sentences[boundary[0]..boundary[1]] {
                let emitted = sentence
                    .emitted
                    .iter()
                    .copied()
                    .collect::<FxHashMap<usize, usize>>();
                let mut rendered = Vec::new();
                let mut index = 0;
                while index < sentence.forms.len() {
                    if let Some(end) = emitted.get(&index).copied() {
                        let label = format!(
                            "{mwe_prefix}{}{mwe_suffix}",
                            sentence.forms[index..end].join(token_separator)
                        );
                        labels.insert(label.clone());
                        rendered.push(label);
                        index = end;
                    } else {
                        rendered.push(sentence.forms[index].clone());
                        index += 1;
                    }
                }
                rendered_sentences.push(rendered.join(" "));
            }
            documents.push(rendered_sentences.join("\n"));
        }

        (documents, labels.into_iter().collect())
    }
}

#[pymethods]
impl PosEngine {
    #[new]
    fn new(
        corpus: Vec<Vec<Vec<(String, String)>>>,
        patterns: Vec<Vec<Vec<String>>>,
        method: &str,
        min_count: usize,
    ) -> PyResult<Self> {
        if patterns.is_empty() {
            return Err(PyValueError::new_err(
                "patterns must contain at least one pattern.",
            ));
        }
        if patterns.iter().any(|pattern| pattern.len() < 2) {
            return Err(PyValueError::new_err(
                "each POS pattern must contain at least two positions.",
            ));
        }
        if patterns
            .iter()
            .flatten()
            .any(|alternatives| alternatives.is_empty())
        {
            return Err(PyValueError::new_err(
                "POS pattern positions must contain at least one alternative.",
            ));
        }

        let mut sentences = Vec::new();
        let mut doc_boundaries = Vec::with_capacity(corpus.len() + 1);
        for document in corpus {
            doc_boundaries.push(sentences.len());
            sentences.extend(document.into_iter().map(TaggedSentence::new));
        }
        doc_boundaries.push(sentences.len());

        Ok(Self {
            sentences,
            doc_boundaries,
            patterns: patterns
                .into_iter()
                .map(|positions| Pattern { positions })
                .collect(),
            method: SelectionMethod::parse(method)?,
            min_count: min_count as i64,
        })
    }

    fn corpus_length(&self) -> usize {
        self.sentences.len()
    }

    #[pyo3(signature = (iterations, min_score=None))]
    fn run(
        &mut self,
        py: Python<'_>,
        iterations: usize,
        min_score: Option<f64>,
    ) -> PyResult<RunOutcome> {
        py.allow_threads(|| {
            catch_unwind(AssertUnwindSafe(|| {
                self.run_internal(iterations, min_score)
            }))
            .map_err(panic_payload_to_string)
        })
        .map_err(|message| {
            PyRuntimeError::new_err(format!(
                "POS-filtered remerge engine panicked during run(): {message}"
            ))
        })
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
    ) -> PyResult<AnnotateRunOutcome> {
        py.allow_threads(|| {
            catch_unwind(AssertUnwindSafe(|| {
                let (status, winners, score, corpus_length) =
                    self.run_internal(iterations, min_score);
                let (documents, labels) =
                    self.annotate_internal(mwe_prefix, mwe_suffix, token_separator);
                (status, winners, score, corpus_length, documents, labels)
            }))
            .map_err(panic_payload_to_string)
        })
        .map_err(|message| {
            PyRuntimeError::new_err(format!(
                "POS-filtered remerge engine panicked during run_and_annotate(): {message}"
            ))
        })
    }
}
