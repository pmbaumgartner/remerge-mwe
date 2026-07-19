use super::annotations::{ActiveSpan, TaggedSentence};
use super::candidates::{Candidate, CandidateKey, MatchKind, Occurrence, PosCandidateIndex};
use crate::py_bindings::{panic_payload_to_string, AnnotateRunOutcome, RunOutcome, StepResult};
use crate::types::{RunStatus, SelectionMethod};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use rustc_hash::FxHashMap;
use std::panic::{catch_unwind, AssertUnwindSafe};

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

#[pyclass]
pub(crate) struct PosEngine {
    sentences: Vec<TaggedSentence>,
    doc_boundaries: Vec<usize>,
    patterns: Vec<Pattern>,
    method: SelectionMethod,
    min_count: i64,
    candidate_index: PosCandidateIndex,
    #[cfg(test)]
    adjacency_inspections: usize,
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

    fn adjacency_contribution(
        &mut self,
        sentence_index: usize,
        left_index: usize,
    ) -> Option<(CandidateKey, Occurrence)> {
        #[cfg(test)]
        {
            self.adjacency_inspections += 1;
        }
        let sentence = &self.sentences[sentence_index];
        let left = sentence.active.get(left_index)?;
        let right = sentence.active.get(left_index + 1)?;
        let start = left.start;
        let end = right.end;
        let kind = self.classify(&sentence.upos[start..end])?;
        Some((
            CandidateKey {
                left: sentence.forms[left.start..left.end].to_vec(),
                right: sentence.forms[right.start..right.end].to_vec(),
                kind,
            },
            Occurrence {
                sentence: sentence_index,
                start,
                end,
            },
        ))
    }

    fn initialize_candidate_index(&mut self) {
        for sentence_index in 0..self.sentences.len() {
            let adjacency_count = self.sentences[sentence_index]
                .active
                .len()
                .saturating_sub(1);
            for left_index in 0..adjacency_count {
                if let Some((key, occurrence)) =
                    self.adjacency_contribution(sentence_index, left_index)
                {
                    self.candidate_index.add(key, occurrence);
                }
            }
        }
        self.candidate_index.refresh();
    }

    fn remove_adjacency(&mut self, sentence_index: usize, left_index: usize) {
        if let Some((key, occurrence)) = self.adjacency_contribution(sentence_index, left_index) {
            self.candidate_index.remove(&key, &occurrence);
        }
    }

    fn add_adjacency(&mut self, sentence_index: usize, left_index: usize) {
        if let Some((key, occurrence)) = self.adjacency_contribution(sentence_index, left_index) {
            self.candidate_index.add(key, occurrence);
        }
    }

    fn select_candidate(&self) -> Option<Candidate> {
        self.candidate_index.select(self.method, self.min_count)
    }

    #[cfg(test)]
    fn select_candidate_slow(&self) -> Option<Candidate> {
        let mut index = PosCandidateIndex::default();
        for (sentence_index, sentence) in self.sentences.iter().enumerate() {
            for pair in sentence.active.windows(2) {
                let start = pair[0].start;
                let end = pair[1].end;
                let Some(kind) = self.classify(&sentence.upos[start..end]) else {
                    continue;
                };
                index.add(
                    CandidateKey {
                        left: sentence.forms[pair[0].start..pair[0].end].to_vec(),
                        right: sentence.forms[pair[1].start..pair[1].end].to_vec(),
                        kind,
                    },
                    Occurrence {
                        sentence: sentence_index,
                        start,
                        end,
                    },
                );
            }
        }
        index.refresh();
        index.select(self.method, self.min_count)
    }

    fn merge_candidate(&mut self, candidate: &Candidate) {
        let mut by_sentence: FxHashMap<usize, Vec<Occurrence>> = FxHashMap::default();
        for occurrence in &candidate.occurrences {
            by_sentence
                .entry(occurrence.sentence)
                .or_default()
                .push(occurrence.clone());
        }

        for (sentence_index, mut occurrences) in by_sentence {
            occurrences.sort_by_key(|item| std::cmp::Reverse(item.start));
            for occurrence in occurrences {
                let left_index = self.sentences[sentence_index]
                    .active
                    .partition_point(|span| span.start < occurrence.start);
                let active_len = self.sentences[sentence_index].active.len();
                let first_affected = left_index.saturating_sub(1);
                let last_affected = (left_index + 1).min(active_len.saturating_sub(2));
                for affected in first_affected..=last_affected {
                    self.remove_adjacency(sentence_index, affected);
                }

                let sentence = &mut self.sentences[sentence_index];
                let left = &sentence.active[left_index];
                let right = &sentence.active[left_index + 1];
                debug_assert_eq!((left.start, right.end), (occurrence.start, occurrence.end));
                sentence.active[left_index] = ActiveSpan {
                    start: occurrence.start,
                    end: occurrence.end,
                };
                sentence.active.remove(left_index + 1);
                if candidate.key.kind == MatchKind::Full {
                    sentence.emit(occurrence.start, occurrence.end);
                }

                let active_len = self.sentences[sentence_index].active.len();
                if left_index > 0 {
                    self.add_adjacency(sentence_index, left_index - 1);
                }
                if left_index + 1 < active_len {
                    self.add_adjacency(sentence_index, left_index);
                }
            }
        }
        self.candidate_index.refresh();
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
            merged_word: candidate.merged.clone(),
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

        let mut engine = Self {
            sentences,
            doc_boundaries,
            patterns: patterns
                .into_iter()
                .map(|positions| Pattern { positions })
                .collect(),
            method: SelectionMethod::parse(method)?,
            min_count: min_count as i64,
            candidate_index: PosCandidateIndex::default(),
            #[cfg(test)]
            adjacency_inspections: 0,
        };
        engine.initialize_candidate_index();
        Ok(engine)
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

#[cfg(test)]
mod tests {
    use super::*;

    fn token(form: &str, upos: &str) -> (String, String) {
        (form.to_string(), upos.to_string())
    }

    fn position(tags: &[&str]) -> Vec<String> {
        tags.iter().map(|tag| (*tag).to_string()).collect()
    }

    fn assert_same_candidate(fast: &Candidate, slow: &Candidate) {
        assert_eq!(fast.key, slow.key);
        assert_eq!(fast.merged, slow.merged);
        assert_eq!(fast.score.to_bits(), slow.score.to_bits());
        assert_eq!(fast.frequency, slow.frequency);
        assert_eq!(fast.occurrences, slow.occurrences);
    }

    #[test]
    fn incremental_selection_matches_full_reconstruction() {
        let corpus = vec![
            vec![
                vec![
                    token("new", "ADJ"),
                    token("york", "NOUN"),
                    token("office", "NOUN"),
                    token("opens", "VERB"),
                ],
                vec![
                    token("record", "VERB"),
                    token("deal", "NOUN"),
                    token("record", "NOUN"),
                    token("deal", "NOUN"),
                ],
            ],
            vec![
                vec![
                    token("a", "NOUN"),
                    token("a", "NOUN"),
                    token("a", "NOUN"),
                    token("a", "NOUN"),
                ],
                vec![
                    token("new", "ADJ"),
                    token("york", "NOUN"),
                    token("office", "NOUN"),
                ],
                vec![
                    token("san", "PROPN"),
                    token("francisco", "PROPN"),
                    token("office", "NOUN"),
                ],
            ],
        ];
        let patterns = vec![
            vec![position(&["ADJ"]), position(&["NOUN"])],
            vec![position(&["ADJ"]), position(&["NOUN"]), position(&["NOUN"])],
            vec![position(&["NOUN"]), position(&["NOUN"])],
            vec![position(&["VERB"]), position(&["NOUN"])],
            vec![
                position(&["PROPN"]),
                position(&["PROPN"]),
                position(&["NOUN"]),
            ],
        ];

        for method in ["frequency", "log_likelihood", "npmi"] {
            let mut engine = PosEngine::new(corpus.clone(), patterns.clone(), method, 1).unwrap();
            let mut saw_support = false;
            let mut saw_full = false;
            for _ in 0..64 {
                let fast = engine.select_candidate();
                let slow = engine.select_candidate_slow();
                match (fast, slow) {
                    (Some(fast), Some(slow)) => {
                        assert_same_candidate(&fast, &slow);
                        saw_support |= fast.key.kind == MatchKind::Support;
                        saw_full |= fast.key.kind == MatchKind::Full;
                        engine.merge_candidate(&fast);
                    }
                    (None, None) => break,
                    _ => panic!("incremental and reconstructed candidates diverged"),
                }
            }
            assert!(saw_support);
            assert!(saw_full);
            assert!(engine.select_candidate().is_none());
            assert!(engine.select_candidate_slow().is_none());
        }
    }

    #[test]
    fn merges_inspect_only_the_changed_sentence_neighborhood() {
        let mut sentences = vec![vec![token("bright", "ADJ"), token("river", "NOUN")]];
        sentences.extend((0..500).map(|index| {
            vec![
                token(&format!("verb-{index}"), "VERB"),
                token(&format!("again-{index}"), "VERB"),
            ]
        }));
        let mut engine = PosEngine::new(
            vec![sentences],
            vec![vec![position(&["ADJ"]), position(&["NOUN"])]],
            "frequency",
            1,
        )
        .unwrap();
        let initialization_inspections = engine.adjacency_inspections;
        assert!(initialization_inspections > 500);

        let candidate = engine.select_candidate().unwrap();
        engine.merge_candidate(&candidate);

        assert!(engine.adjacency_inspections - initialization_inspections <= 3);
        assert!(engine.select_candidate().is_none());
    }

    #[test]
    fn full_matches_win_the_explicit_pos_identity_tie() {
        let corpus = vec![vec![
            vec![token("record", "VERB"), token("deal", "NOUN")],
            vec![token("record", "ADJ"), token("deal", "VERB")],
        ]];
        let patterns = vec![
            vec![position(&["VERB"]), position(&["NOUN"])],
            vec![position(&["ADJ"]), position(&["VERB"]), position(&["NOUN"])],
        ];
        let engine = PosEngine::new(corpus, patterns, "frequency", 1).unwrap();

        assert_eq!(engine.select_candidate().unwrap().key.kind, MatchKind::Full);
    }
}
