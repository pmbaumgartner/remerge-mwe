use crate::bigram_data::BigramData;
use crate::engine::Engine;
use crate::interner::Interner;
use crate::lexeme_data::LexemeData;
use crate::lexeme_store::LexemeStore;
use crate::types::{
    RunStatus, SearchStrategy, SelectionMethod, Splitter, StopwordPolicy, DEFAULT_RESCORE_INTERVAL,
};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};
use std::panic::{catch_unwind, AssertUnwindSafe};

#[pyclass(frozen)]
#[derive(Clone, Debug, PartialEq)]
pub struct StepResult {
    #[pyo3(get)]
    pub(crate) score: f64,
    #[pyo3(get)]
    pub(crate) left_word: Vec<String>,
    #[pyo3(get)]
    pub(crate) left_ix: usize,
    #[pyo3(get)]
    pub(crate) right_word: Vec<String>,
    #[pyo3(get)]
    pub(crate) right_ix: usize,
    #[pyo3(get)]
    pub(crate) merged_word: Vec<String>,
    #[pyo3(get)]
    pub(crate) merged_ix: usize,
    #[pyo3(get)]
    pub(crate) merge_token_count: usize,
    #[pyo3(get)]
    pub(crate) merge_segment_range: usize,
}

pub(crate) type RunOutcome = (u8, Vec<StepResult>, Option<f64>, usize);
pub(crate) type AnnotateRunOutcome = (
    u8,
    Vec<StepResult>,
    Option<f64>,
    usize,
    Vec<String>,
    Vec<String>,
);

fn panic_payload_to_string(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        return (*message).to_string();
    }
    if let Some(message) = payload.downcast_ref::<String>() {
        return message.clone();
    }
    "unknown panic payload".to_string()
}

#[pymethods]
impl Engine {
    #[new]
    #[pyo3(signature = (
        corpus,
        method,
        min_count,
        splitter="delimiter",
        line_delimiter=Some("\n"),
        sentencex_language="en",
        rescore_interval=DEFAULT_RESCORE_INTERVAL,
        stopwords=None,
        stopword_policy="none",
        block_punct_only=false,
        min_range=1,
        range_alpha=0.0,
        min_p_ab=None,
        min_p_ba=None,
        min_merge_count=1,
        search_strategy="greedy",
        beam_width=1,
        beam_top_m=8,
    ))]
    fn new(
        corpus: Vec<String>,
        method: &str,
        min_count: usize,
        splitter: &str,
        line_delimiter: Option<&str>,
        sentencex_language: &str,
        rescore_interval: usize,
        stopwords: Option<Vec<String>>,
        stopword_policy: &str,
        block_punct_only: bool,
        min_range: usize,
        range_alpha: f64,
        min_p_ab: Option<f64>,
        min_p_ba: Option<f64>,
        min_merge_count: usize,
        search_strategy: &str,
        beam_width: usize,
        beam_top_m: usize,
    ) -> PyResult<Self> {
        if rescore_interval == 0 {
            return Err(PyValueError::new_err(
                "rescore_interval must be greater than or equal to 1.",
            ));
        }
        if min_range == 0 {
            return Err(PyValueError::new_err(
                "min_range must be greater than or equal to 1.",
            ));
        }
        if !range_alpha.is_finite() {
            return Err(PyValueError::new_err("range_alpha must be finite."));
        }
        if range_alpha < 0.0 {
            return Err(PyValueError::new_err(
                "range_alpha must be greater than or equal to 0.",
            ));
        }
        if min_merge_count == 0 {
            return Err(PyValueError::new_err(
                "min_merge_count must be greater than or equal to 1.",
            ));
        }
        if beam_width == 0 {
            return Err(PyValueError::new_err(
                "beam_width must be greater than or equal to 1.",
            ));
        }
        if beam_top_m == 0 {
            return Err(PyValueError::new_err(
                "beam_top_m must be greater than or equal to 1.",
            ));
        }

        let validate_probability = |name: &str, value: Option<f64>| -> PyResult<Option<f64>> {
            if let Some(v) = value {
                if !v.is_finite() {
                    return Err(PyValueError::new_err(format!("{name} must be finite.")));
                }
                if !(0.0..=1.0).contains(&v) {
                    return Err(PyValueError::new_err(format!(
                        "{name} must be between 0.0 and 1.0 inclusive.",
                    )));
                }
            }
            Ok(value)
        };
        let min_p_ab = validate_probability("min_p_ab", min_p_ab)?;
        let min_p_ba = validate_probability("min_p_ba", min_p_ba)?;

        let method = SelectionMethod::parse(method)?;
        let splitter = Splitter::parse(splitter, line_delimiter, sentencex_language)?;
        let stopword_policy = StopwordPolicy::parse(stopword_policy)?;
        let search_strategy = SearchStrategy::parse(search_strategy)?;
        let segment_delimiter = match splitter {
            Splitter::Delimiter(Some(delim)) => delim.to_string(),
            Splitter::Delimiter(None) => String::new(),
            Splitter::Sentencex { .. } => " ".to_string(),
        };

        let (interner, corpus_ids, doc_boundaries) = Interner::from_documents(&corpus, splitter)?;

        let mut lexeme_store = LexemeStore::default();
        let lexemes = LexemeData::from_corpus(&corpus_ids, doc_boundaries, &mut lexeme_store);
        let bigrams = BigramData::from_lexemes(&lexemes, &lexeme_store);
        let stopword_token_ids = stopwords
            .unwrap_or_default()
            .into_iter()
            .filter_map(|token| interner.maybe_id_for(&token))
            .collect::<FxHashSet<_>>();

        let mut engine = Self {
            interner,
            lexeme_store,
            lexemes,
            bigrams,
            segment_delimiter,
            method,
            min_count: min_count as i64,
            rescore_interval,
            stopword_token_ids,
            stopword_policy,
            block_punct_only,
            min_range,
            range_alpha,
            min_p_ab,
            min_p_ba,
            min_merge_count,
            search_strategy,
            beam_width,
            beam_top_m,
            candidate_scores: FxHashMap::default(),
            candidate_heap: std::collections::BinaryHeap::new(),
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
    fn run(
        &mut self,
        py: Python<'_>,
        iterations: usize,
        min_score: Option<f64>,
    ) -> PyResult<RunOutcome> {
        let result = py.allow_threads(|| {
            catch_unwind(AssertUnwindSafe(|| {
                self.run_internal(iterations, min_score)
            }))
            .map_err(panic_payload_to_string)
        });
        result.map_err(|message| {
            PyRuntimeError::new_err(format!("remerge engine panicked during run(): {message}"))
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
        let result = py.allow_threads(|| {
            catch_unwind(AssertUnwindSafe(|| {
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
            }))
            .map_err(panic_payload_to_string)
        });
        result.map_err(|message| {
            PyRuntimeError::new_err(format!(
                "remerge engine panicked during run_and_annotate(): {message}"
            ))
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
