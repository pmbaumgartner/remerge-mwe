use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
#[cfg(feature = "location-hashset")]
use rustc_hash::FxHashSet;

pub(crate) const SMALL: f64 = 1e-10;
pub(crate) const SCORE_ATOL: f64 = 1e-12;
pub(crate) const SCORE_RTOL: f64 = 1e-12;
pub(crate) const DEFAULT_RESCORE_INTERVAL: usize = 25;
pub(crate) const PARALLEL_SCORE_THRESHOLD: usize = 500;

pub(crate) type TokenId = u32;
pub(crate) type LexemeId = u32;
pub(crate) type Location = (usize, usize);
#[cfg(feature = "location-hashset")]
pub(crate) type BigramLocations = FxHashSet<Location>;
#[cfg(not(feature = "location-hashset"))]
pub(crate) type BigramLocations = Vec<Location>;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum SelectionMethod {
    Frequency,
    LogLikelihood,
    Npmi,
    LogDice,
    TScore,
    DeltaP,
}

impl SelectionMethod {
    pub(crate) fn parse(value: &str) -> PyResult<Self> {
        match value {
            "frequency" => Ok(Self::Frequency),
            "log_likelihood" => Ok(Self::LogLikelihood),
            "npmi" => Ok(Self::Npmi),
            "logdice" => Ok(Self::LogDice),
            "t_score" | "tscore" => Ok(Self::TScore),
            "delta_p" | "deltap" => Ok(Self::DeltaP),
            _ => Err(PyValueError::new_err(format!(
                "Invalid method {value:?}. Expected one of: 'frequency', 'log_likelihood', 'npmi', 'logdice', 't_score', 'delta_p'."
            ))),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum SearchStrategy {
    Greedy,
    Beam,
}

impl SearchStrategy {
    pub(crate) fn parse(value: &str) -> PyResult<Self> {
        match value {
            "greedy" => Ok(Self::Greedy),
            "beam" => Ok(Self::Beam),
            _ => Err(PyValueError::new_err(format!(
                "Invalid search_strategy {value:?}. Expected one of: 'greedy', 'beam'."
            ))),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum StopwordPolicy {
    None,
    BlockStopwordStopword,
    BlockAnyStopword,
}

impl StopwordPolicy {
    pub(crate) fn parse(value: &str) -> PyResult<Self> {
        match value {
            "none" => Ok(Self::None),
            "block_stopword_stopword" => Ok(Self::BlockStopwordStopword),
            "block_any_stopword" => Ok(Self::BlockAnyStopword),
            _ => Err(PyValueError::new_err(format!(
                "Invalid stopword_policy {value:?}. Expected one of: 'none', 'block_stopword_stopword', 'block_any_stopword'."
            ))),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Splitter<'a> {
    Delimiter(Option<&'a str>),
    Sentencex { language: &'a str },
}

impl<'a> Splitter<'a> {
    pub(crate) fn parse(
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub(crate) enum RunStatus {
    Completed = 0,
    NoCandidate = 1,
    BelowMinScore = 2,
}

impl RunStatus {
    pub(crate) fn code(self) -> u8 {
        self as u8
    }
}
