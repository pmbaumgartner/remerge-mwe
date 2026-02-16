use crate::types::{Splitter, TokenId};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};
use sentencex::segment;

pub(crate) fn validate_token_vocabulary_size(vocab_size: usize) -> PyResult<()> {
    let capacity = (u32::MAX as usize).saturating_add(1);
    if vocab_size > capacity {
        return Err(PyValueError::new_err(
            "token vocabulary exceeded TokenId capacity (u32).",
        ));
    }
    Ok(())
}

#[derive(Default)]
pub(crate) struct Interner {
    str_to_id: FxHashMap<String, TokenId>,
    id_to_str: Vec<String>,
}

impl Interner {
    pub(crate) fn from_documents(
        documents: &[String],
        splitter: Splitter<'_>,
    ) -> PyResult<(Self, Vec<Vec<TokenId>>, Vec<usize>)> {
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
        validate_token_vocabulary_size(sorted.len())?;

        let mut interner = Self::default();
        interner.str_to_id.reserve(sorted.len());
        interner.id_to_str.reserve(sorted.len());
        for token in sorted {
            let id = interner.id_to_str.len() as TokenId;
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

        Ok((interner, corpus_ids, doc_boundaries))
    }

    pub(crate) fn id_for(&self, value: &str) -> TokenId {
        *self
            .str_to_id
            .get(value)
            .expect("token missing in interner while converting corpus")
    }

    pub(crate) fn ids_to_strings(&self, ids: &[TokenId]) -> Vec<String> {
        ids.iter()
            .map(|id| self.id_to_str[*id as usize].clone())
            .collect()
    }
}
