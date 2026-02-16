use crate::lexeme_store::{Lexeme, LexemeStore};
use crate::types::{LexemeId, Location, TokenId};
use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::smallvec;

#[derive(Default)]
pub(crate) struct LexemeData {
    pub(crate) lexemes_to_locations: FxHashMap<LexemeId, FxHashSet<Location>>,
    pub(crate) locations_to_lexemes: Vec<Vec<LexemeId>>,
    pub(crate) doc_boundaries: Vec<usize>,
}

impl LexemeData {
    pub(crate) fn from_corpus(
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

    pub(crate) fn corpus_length(&self) -> usize {
        self.locations_to_lexemes.len()
    }

    pub(crate) fn root_items_for_line(
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
