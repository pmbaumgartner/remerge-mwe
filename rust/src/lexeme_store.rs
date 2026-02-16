use crate::types::{LexemeId, TokenId};
use rustc_hash::FxHashMap;
use smallvec::SmallVec;

#[derive(Clone, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub(crate) struct Lexeme {
    pub(crate) word: SmallVec<[TokenId; 3]>,
    pub(crate) ix: usize,
}

#[derive(Default)]
pub(crate) struct LexemeStore {
    lexeme_to_id: FxHashMap<Lexeme, LexemeId>,
    pub(crate) id_to_lexeme: Vec<Lexeme>,
}

impl LexemeStore {
    pub(crate) fn intern(&mut self, lexeme: Lexeme) -> LexemeId {
        if let Some(id) = self.lexeme_to_id.get(&lexeme) {
            return *id;
        }

        let id = LexemeId::try_from(self.id_to_lexeme.len())
            .expect("lexeme vocabulary exceeded LexemeId capacity (u32)");
        self.id_to_lexeme.push(lexeme.clone());
        self.lexeme_to_id.insert(lexeme, id);
        id
    }

    pub(crate) fn get(&self, id: LexemeId) -> &Lexeme {
        &self.id_to_lexeme[id as usize]
    }

    pub(crate) fn id_for_lexeme(&self, lexeme: &Lexeme) -> Option<LexemeId> {
        self.lexeme_to_id.get(lexeme).copied()
    }
}
