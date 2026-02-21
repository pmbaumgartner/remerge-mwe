use crate::lexeme_data::LexemeData;
use crate::lexeme_store::LexemeStore;
use crate::types::{BigramLocations, LexemeId, Location};
use rustc_hash::{FxHashMap, FxHashSet};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub(crate) struct BigramId {
    pub(crate) left: LexemeId,
    pub(crate) right: LexemeId,
}

#[derive(Clone, Default)]
pub(crate) struct BigramData {
    pub(crate) bigrams_to_freqs: FxHashMap<BigramId, i64>,
    pub(crate) total_bigram_count: i64,
    pub(crate) bigrams_to_locations: FxHashMap<BigramId, BigramLocations>,
    pub(crate) left_lex_freqs: FxHashMap<LexemeId, i64>,
    pub(crate) right_lex_freqs: FxHashMap<LexemeId, i64>,
}

impl BigramData {
    pub(crate) fn from_lexemes(lexeme_data: &LexemeData, lexeme_store: &LexemeStore) -> Self {
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

    pub(crate) fn add_bigram(&mut self, bigram: BigramId, delta: i64) {
        *self.left_lex_freqs.entry(bigram.left).or_insert(0) += delta;
        *self.right_lex_freqs.entry(bigram.right).or_insert(0) += delta;
        *self.bigrams_to_freqs.entry(bigram).or_insert(0) += delta;
        self.total_bigram_count += delta;
    }

    pub(crate) fn add_location(&mut self, bigram: BigramId, location: Location) {
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

    pub(crate) fn remove_locations_batch(
        &mut self,
        removals: &FxHashMap<BigramId, FxHashSet<Location>>,
    ) {
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

    pub(crate) fn add_locations_batch(&mut self, additions: &FxHashMap<BigramId, Vec<Location>>) {
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

    pub(crate) fn locations_for_bigram(&self, bigram: BigramId) -> Vec<Location> {
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

    pub(crate) fn segment_range_for_bigram(&self, bigram: BigramId) -> usize {
        let Some(locations) = self.bigrams_to_locations.get(&bigram) else {
            return 0;
        };

        let mut segments = FxHashSet::default();
        for (line_ix, _) in locations {
            segments.insert(*line_ix);
        }
        segments.len()
    }

    pub(crate) fn maybe_remove_bigram(&mut self, bigram: BigramId) {
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

    pub(crate) fn maybe_remove_lr_lexemes(&mut self, bigram: BigramId) {
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
