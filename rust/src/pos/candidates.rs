use crate::scoring::{compare_candidate_rank, score_ll_npmi, CandidateRank};
use crate::types::SelectionMethod;
use rustc_hash::{FxHashMap, FxHashSet};
use std::cmp::Ordering;
use std::collections::BTreeSet;

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(super) enum MatchKind {
    Support,
    Full,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(super) struct CandidateKey {
    pub(super) left: Vec<String>,
    pub(super) right: Vec<String>,
    pub(super) kind: MatchKind,
}

impl CandidateKey {
    fn merged(&self) -> Vec<String> {
        self.left.iter().chain(&self.right).cloned().collect()
    }

    fn stable_identity(&self) -> PosStableIdentity<'_> {
        PosStableIdentity {
            prefer_full: self.kind == MatchKind::Full,
            left: &self.left,
            right: &self.right,
        }
    }
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(super) struct Occurrence {
    pub(super) sentence: usize,
    pub(super) start: usize,
    pub(super) end: usize,
}

#[derive(Clone, Debug)]
pub(super) struct Candidate {
    pub(super) key: CandidateKey,
    pub(super) merged: Vec<String>,
    pub(super) score: f64,
    pub(super) frequency: i64,
    pub(super) occurrences: Vec<Occurrence>,
}

impl Candidate {
    pub(super) fn compare(&self, other: &Self) -> Ordering {
        compare_candidate_rank(
            CandidateRank {
                score: self.score,
                frequency: self.frequency,
                merged: &self.merged,
                identity: &self.key.stable_identity(),
            },
            CandidateRank {
                score: other.score,
                frequency: other.frequency,
                merged: &other.merged,
                identity: &other.key.stable_identity(),
            },
        )
    }
}

#[derive(Eq, PartialEq)]
struct PosStableIdentity<'a> {
    prefer_full: bool,
    left: &'a [String],
    right: &'a [String],
}

impl Ord for PosStableIdentity<'_> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.prefer_full
            .cmp(&other.prefer_full)
            .then_with(|| other.left.cmp(self.left))
            .then_with(|| other.right.cmp(self.right))
    }
}

impl PartialOrd for PosStableIdentity<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Default)]
struct CandidateOccurrences {
    raw: BTreeSet<Occurrence>,
    clean: Vec<Occurrence>,
}

#[derive(Default)]
pub(super) struct PosCandidateIndex {
    occurrences: FxHashMap<CandidateKey, CandidateOccurrences>,
    left_frequencies: FxHashMap<Vec<String>, i64>,
    right_frequencies: FxHashMap<Vec<String>, i64>,
    total: i64,
    dirty: FxHashSet<CandidateKey>,
}

impl PosCandidateIndex {
    pub(super) fn add(&mut self, key: CandidateKey, occurrence: Occurrence) {
        if self
            .occurrences
            .entry(key.clone())
            .or_default()
            .raw
            .insert(occurrence)
        {
            self.dirty.insert(key);
        }
    }

    pub(super) fn remove(&mut self, key: &CandidateKey, occurrence: &Occurrence) {
        let Some(locations) = self.occurrences.get_mut(key) else {
            return;
        };
        if locations.raw.remove(occurrence) {
            self.dirty.insert(key.clone());
        }
    }

    pub(super) fn refresh(&mut self) {
        for key in std::mem::take(&mut self.dirty) {
            let (old_frequency, new_frequency, remove_entry) = {
                let locations = self
                    .occurrences
                    .get_mut(&key)
                    .expect("dirty candidate must have occurrence state");
                let old_frequency = locations.clean.len() as i64;
                locations.clean = clean_occurrences(&locations.raw);
                (
                    old_frequency,
                    locations.clean.len() as i64,
                    locations.raw.is_empty(),
                )
            };

            self.adjust_marginals(&key, new_frequency - old_frequency);
            if remove_entry {
                self.occurrences.remove(&key);
            }
        }
    }

    pub(super) fn select(&self, method: SelectionMethod, min_count: i64) -> Option<Candidate> {
        self.occurrences
            .iter()
            .filter_map(|(key, locations)| {
                let frequency = locations.clean.len() as i64;
                if frequency < min_count {
                    return None;
                }
                let score = score_ll_npmi(
                    method,
                    frequency,
                    *self.left_frequencies.get(&key.left).unwrap_or(&0),
                    *self.right_frequencies.get(&key.right).unwrap_or(&0),
                    self.total,
                );
                (score != f64::NEG_INFINITY).then(|| Candidate {
                    key: key.clone(),
                    merged: key.merged(),
                    score,
                    frequency,
                    occurrences: locations.clean.clone(),
                })
            })
            .max_by(Candidate::compare)
    }

    fn adjust_marginals(&mut self, key: &CandidateKey, delta: i64) {
        if delta == 0 {
            return;
        }
        adjust_frequency(&mut self.left_frequencies, &key.left, delta);
        adjust_frequency(&mut self.right_frequencies, &key.right, delta);
        self.total += delta;
        debug_assert!(self.total >= 0);
    }
}

fn adjust_frequency(frequencies: &mut FxHashMap<Vec<String>, i64>, key: &[String], delta: i64) {
    let frequency = frequencies.entry(key.to_vec()).or_default();
    *frequency += delta;
    debug_assert!(*frequency >= 0);
    if *frequency == 0 {
        frequencies.remove(key);
    }
}

fn clean_occurrences(raw: &BTreeSet<Occurrence>) -> Vec<Occurrence> {
    let mut clean = Vec::with_capacity(raw.len());
    let mut previous_sentence = usize::MAX;
    let mut next_valid = 0;
    for occurrence in raw {
        if occurrence.sentence != previous_sentence {
            previous_sentence = occurrence.sentence;
            next_valid = 0;
        }
        if occurrence.start >= next_valid {
            next_valid = occurrence.end;
            clean.push(occurrence.clone());
        }
    }
    clean
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key() -> CandidateKey {
        CandidateKey {
            left: vec!["a".to_string()],
            right: vec!["a".to_string()],
            kind: MatchKind::Full,
        }
    }

    #[test]
    fn removing_an_overlap_promotes_the_next_occurrence() {
        let key = key();
        let first = Occurrence {
            sentence: 0,
            start: 0,
            end: 2,
        };
        let overlap = Occurrence {
            sentence: 0,
            start: 1,
            end: 3,
        };
        let mut index = PosCandidateIndex::default();
        index.add(key.clone(), first.clone());
        index.add(key.clone(), overlap.clone());
        index.refresh();
        assert_eq!(
            index
                .select(SelectionMethod::Frequency, 0)
                .unwrap()
                .occurrences
                .as_slice(),
            std::slice::from_ref(&first)
        );

        index.remove(&key, &first);
        index.refresh();
        assert_eq!(
            index
                .select(SelectionMethod::Frequency, 0)
                .unwrap()
                .occurrences,
            [overlap]
        );
    }
}
