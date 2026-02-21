use crate::bigram_data::{BigramData, BigramId};
use crate::engine::{
    apply_range_penalty, clean_bigram_locations, merge_winner, merged_lexeme_ids_for_bigram,
    merged_word_ids, winner_from_bigram_with_data, Engine, HeapEntry, StepStatus,
};
use crate::interner::{validate_token_vocabulary_size, Interner};
use crate::lexeme_data::LexemeData;
use crate::lexeme_store::{Lexeme, LexemeStore};
use crate::scoring::score_association;
use crate::types::{
    SearchStrategy, SelectionMethod, Splitter, StopwordPolicy, DEFAULT_RESCORE_INTERVAL,
};
use proptest::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::smallvec;
use std::collections::BinaryHeap;

fn build_engine(
    corpus: Vec<&str>,
    method: SelectionMethod,
    min_count: i64,
    rescore_interval: usize,
) -> Engine {
    let corpus = corpus.into_iter().map(str::to_string).collect::<Vec<_>>();
    let (interner, corpus_ids, doc_boundaries) =
        Interner::from_documents(&corpus, Splitter::Delimiter(Some("\n")))
            .expect("failed to tokenize corpus");

    let mut lexeme_store = LexemeStore::default();
    let lexemes = LexemeData::from_corpus(&corpus_ids, doc_boundaries, &mut lexeme_store);
    let bigrams = BigramData::from_lexemes(&lexemes, &lexeme_store);

    let mut engine = Engine {
        interner,
        lexeme_store,
        lexemes,
        bigrams,
        segment_delimiter: "\n".to_string(),
        method,
        min_count,
        rescore_interval,
        stopword_token_ids: FxHashSet::default(),
        stopword_policy: StopwordPolicy::None,
        block_punct_only: true,
        min_range: 1,
        range_alpha: 0.0,
        min_p_ab: None,
        min_p_ba: None,
        min_merge_count: 1,
        search_strategy: SearchStrategy::Greedy,
        beam_width: 1,
        beam_top_m: 8,
        candidate_scores: FxHashMap::default(),
        candidate_heap: BinaryHeap::new(),
        bigram_generation: FxHashMap::default(),
        generation_counter: 0,
        dirty_bigrams: FxHashSet::default(),
        iteration_counter: 0,
    };
    engine.refresh_candidate_state(true);
    engine
}

fn total_root_count(engine: &Engine) -> usize {
    (0..engine.lexemes.corpus_length())
        .map(|line_ix| {
            engine
                .lexemes
                .root_items_for_line(line_ix, &engine.lexeme_store)
                .len()
        })
        .sum()
}

#[test]
fn interner_roundtrip() {
    let corpus = vec!["b a b".to_string()];
    let (interner, corpus_ids, doc_boundaries) =
        Interner::from_documents(&corpus, Splitter::Delimiter(Some("\n")))
            .expect("failed to tokenize corpus");
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
    assert_eq!(doc_boundaries, vec![0, 1]);
}

#[test]
fn token_vocabulary_size_overflow_returns_error() {
    assert!(validate_token_vocabulary_size((u32::MAX as usize).saturating_add(2)).is_err());
}

#[test]
fn deterministic_tie_break_prefers_frequency_then_lexicographic() {
    let mut engine = build_engine(
        vec!["a d", "a c", "a b", "a b"],
        SelectionMethod::Frequency,
        0,
        DEFAULT_RESCORE_INTERVAL,
    );

    let StepStatus::Winner(step) = engine.step_internal(None) else {
        panic!("expected winner");
    };
    assert_eq!(
        engine.token_ids_to_strings(&engine.lexeme_store.get(step.winner.merged_lexeme).word),
        vec!["a", "b"]
    );
}

#[test]
fn engine_run_matches_repeated_step_for_frequency() {
    let corpus = vec!["a a a a"];
    let mut run_engine = build_engine(
        corpus.clone(),
        SelectionMethod::Frequency,
        0,
        DEFAULT_RESCORE_INTERVAL,
    );
    let mut step_engine = build_engine(
        corpus,
        SelectionMethod::Frequency,
        0,
        DEFAULT_RESCORE_INTERVAL,
    );

    let (_, run_payloads, _, _) = run_engine.run_internal(3, None);

    let mut step_payloads = Vec::new();
    for _ in 0..3 {
        match step_engine.step_internal(None) {
            StepStatus::Winner(step_data) => {
                step_payloads.push(step_engine.step_result(step_data));
            }
            _ => break,
        }
    }

    assert_eq!(run_payloads, step_payloads);
}

#[test]
fn merge_winner_updates_all_indexes_consistently() {
    let mut engine = build_engine(
        vec!["a b a b"],
        SelectionMethod::Frequency,
        0,
        DEFAULT_RESCORE_INTERVAL,
    );

    let left = engine.lexeme_store.intern(Lexeme {
        word: smallvec![engine.interner.id_for("a")],
        ix: 0,
    });
    let right = engine.lexeme_store.intern(Lexeme {
        word: smallvec![engine.interner.id_for("b")],
        ix: 0,
    });

    let winner_bigram = BigramId { left, right };
    let merged_ids = merged_lexeme_ids_for_bigram(winner_bigram, &mut engine.lexeme_store);
    let winner = winner_from_bigram_with_data(
        winner_bigram,
        merged_ids[0],
        merged_ids.len(),
        &engine.bigrams,
    );
    assert_eq!(winner.bigram_locations, vec![(0, 0), (0, 2)]);

    let touched = merge_winner(
        &winner,
        &merged_ids,
        &mut engine.lexemes,
        &engine.lexeme_store,
        &mut engine.bigrams,
    );

    let merged_bigram = BigramId {
        left: merged_ids[0],
        right: merged_ids[0],
    };
    let bridge_bigram = BigramId {
        left: right,
        right: left,
    };

    assert_eq!(
        touched,
        FxHashSet::from_iter([winner_bigram, bridge_bigram, merged_bigram])
    );
    assert_eq!(
        engine.lexemes.locations_to_lexemes[0],
        vec![merged_ids[0], merged_ids[1], merged_ids[0], merged_ids[1]]
    );
    assert_eq!(
        engine
            .lexemes
            .lexemes_to_locations
            .get(&merged_ids[0])
            .cloned(),
        Some(FxHashSet::from_iter([(0, 0), (0, 2)]))
    );
    assert_eq!(
        engine
            .lexemes
            .lexemes_to_locations
            .get(&merged_ids[1])
            .cloned(),
        Some(FxHashSet::from_iter([(0, 1), (0, 3)]))
    );

    assert_eq!(
        engine.bigrams.bigrams_to_freqs,
        FxHashMap::from_iter([(merged_bigram, 1)])
    );

    let mut locations = engine.bigrams.locations_for_bigram(merged_bigram);
    locations.sort_unstable();
    assert_eq!(locations, vec![(0, 0)]);
    assert_eq!(engine.bigrams.total_bigram_count, 1);
    assert_eq!(
        engine.bigrams.left_lex_freqs,
        FxHashMap::from_iter([(merged_ids[0], 1)])
    );
    assert_eq!(
        engine.bigrams.right_lex_freqs,
        FxHashMap::from_iter([(merged_ids[0], 1)])
    );
}

#[test]
fn min_score_blocks_low_score_winner() {
    let mut engine = build_engine(
        vec!["a b c"],
        SelectionMethod::Frequency,
        0,
        DEFAULT_RESCORE_INTERVAL,
    );
    let status = engine.step_internal(Some(10.0));
    let StepStatus::BelowMinScore(score) = status else {
        panic!("expected below-min-score status");
    };
    assert_eq!(score, 1.0);
}

#[test]
fn range_penalty_sign_preserving_for_negative_scores() {
    let positive = apply_range_penalty(10.0, 0.5, 1.0);
    let negative = apply_range_penalty(-10.0, 0.5, 1.0);
    assert_eq!(positive, 5.0);
    assert_eq!(negative, -20.0);
}

#[test]
fn min_range_filters_single_segment_frequency_winner() {
    let mut engine = build_engine(
        vec!["x y x y x y", "a b", "a b"],
        SelectionMethod::Frequency,
        0,
        DEFAULT_RESCORE_INTERVAL,
    );
    engine.min_range = 2;
    engine.refresh_candidate_state(true);

    let StepStatus::Winner(step_data) = engine.step_internal(None) else {
        panic!("expected winner");
    };

    let winner_tokens =
        engine.token_ids_to_strings(&engine.lexeme_store.get(step_data.winner.merged_lexeme).word);
    assert_eq!(winner_tokens, vec!["a", "b"]);
}

#[test]
fn range_alpha_prefers_dispersed_frequency_candidate() {
    let concentrated = vec!["x y"; 20].join(" ");
    let mut corpus = vec![concentrated];
    corpus.extend(vec!["a b".to_string(); 10]);
    let corpus_refs = corpus.iter().map(String::as_str).collect::<Vec<_>>();

    let mut engine = build_engine(
        corpus_refs,
        SelectionMethod::Frequency,
        0,
        DEFAULT_RESCORE_INTERVAL,
    );
    engine.range_alpha = 1.0;
    engine.refresh_candidate_state(true);

    let StepStatus::Winner(step_data) = engine.step_internal(None) else {
        panic!("expected winner");
    };

    let winner_tokens =
        engine.token_ids_to_strings(&engine.lexeme_store.get(step_data.winner.merged_lexeme).word);
    assert_eq!(winner_tokens, vec!["a", "b"]);
}

#[test]
fn scoring_methods_formula_sanity() {
    let freq = 10;
    let left = 20;
    let right = 25;
    let total = 200;

    let logdice = score_association(SelectionMethod::LogDice, freq, left, right, total);
    let expected_logdice = 14.0 + ((2.0 * freq as f64) / ((left + right) as f64)).log2();
    assert!((logdice - expected_logdice).abs() < 1e-12);

    let t_score = score_association(SelectionMethod::TScore, freq, left, right, total);
    let expected_t_score =
        ((freq as f64) - ((left as f64 * right as f64) / total as f64)) / (freq as f64).sqrt();
    assert!((t_score - expected_t_score).abs() < 1e-12);

    let delta_p = score_association(SelectionMethod::DeltaP, freq, left, right, total);
    let p_b_given_a = freq as f64 / left as f64;
    let p_b_given_not_a = (right - freq) as f64 / (total - left) as f64;
    let p_a_given_b = freq as f64 / right as f64;
    let p_a_given_not_b = (left - freq) as f64 / (total - right) as f64;
    let expected_delta_p =
        0.5 * ((p_b_given_a - p_b_given_not_a) + (p_a_given_b - p_a_given_not_b));
    assert!((delta_p - expected_delta_p).abs() < 1e-12);
}

#[test]
fn probability_gates_can_filter_all_candidates() {
    let mut engine = build_engine(
        vec!["a b a c a d"],
        SelectionMethod::Frequency,
        0,
        DEFAULT_RESCORE_INTERVAL,
    );
    engine.min_p_ab = Some(0.9);
    engine.min_p_ba = Some(0.9);
    engine.refresh_candidate_state(true);

    let status = engine.step_internal(None);
    assert!(matches!(status, StepStatus::NoCandidate));
}

#[test]
fn min_merge_count_filters_candidates_in_selection() {
    let mut engine = build_engine(
        vec!["a b a b a b"],
        SelectionMethod::Frequency,
        0,
        DEFAULT_RESCORE_INTERVAL,
    );
    engine.min_merge_count = 4;
    engine.refresh_candidate_state(true);

    let status = engine.step_internal(None);
    assert!(matches!(status, StepStatus::NoCandidate));
}

#[test]
fn beam_width_one_matches_greedy_path() {
    let corpus = vec!["a b a b c d c d a b"];
    let mut greedy = build_engine(
        corpus.clone(),
        SelectionMethod::Frequency,
        0,
        DEFAULT_RESCORE_INTERVAL,
    );
    let mut beam = build_engine(
        corpus,
        SelectionMethod::Frequency,
        0,
        DEFAULT_RESCORE_INTERVAL,
    );
    beam.search_strategy = SearchStrategy::Beam;
    beam.beam_width = 1;
    beam.beam_top_m = 4;

    let (_, greedy_winners, _, _) = greedy.run_internal(3, None);
    let (_, beam_winners, _, _) = beam.run_internal(3, None);
    assert_eq!(beam_winners, greedy_winners);
}

#[test]
fn beam_search_is_deterministic_under_ties() {
    let corpus = vec!["a b", "c d", "a b", "c d"];

    let mut beam_a = build_engine(
        corpus.clone(),
        SelectionMethod::Frequency,
        0,
        DEFAULT_RESCORE_INTERVAL,
    );
    beam_a.search_strategy = SearchStrategy::Beam;
    beam_a.beam_width = 3;
    beam_a.beam_top_m = 4;

    let mut beam_b = build_engine(
        corpus,
        SelectionMethod::Frequency,
        0,
        DEFAULT_RESCORE_INTERVAL,
    );
    beam_b.search_strategy = SearchStrategy::Beam;
    beam_b.beam_width = 3;
    beam_b.beam_top_m = 4;

    let (_, winners_a, _, _) = beam_a.run_internal(2, None);
    let (_, winners_b, _, _) = beam_b.run_internal(2, None);
    assert_eq!(winners_a, winners_b);
}

#[test]
fn beam_cumulative_objective_is_not_worse_than_greedy() {
    let corpus = vec!["a b a c a b a c d e d e"];

    let mut greedy = build_engine(
        corpus.clone(),
        SelectionMethod::LogLikelihood,
        0,
        DEFAULT_RESCORE_INTERVAL,
    );
    let mut beam = build_engine(
        corpus,
        SelectionMethod::LogLikelihood,
        0,
        DEFAULT_RESCORE_INTERVAL,
    );
    beam.search_strategy = SearchStrategy::Beam;
    beam.beam_width = 3;
    beam.beam_top_m = 8;

    let (_, greedy_winners, _, _) = greedy.run_internal(4, None);
    let (_, beam_winners, _, _) = beam.run_internal(4, None);

    let greedy_score: f64 = greedy_winners.iter().map(|winner| winner.score).sum();
    let beam_score: f64 = beam_winners.iter().map(|winner| winner.score).sum();
    assert!(beam_score >= greedy_score);
}

#[test]
fn rescore_interval_one_forces_full_rescore_every_step() {
    let mut engine = build_engine(vec!["a b a c a b"], SelectionMethod::Npmi, 0, 1);

    let initial_count = engine.bigrams.bigrams_to_freqs.len();
    assert_eq!(engine.candidate_scores.len(), initial_count);

    let StepStatus::Winner(_) = engine.step_internal(None) else {
        panic!("expected winner");
    };

    engine.refresh_candidate_state(false);

    let eligible = engine
        .bigrams
        .bigrams_to_freqs
        .iter()
        .filter(|(_, freq)| **freq >= engine.min_count)
        .count();
    assert_eq!(engine.candidate_scores.len(), eligible);
}

#[test]
fn candidate_heap_skips_stale_frequency_entries() {
    let mut engine = build_engine(
        vec!["a b a b"],
        SelectionMethod::Frequency,
        0,
        DEFAULT_RESCORE_INTERVAL,
    );

    let bigram = *engine
        .bigrams
        .bigrams_to_freqs
        .keys()
        .next()
        .expect("expected at least one bigram");

    engine.candidate_heap.push(HeapEntry {
        score: 9999.0,
        frequency: 9999,
        merged_word: merged_word_ids(&engine.lexeme_store, bigram),
        bigram,
        generation: 0,
    });

    let Some((_winner, candidate)) = engine.select_candidate() else {
        panic!("expected candidate")
    };
    assert!(candidate.frequency < 9999);
}

#[test]
fn clean_bigram_locations_filters_overlaps_per_line() {
    let locations = vec![(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 4)];
    let cleaned = clean_bigram_locations(locations, 2);
    assert_eq!(cleaned, vec![(0, 0), (0, 2), (1, 0), (1, 4)]);
}

#[test]
fn clean_bigram_locations_is_line_local() {
    let locations = vec![(0, 0), (0, 3), (1, 1), (1, 2)];
    let cleaned = clean_bigram_locations(locations, 3);
    assert_eq!(cleaned, vec![(0, 0), (0, 3), (1, 1)]);
}

#[test]
fn merged_lexeme_ids_for_bigram_reuses_existing_ids() {
    let mut engine = build_engine(
        vec!["a b a b"],
        SelectionMethod::Frequency,
        0,
        DEFAULT_RESCORE_INTERVAL,
    );
    let left = engine.lexeme_store.intern(Lexeme {
        word: smallvec![engine.interner.id_for("a")],
        ix: 0,
    });
    let right = engine.lexeme_store.intern(Lexeme {
        word: smallvec![engine.interner.id_for("b")],
        ix: 0,
    });
    let bigram = BigramId { left, right };

    let pre_len = engine.lexeme_store.id_to_lexeme.len();
    let first_ids = merged_lexeme_ids_for_bigram(bigram, &mut engine.lexeme_store);
    let after_first_len = engine.lexeme_store.id_to_lexeme.len();
    let second_ids = merged_lexeme_ids_for_bigram(bigram, &mut engine.lexeme_store);
    let after_second_len = engine.lexeme_store.id_to_lexeme.len();

    assert_eq!(first_ids, second_ids);
    assert_eq!(after_first_len, pre_len + first_ids.len());
    assert_eq!(after_second_len, after_first_len);
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(48))]

    #[test]
    fn total_root_count_changes_only_on_winner(corpus in "[a-c ]{5,30}") {
        let mut engine = build_engine(
            vec![corpus.as_str()],
            SelectionMethod::Frequency,
            0,
            DEFAULT_RESCORE_INTERVAL,
        );

        let before = total_root_count(&engine);
        let status = engine.step_internal(None);
        let after = total_root_count(&engine);

        match status {
            StepStatus::Winner(_) => prop_assert!(after < before),
            StepStatus::NoCandidate | StepStatus::BelowMinScore(_) => {
                prop_assert_eq!(after, before)
            }
        }
    }
}
