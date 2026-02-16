use crate::bigram_data::BigramId;
use crate::types::{SelectionMethod, PARALLEL_SCORE_THRESHOLD, SCORE_ATOL, SCORE_RTOL, SMALL};
use rayon::prelude::*;

#[derive(Clone, Copy)]
pub(crate) struct CandidateScore {
    pub(crate) score: f64,
    pub(crate) frequency: i64,
}

#[derive(Clone, Copy)]
pub(crate) struct CandidateStats {
    pub(crate) bigram: BigramId,
    pub(crate) freq: Option<i64>,
    pub(crate) left_freq: i64,
    pub(crate) right_freq: i64,
}

fn safe_ll_term(observed: f64, expected: f64) -> f64 {
    if observed > 0.0 {
        observed * (((observed / (expected + SMALL)) + SMALL).ln())
    } else {
        0.0
    }
}

pub(crate) fn scores_close(a: f64, b: f64) -> bool {
    (a - b).abs() <= (SCORE_ATOL + SCORE_RTOL * b.abs())
}

fn is_close_default(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-8 + (1e-5 * b.abs())
}

fn coerce_score(score: f64) -> f64 {
    if score.is_finite() {
        score
    } else {
        f64::NEG_INFINITY
    }
}

pub(crate) fn score_ll_npmi(
    method: SelectionMethod,
    bigram_freq: i64,
    left_freq: i64,
    right_freq: i64,
    total_bigram_count: i64,
) -> f64 {
    if total_bigram_count == 0 {
        return f64::NEG_INFINITY;
    }

    let total = total_bigram_count as f64;

    match method {
        SelectionMethod::LogLikelihood => {
            let obs_a = bigram_freq as f64;
            let obs_b = left_freq as f64 - obs_a;
            let obs_c = right_freq as f64 - obs_a;
            let mut obs_d = total - obs_a - obs_b - obs_c;
            if obs_d < 0.0 {
                obs_d = 0.0;
            }

            let exp_a = ((obs_a + obs_b) * (obs_a + obs_c)) / total;
            let exp_b = ((obs_a + obs_b) * (obs_b + obs_d)) / total;
            let exp_c = ((obs_c + obs_d) * (obs_a + obs_c)) / total;
            let exp_d = ((obs_c + obs_d) * (obs_b + obs_d)) / total;

            let ll_a = safe_ll_term(obs_a, exp_a);
            let ll_b = safe_ll_term(obs_b, exp_b);
            let ll_c = safe_ll_term(obs_c, exp_c);
            let ll_d = safe_ll_term(obs_d, exp_d);
            let ll = 2.0 * (ll_a + ll_b + ll_c + ll_d);
            coerce_score(if obs_a > exp_a { ll } else { -ll })
        }
        SelectionMethod::Npmi => {
            let prob_ab = bigram_freq as f64 / total;
            let prob_a = left_freq as f64 / total;
            let prob_b = right_freq as f64 / total;
            let numerator = (prob_ab / (prob_a * prob_b)).ln();
            let denominator = -(prob_ab.ln());
            let npmi = if denominator > 0.0 {
                numerator / denominator
            } else {
                f64::NAN
            };
            let perfect_association =
                is_close_default(denominator, 0.0) && is_close_default(numerator, 0.0);
            coerce_score(if perfect_association { 1.0 } else { npmi })
        }
        SelectionMethod::Frequency => bigram_freq as f64,
    }
}

fn score_stats(
    method: SelectionMethod,
    min_count: i64,
    total_bigram_count: i64,
    stats: CandidateStats,
) -> (BigramId, Option<CandidateScore>) {
    let Some(freq) = stats.freq else {
        return (stats.bigram, None);
    };

    if freq < min_count {
        return (stats.bigram, None);
    }

    let score = match method {
        SelectionMethod::Frequency => freq as f64,
        _ => score_ll_npmi(
            method,
            freq,
            stats.left_freq,
            stats.right_freq,
            total_bigram_count,
        ),
    };

    if score == f64::NEG_INFINITY {
        (stats.bigram, None)
    } else {
        (
            stats.bigram,
            Some(CandidateScore {
                score,
                frequency: freq,
            }),
        )
    }
}

pub(crate) fn compute_scores(
    method: SelectionMethod,
    min_count: i64,
    total_bigram_count: i64,
    stats: &[CandidateStats],
) -> Vec<(BigramId, Option<CandidateScore>)> {
    if stats.len() >= PARALLEL_SCORE_THRESHOLD {
        stats
            .par_iter()
            .copied()
            .map(|item| score_stats(method, min_count, total_bigram_count, item))
            .collect()
    } else {
        stats
            .iter()
            .copied()
            .map(|item| score_stats(method, min_count, total_bigram_count, item))
            .collect()
    }
}
