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

fn score_log_likelihood(bigram_freq: f64, left_freq: f64, right_freq: f64, total: f64) -> f64 {
    let obs_a = bigram_freq;
    let obs_b = left_freq - obs_a;
    let obs_c = right_freq - obs_a;
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

fn score_npmi(bigram_freq: f64, left_freq: f64, right_freq: f64, total: f64) -> f64 {
    let prob_ab = bigram_freq / total;
    let prob_a = left_freq / total;
    let prob_b = right_freq / total;
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

fn score_logdice(bigram_freq: f64, left_freq: f64, right_freq: f64) -> f64 {
    let denom = (left_freq + right_freq).max(SMALL);
    coerce_score(14.0 + ((2.0 * bigram_freq) / denom).log2())
}

fn score_t_score(bigram_freq: f64, left_freq: f64, right_freq: f64, total: f64) -> f64 {
    let expected = (left_freq * right_freq) / total.max(SMALL);
    coerce_score((bigram_freq - expected) / bigram_freq.max(SMALL).sqrt())
}

fn delta_p_directed(f_ab: f64, f_a: f64, f_b: f64, n: f64) -> f64 {
    let p_b_given_a = f_ab / f_a.max(SMALL);
    let p_b_given_not_a = (f_b - f_ab).max(0.0) / (n - f_a).max(SMALL);
    p_b_given_a - p_b_given_not_a
}

fn score_delta_p_sym(bigram_freq: f64, left_freq: f64, right_freq: f64, total: f64) -> f64 {
    let ab = delta_p_directed(bigram_freq, left_freq, right_freq, total);
    let ba = delta_p_directed(bigram_freq, right_freq, left_freq, total);
    coerce_score(0.5 * (ab + ba))
}

pub(crate) fn score_association(
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
    let f_ab = bigram_freq as f64;
    let f_a = left_freq as f64;
    let f_b = right_freq as f64;

    match method {
        SelectionMethod::Frequency => f_ab,
        SelectionMethod::LogLikelihood => score_log_likelihood(f_ab, f_a, f_b, total),
        SelectionMethod::Npmi => score_npmi(f_ab, f_a, f_b, total),
        SelectionMethod::LogDice => score_logdice(f_ab, f_a, f_b),
        SelectionMethod::TScore => score_t_score(f_ab, f_a, f_b, total),
        SelectionMethod::DeltaP => score_delta_p_sym(f_ab, f_a, f_b, total),
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

    let score = score_association(
        method,
        freq,
        stats.left_freq,
        stats.right_freq,
        total_bigram_count,
    );

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
