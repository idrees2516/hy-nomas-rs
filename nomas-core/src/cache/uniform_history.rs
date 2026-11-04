//! Archived-history frame sampling — port of `uniform_history_cache.py`'s
//! sampling functions (exact integer arithmetic preserved).

use crate::config::HistorySampling;
use crate::error::Error;
use crate::Result;

/// Port of `uniform_frame_indices`: sorted, endpoint-inclusive integer grid
/// over the complete history.
pub fn uniform_frame_indices(total_frames: usize, past_frames: usize) -> Vec<usize> {
    let count = total_frames.min(past_frames);
    match count {
        0 => Vec::new(),
        1 => vec![total_frames - 1],
        _ => (0..count)
            .map(|i| i * (total_frames - 1) / (count - 1))
            .collect(),
    }
}

/// Port of `recent_mix_uniform_frame_indices`: keep `past_frames // 2`
/// contiguous recent frames and uniformly sample the older prefix.
pub fn recent_mix_uniform_frame_indices(total_frames: usize, past_frames: usize) -> Vec<usize> {
    if total_frames <= past_frames {
        return (0..total_frames).collect();
    }
    let recent = past_frames / 2;
    let older_end = total_frames - recent;
    let mut ids = uniform_frame_indices(older_end, past_frames - recent);
    ids.extend(older_end..total_frames);
    ids
}

/// Port of `temporal_context_frame_indices`: power-spaced distances from the
/// newest frame, returned oldest first.
///
/// `d_i = i + rint((T - N) * u_i^gamma)` with `u_i = i / max(N-1, 1)`; the
/// original frame positions are `T - 1 - d_i`. `gamma > 1` biases selection
/// towards recent frames; `gamma = 1` is evenly spaced.
pub fn temporal_context_frame_indices(
    total_frames: usize,
    past_frames: usize,
    gamma: f64,
) -> Vec<usize> {
    if !gamma.is_finite() || gamma <= 0.0 {
        return Vec::new();
    }
    let count = total_frames.min(past_frames);
    match count {
        0 => Vec::new(),
        1 => vec![total_frames - 1],
        _ if count == total_frames => (0..total_frames).collect(),
        _ => {
            let slack = total_frames - count;
            let mut distances = Vec::with_capacity(count);
            for i in 0..count {
                let u = i as f64 / (count - 1) as f64;
                let offset = (slack as f64 * u.powf(gamma)).round();
                // Keep the mathematical bounds exact even for huge histories.
                let offset = offset.clamp(0.0, slack as f64) as usize;
                distances.push(i + offset);
            }
            distances.reverse();
            distances
                .into_iter()
                .map(|d| total_frames - 1 - d)
                .collect()
        }
    }
}

/// Validate + dispatch — port of `history_frame_indices`.
///
/// Guarantees (checked, matching the upstream `RuntimeError`s): the count is
/// `min(total, past)`, indices are unique, in-range and strictly chronological.
pub fn history_frame_indices(
    total_frames: usize,
    past_frames: usize,
    sampling: &HistorySampling,
) -> Result<Vec<usize>> {
    let gamma = match sampling {
        HistorySampling::TemporalContext { gamma } => *gamma,
        _ => 2.0,
    };
    if matches!(sampling, HistorySampling::TemporalContext { gamma } if {
        let g: f64 = *gamma;
        !g.is_finite() || g <= 0.0
    }) {
        return Err(Error::InvalidConfig(format!(
            "temporal_context gamma must be finite and positive, got {gamma}"
        )));
    }
    let ids = match sampling {
        HistorySampling::Uniform => uniform_frame_indices(total_frames, past_frames),
        HistorySampling::RecentMixUniform => {
            recent_mix_uniform_frame_indices(total_frames, past_frames)
        }
        HistorySampling::TemporalContext { .. } => {
            temporal_context_frame_indices(total_frames, past_frames, gamma)
        }
        HistorySampling::Recent => (0..total_frames.min(past_frames))
            .map(|i| total_frames - total_frames.min(past_frames) + i)
            .collect(),
    };
    let expected = total_frames.min(past_frames);
    if ids.len() != expected {
        return Err(Error::Msg(format!(
            "history sampling {sampling} selected {} frames; expected {expected}",
            ids.len()
        )));
    }
    for &f in &ids {
        if f >= total_frames {
            return Err(Error::Msg(format!(
                "history sampling {sampling} produced out-of-range frame {f} (total {total_frames})"
            )));
        }
    }
    if ids.windows(2).any(|w| w[0] >= w[1]) {
        return Err(Error::Msg(format!(
            "history sampling {sampling} must return chronological, unique frame indices"
        )));
    }
    Ok(ids)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_endpoints_inclusive() {
        assert_eq!(uniform_frame_indices(10, 4), vec![0, 3, 6, 9]);
        assert_eq!(uniform_frame_indices(0, 4), Vec::<usize>::new());
        assert_eq!(uniform_frame_indices(1, 4), vec![0]);
        assert_eq!(uniform_frame_indices(5, 10), vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn recent_mix_keeps_contiguous_tail() {
        let ids = recent_mix_uniform_frame_indices(20, 6);
        assert_eq!(ids.len(), 6);
        // recent = 3 -> last three are 17, 18, 19
        assert_eq!(&ids[3..], &[17, 18, 19]);
        // older prefix: uniform over 0..17 with budget 3 -> 0, 8, 16
        assert_eq!(&ids[..3], &[0, 8, 16]);
        assert!(ids.windows(2).all(|w| w[0] < w[1]));
    }

    #[test]
    fn short_history_kept_completely() {
        assert_eq!(recent_mix_uniform_frame_indices(4, 10), vec![0, 1, 2, 3]);
    }

    #[test]
    fn temporal_context_gamma_one_is_uniform() {
        let a = temporal_context_frame_indices(17, 5, 1.0);
        let b = uniform_frame_indices(17, 5);
        assert_eq!(a, b);
        assert_eq!(a, vec![0, 4, 8, 12, 16]);
    }

    #[test]
    fn temporal_context_biases_recent_for_large_gamma() {
        let ids = temporal_context_frame_indices(100, 5, 4.0);
        assert_eq!(ids.len(), 5);
        assert_eq!(ids.last(), Some(&99));
        // With gamma=4 most slack concentrates on the oldest frames.
        assert!(ids[0] < 40, "expected strong recent bias, got {ids:?}");
        assert!(ids.windows(2).all(|w| w[0] < w[1]));
    }

    #[test]
    fn validator_rejects_bad_gamma() {
        let e = history_frame_indices(10, 4, &HistorySampling::TemporalContext { gamma: 0.0 });
        assert!(e.is_err());
    }

    #[test]
    fn all_samplers_pass_validation_on_random_budgets() {
        let samplers = [
            HistorySampling::Uniform,
            HistorySampling::RecentMixUniform,
            HistorySampling::TemporalContext { gamma: 2.0 },
        ];
        for total in 0..40usize {
            for past in 1..12usize {
                for s in &samplers {
                    let ids = history_frame_indices(total, past, s)
                        .unwrap_or_else(|e| panic!("{s} t={total} p={past}: {e}"));
                    assert_eq!(ids.len(), total.min(past));
                    assert!(ids.windows(2).all(|w| w[0] < w[1]));
                }
            }
        }
    }
}
