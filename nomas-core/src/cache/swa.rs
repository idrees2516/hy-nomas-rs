//! Sliding-window eviction with frame alignment — port of `swa_cache.py`
//! (`DenseVideo` records, `eviction_cut`, `trim_records`).
//!
//! The recent policy keeps the newest `capacity` tokens of the archive. Cuts
//! always land on frame boundaries, and a segment's text block is only kept
//! while at least one of its video frames survives — mirroring the upstream
//! `trim_records` semantics used by dense SWA.

/// Geometry of one archived segment, in archive coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DenseRecord {
    /// Global segment index.
    pub segment_index: usize,
    /// Global latent-frame index of the first frame.
    pub frame_start: usize,
    /// Number of latent frames.
    pub frame_count: usize,
    /// Video tokens (`frame_count * tokens_per_frame`).
    pub image_tokens: usize,
    /// Text tokens.
    pub text_tokens: usize,
}

impl DenseRecord {
    /// Total tokens.
    pub fn total_tokens(&self) -> usize {
        self.image_tokens + self.text_tokens
    }
}

/// The result of a recent-policy trim.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrimPlan {
    /// How many *frames* to drop from the head of each surviving record, keyed
    /// by archive position. Records absent from the map are kept intact.
    pub drop_frames: Vec<(usize, usize)>,
    /// Archive positions to drop entirely (all video evicted → text too).
    pub drop_records: Vec<usize>,
    /// Tokens removed in total.
    pub evicted_tokens: usize,
}

/// Decide the eviction cut that brings the archive within `capacity` tokens
/// while keeping complete frames (`eviction_cut` + `trim_records` upstream).
///
/// Eviction proceeds from the oldest segment:
/// 1. whole segments are dropped while the deficit covers them;
/// 2. otherwise whole frames are trimmed from the oldest segment's video;
/// 3. a segment whose video is fully trimmed loses its text block as well
///    (only when the budget still requires it — a partially-evicted segment
///    keeps its text, exactly like the upstream record trimming).
pub fn plan_recent_trim(
    records: &[DenseRecord],
    capacity: usize,
    tokens_per_frame: usize,
) -> TrimPlan {
    let mut total: usize = records.iter().map(|r| r.total_tokens()).sum();
    let mut drop_frames: Vec<(usize, usize)> = Vec::new();
    let mut drop_records: Vec<usize> = Vec::new();
    let mut evicted = 0usize;

    let mut idx = 0usize;
    while total > capacity && idx < records.len() {
        let rec = records[idx];
        let deficit = total - capacity;
        if deficit < rec.image_tokens {
            // Partial trim: whole frames off the head of this record's video
            // (frame-aligned, rounded up so the loop terminates).
            let frames = deficit.div_ceil(tokens_per_frame).min(rec.frame_count);
            let removed = frames * tokens_per_frame;
            drop_frames.push((idx, frames));
            let _ = total;
            evicted += removed;
            break;
        } else if total - rec.image_tokens <= capacity {
            // Dropping the whole video is enough: the segment keeps its text
            // block (it still belongs to a represented timeline).
            drop_frames.push((idx, rec.frame_count));
            let _ = total;
            evicted += rec.image_tokens;
            break;
        } else {
            // The whole record must go (video *and* text).
            drop_records.push(idx);
            total -= rec.total_tokens();
            evicted += rec.total_tokens();
            idx += 1;
        }
    }
    TrimPlan {
        drop_frames,
        drop_records,
        evicted_tokens: evicted,
    }
}

/// Apply a [`TrimPlan`]: returns the surviving records with adjusted
/// geometry. `tokens_per_frame` is the fallback geometry for records that
/// lost all frames.
pub fn apply_trim(
    records: &[DenseRecord],
    plan: &TrimPlan,
    tokens_per_frame: usize,
) -> Vec<DenseRecord> {
    let mut out = Vec::with_capacity(records.len());
    for (idx, rec) in records.iter().enumerate() {
        if plan.drop_records.contains(&idx) {
            continue;
        }
        if let Some(&(_, frames)) = plan.drop_frames.iter().find(|(i, _)| *i == idx) {
            let kept_frames = rec.frame_count - frames;
            let tpf = rec
                .frame_count
                .checked_div(1)
                .filter(|_| rec.frame_count > 0)
                .map(|_| rec.image_tokens / rec.frame_count)
                .unwrap_or(tokens_per_frame);
            out.push(DenseRecord {
                segment_index: rec.segment_index,
                frame_start: rec.frame_start + frames,
                frame_count: kept_frames,
                image_tokens: kept_frames * tpf,
                text_tokens: rec.text_tokens,
            });
        } else {
            out.push(*rec);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rec(seg: usize, frames: usize, tpf: usize, text: usize) -> DenseRecord {
        DenseRecord {
            segment_index: seg,
            frame_start: seg * frames,
            frame_count: frames,
            image_tokens: frames * tpf,
            text_tokens: text,
        }
    }

    #[test]
    fn no_trim_when_under_capacity() {
        let records = vec![rec(0, 2, 4, 3), rec(1, 2, 4, 3)];
        let plan = plan_recent_trim(&records, 100, 4);
        assert!(plan.drop_frames.is_empty() && plan.drop_records.is_empty());
        assert_eq!(plan.evicted_tokens, 0);
        assert_eq!(apply_trim(&records, &plan, 4).len(), 2);
    }

    #[test]
    fn frame_aligned_partial_trim() {
        // total = 2*8+2 + 2*8+2 = 36; capacity 30 -> deficit 6 -> 1 frame (8 tokens) from seg 0
        let records = vec![rec(0, 2, 8, 2), rec(1, 2, 8, 2)];
        let plan = plan_recent_trim(&records, 30, 8);
        assert_eq!(plan.drop_frames, vec![(0, 1)]);
        assert!(plan.drop_records.is_empty());
        let kept = apply_trim(&records, &plan, 4);
        assert_eq!(kept[0].frame_count, 1);
        assert_eq!(kept[0].image_tokens, 8);
        assert_eq!(
            kept[0].text_tokens, 2,
            "partially evicted segment keeps text"
        );
        assert_eq!(kept[1].frame_count, 2);
        let total: usize = kept.iter().map(|r| r.total_tokens()).sum();
        assert!(total <= 30);
    }

    #[test]
    fn whole_segment_eviction_drops_text() {
        // deficit covers segment 0 entirely
        let records = vec![rec(0, 2, 4, 3), rec(1, 2, 4, 3)];
        let plan = plan_recent_trim(&records, 11, 4);
        assert_eq!(plan.drop_records, vec![0]);
        let kept = apply_trim(&records, &plan, 4);
        assert_eq!(kept.len(), 1);
        assert_eq!(kept[0].segment_index, 1);
        assert_eq!(plan.evicted_tokens, 11);
    }

    #[test]
    fn video_gone_but_text_within_budget_is_kept() {
        // capacity 14: dropping record 0's video (8) brings 22 -> 14, so its
        // text block survives as a text-only record.
        let records = vec![rec(0, 2, 4, 3), rec(1, 2, 4, 3)];
        let plan = plan_recent_trim(&records, 14, 4);
        assert_eq!(
            plan.drop_frames,
            vec![(0, 2)],
            "drop the video, keep the text"
        );
        assert!(plan.drop_records.is_empty());
        let kept = apply_trim(&records, &plan, 4);
        assert_eq!(kept.len(), 2);
        assert_eq!(kept[0].frame_count, 0);
        assert_eq!(
            kept[0].text_tokens, 3,
            "text survives a fully-evicted video"
        );
        assert_eq!(kept[1].frame_count, 2);
        let total: usize = kept.iter().map(|r| r.total_tokens()).sum();
        assert_eq!(total, 14);
    }

    #[test]
    fn text_dropped_when_video_and_text_both_needed() {
        // capacity 13: dropping only record 0's video leaves 14 > 13, so the
        // whole record (text included) must go.
        let records = vec![rec(0, 2, 4, 3), rec(1, 2, 4, 3)];
        let plan = plan_recent_trim(&records, 13, 4);
        assert_eq!(plan.drop_records, vec![0]);
        let kept = apply_trim(&records, &plan, 4);
        assert_eq!(kept.len(), 1);
        assert_eq!(kept[0].segment_index, 1);
    }

    #[test]
    fn capacity_smaller_than_one_segment() {
        let records = vec![rec(0, 4, 4, 3)];
        let plan = plan_recent_trim(&records, 8, 4);
        let kept = apply_trim(&records, &plan, 4);
        // keep at most 2 frames of the only segment, no text
        assert!(kept.iter().map(|r| r.total_tokens()).sum::<usize>() <= 8);
        assert!(kept.iter().all(|r| r.frame_count >= 1));
    }
}
