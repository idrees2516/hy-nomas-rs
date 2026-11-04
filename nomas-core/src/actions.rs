//! Timed action conditioning — the `contained_actions` prompt structure.
//!
//! An upstream prompt carries structured actions
//! `({start_time}, {end_time}, {action_description})`; during segment
//! generation each action is resolved onto the segment's latent-frame timeline
//! and restricts which video tokens may *see* the action's text tokens (and
//! vice versa). This is the "intra-segment per-action masking" of
//! `nomas_cpt.py`'s `_build_intra_seg_cache_ranges`.

use serde::{Deserialize, Serialize};

/// One timed action inside a segment prompt.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ActionSpec {
    /// Action start time in seconds, relative to the *segment* start.
    pub start_time: f64,
    /// Action end time in seconds (exclusive), relative to the segment start.
    pub end_time: f64,
    /// Free-form action description, encoded as text tokens.
    pub description: String,
}

impl ActionSpec {
    /// Create an action covering the whole segment.
    pub fn whole_segment(description: impl Into<String>) -> Self {
        Self {
            start_time: 0.0,
            end_time: f64::INFINITY,
            description: description.into(),
        }
    }

    /// Resolve the action to an inclusive-exclusive latent-frame window
    /// `[first, last)` inside a segment of `frames` latent frames at
    /// `latent_fps` frames per second.
    ///
    /// Frames are sampled at their temporal center (`(f + 0.5) / fps`), which
    /// keeps sub-frame windows empty and makes a whole-segment action cover
    /// exactly `0..frames`.
    pub fn frame_window(&self, frames: usize, latent_fps: f64) -> (usize, usize) {
        let end = if self.end_time.is_finite() {
            self.end_time
        } else {
            f64::INFINITY
        };
        let covers = |f: usize| -> bool {
            let t = (f as f64 + 0.5) / latent_fps;
            t >= self.start_time && t < end
        };
        let first = (0..frames).find(|&f| covers(f));
        match first {
            None => (0, 0),
            Some(a) => {
                let last = (a..frames)
                    .rev()
                    .find(|&f| covers(f))
                    .expect("covered frame")
                    + 1;
                (a, last)
            }
        }
    }
}

/// Where the action tokens live inside a segment's packed text block.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ActionTokenRange {
    /// First action token (offset inside the text block).
    pub start: usize,
    /// One-past-the-end action token offset inside the text block.
    pub end: usize,
    /// First latent frame that may attend to these tokens.
    pub frame_first: usize,
    /// One-past-the-last latent frame.
    pub frame_last: usize,
}

/// Text-block layout for one segment: token ranges of the base prompt and of
/// each timed action.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TextBlockLayout {
    /// Tokens of the base prompt (always visible segment-wide).
    pub base_tokens: usize,
    /// Timed action token ranges with their resolved frame windows.
    pub actions: Vec<ActionTokenRange>,
    /// Total text tokens (`base + sum(actions)` + optional separator).
    pub total_tokens: usize,
}

impl TextBlockLayout {
    /// Lay out a text block given the base prompt's token count and
    /// per-action token counts.
    ///
    /// Layout: `[base prompt tokens | action 1 tokens | … | action k tokens
    /// | separator]`. The optional separator token (see
    /// [`crate::config::NomasConfig::text_separator`]) is appended last and is
    /// always visible.
    pub fn new(
        base_tokens: usize,
        actions: &[ActionSpec],
        action_tokens: &[usize],
        frames: usize,
        latent_fps: f64,
        separator: bool,
    ) -> Self {
        assert_eq!(
            actions.len(),
            action_tokens.len(),
            "action/token-count mismatch"
        );
        let mut ranges = Vec::with_capacity(actions.len());
        let mut cursor = base_tokens;
        for (spec, &count) in actions.iter().zip(action_tokens.iter()) {
            let (frame_first, frame_last) = spec.frame_window(frames, latent_fps);
            ranges.push(ActionTokenRange {
                start: cursor,
                end: cursor + count,
                frame_first,
                frame_last,
            });
            cursor += count;
        }
        Self {
            base_tokens,
            actions: ranges,
            total_tokens: cursor + usize::from(separator),
        }
    }

    /// Token ranges that are visible to latent frame `f` (offsets inside the
    /// text block). Base + separator tokens are always visible; action tokens
    /// only within their frame window.
    pub fn visible_text_ranges(&self, f: usize) -> Vec<(usize, usize)> {
        let mut out = Vec::new();
        if self.base_tokens > 0 {
            out.push((0, self.base_tokens));
        }
        for a in &self.actions {
            if f >= a.frame_first && f < a.frame_last && a.end > a.start {
                out.push((a.start, a.end));
            }
        }
        // Separator (last token) is always visible.
        if self.total_tokens
            > self.base_tokens + self.actions.iter().map(|a| a.end - a.start).sum::<usize>()
        {
            out.push((self.total_tokens - 1, self.total_tokens));
        }
        out
    }

    /// Offset of the first action token inside the text block (`action_start`
    /// in the upstream cache layout); `-1`-equivalent (None) when the block has
    /// no actions.
    pub fn action_start(&self) -> Option<usize> {
        self.actions.first().map(|a| a.start)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn whole_segment_covers_everything() {
        let a = ActionSpec::whole_segment("orbit left");
        assert_eq!(a.frame_window(7, 4.0), (0, 7));
        assert_eq!(a.frame_window(1, 4.0), (0, 1));
    }

    #[test]
    fn frame_window_uses_frame_centers() {
        let a = ActionSpec {
            start_time: 0.5,
            end_time: 1.25,
            description: "wave".into(),
        };
        // fps=2 -> centers at 0.25, 0.75, 1.25, 1.75...
        // covered: 0.75 (frame 1) only (1.25 is not < 1.25)
        assert_eq!(a.frame_window(4, 2.0), (1, 2));
    }

    #[test]
    fn empty_window_when_no_cover() {
        let a = ActionSpec {
            start_time: 10.0,
            end_time: 20.0,
            description: "x".into(),
        };
        assert_eq!(a.frame_window(3, 4.0), (0, 0));
    }

    #[test]
    fn layout_visibility() {
        let actions = vec![
            ActionSpec {
                start_time: 0.0,
                end_time: 1.0,
                description: "walk".into(),
            },
            ActionSpec {
                start_time: 1.0,
                end_time: 2.0,
                description: "run".into(),
            },
        ];
        let layout = TextBlockLayout::new(0, &actions, &[2, 2], 4, 2.0, true);
        assert_eq!(layout.total_tokens, 5);
        assert_eq!(layout.action_start(), Some(0));
        // frame 0 (center 0.25s): sees walk tokens [0,2) + separator [4,5)
        assert_eq!(layout.visible_text_ranges(0), vec![(0, 2), (4, 5)]);
        // frame 3 (center 1.75s): sees run tokens [2,4) + separator
        assert_eq!(layout.visible_text_ranges(3), vec![(2, 4), (4, 5)]);
    }

    #[test]
    fn serde_action() {
        let a = ActionSpec {
            start_time: 0.0,
            end_time: 2.5,
            description: "turn".into(),
        };
        let s = serde_json::to_string(&a).unwrap();
        let b: ActionSpec = serde_json::from_str(&s).unwrap();
        assert_eq!(a, b);
    }
}
