//! Action-aware causal attention masks.
//!
//! # Token universe of one denoising step
//!
//! ```text
//! [ history tokens (working set) | current video tokens | current text tokens ]
//!    0 ............................ Lh .................. Lh + Nv .... Lh + Nv + Nt
//! ```
//!
//! # Semantics (mirrors `nomas_cpt.py`)
//!
//! * **Inter-segment causal**: every current token may attend to every
//!   history token; history tokens are keys-only.
//! * **Intra-segment bidirectional**: current video and text tokens attend to
//!   each other (the CPT "bidirectional attention between video and text
//!   tokens"), refined by the intra-segment causality mode:
//!     - [`IntraCausality::Full`] — chunk mode `1` (upstream default),
//!     - [`IntraCausality::Frame`] — chunk mode `0`, strict per-frame
//!       causality inside the segment,
//!     - [`IntraCausality::Groups(k)`] — chunk mode `k > 1`, bidirectional
//!       within each group of `k` latent frames, causal across groups.
//! * **Per-action time windows** (the "intra-segment per-action masking"):
//!   action text tokens are visible only to video tokens of frames inside the
//!   action's `[start_time, end_time)` window — symmetrically, action tokens
//!   as queries only attend to in-window video tokens (plus text and history).
//! * History action tokens (from earlier segments) act as regular text for
//!   future queries: their time windows were relative to their own segment.

use candle_core::{Device, Tensor};

use crate::actions::TextBlockLayout;
use crate::cache::SegmentLayout;
use crate::Result;

/// Intra-segment causality mode (upstream `inference_chunk_size`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntraCausality {
    /// `chunk_size = 1`: full bidirectional attention inside the segment.
    Full,
    /// `chunk_size = 0`: per-frame causal — a frame sees itself and earlier
    /// frames only.
    Frame,
    /// `chunk_size = k > 1`: bidirectional inside groups of `k` frames,
    /// causal across groups.
    Groups(usize),
}

impl IntraCausality {
    /// Build from the upstream `inference_chunk_size` value.
    pub fn from_chunk_size(size: usize) -> Self {
        match size {
            0 => Self::Frame,
            1 => Self::Full,
            k => Self::Groups(k),
        }
    }
}

/// Everything the mask builder needs to know about one denoising step.
#[derive(Debug, Clone)]
pub struct MaskSpec<'a> {
    /// Layout of the history working set (from the KV cache).
    pub history_layout: &'a [SegmentLayout],
    /// Number of history tokens.
    pub history_len: usize,
    /// Video tokens in the current step.
    pub video_tokens: usize,
    /// Latent frames covered by the current step.
    pub frames: usize,
    /// Tokens per latent frame.
    pub tokens_per_frame: usize,
    /// Text block layout of the current step (actions + separator).
    pub text_layout: &'a TextBlockLayout,
    /// Intra-segment causality.
    pub causality: IntraCausality,
}

impl<'a> MaskSpec<'a> {
    /// Total key length.
    pub fn key_len(&self) -> usize {
        self.history_len + self.video_tokens + self.text_layout.total_tokens
    }

    /// Total query length (current tokens only).
    pub fn query_len(&self) -> usize {
        self.video_tokens + self.text_layout.total_tokens
    }

    /// Frame index of a current video token.
    fn frame_of(&self, video_token: usize) -> usize {
        video_token / self.tokens_per_frame
    }

    fn group_of(&self, frame: usize) -> usize {
        match self.causality {
            IntraCausality::Groups(k) => frame / k,
            _ => frame,
        }
    }

    /// Whether video token `q` (frame `fq`) may see video token `k` (frame `fk`).
    fn video_sees_video(&self, fq: usize, fk: usize) -> bool {
        match self.causality {
            IntraCausality::Full => true,
            IntraCausality::Frame => fk <= fq,
            IntraCausality::Groups(_) => self.group_of(fk) <= self.group_of(fq),
        }
    }
}

/// Additive attention mask of shape `(1, 1, Lq, Lk)`: `0` where attention is
/// allowed and `NEG` (a large negative finite value) where it is masked.
pub const MASKED: f32 = -1.0e9;

/// Build the additive mask for one denoising step.
///
/// Rows are current tokens `[video | text]`, columns are
/// `[history | current video | current text]`.
pub fn build_attention_mask(spec: &MaskSpec<'_>, dev: &Device) -> Result<Tensor> {
    let lq = spec.query_len();
    let lk = spec.key_len();
    let nt = spec.text_layout.total_tokens;
    crate::ensure!(
        spec.video_tokens == spec.frames * spec.tokens_per_frame,
        "video_tokens {} != frames {} x tokens_per_frame {}",
        spec.video_tokens,
        spec.frames,
        spec.tokens_per_frame
    );
    let mut data = vec![MASKED; lq * lk];
    let allow = |data: &mut [f32], q: usize, k: usize| {
        // allowed == write 0.0
        data[q * lk + k] = 0.0;
    };
    let row_allows = |data: &mut [f32], q: usize, ks: std::ops::Range<usize>| {
        for k in ks {
            data[q * lk + k] = 0.0;
        }
    };

    let nv = spec.video_tokens;
    for q in 0..lq {
        let is_video_query = q < nv;
        let q_frame = if is_video_query { spec.frame_of(q) } else { 0 };
        // 1) history: all current queries see all history.
        row_allows(&mut data, q, 0..spec.history_len);

        if is_video_query {
            // 2) video -> video per causality.
            for k in 0..nv {
                let kf = spec.frame_of(k);
                if spec.video_sees_video(q_frame, kf) {
                    allow(&mut data, q, spec.history_len + k);
                }
            }
            // 3) video -> text: base/separator always; actions within window.
            for (start, end) in spec.text_layout.visible_text_ranges(q_frame) {
                row_allows(
                    &mut data,
                    q,
                    spec.history_len + nv + start..spec.history_len + nv + end,
                );
            }
        } else {
            // text query (base, action or separator)
            let text_off = q - nv;
            let is_action = spec
                .text_layout
                .actions
                .iter()
                .any(|a| text_off >= a.start && text_off < a.end);
            // 4) text -> current video.
            for k in 0..nv {
                let kf = spec.frame_of(k);
                let visible = if is_action {
                    // Action tokens see the video frames of their own window.
                    spec.text_layout.actions.iter().any(|a| {
                        text_off >= a.start
                            && text_off < a.end
                            && kf >= a.frame_first
                            && kf < a.frame_last
                    })
                } else {
                    true
                };
                if visible {
                    allow(&mut data, q, spec.history_len + k);
                }
            }
            // 5) text -> text (all current text sees all current text).
            row_allows(
                &mut data,
                q,
                spec.history_len + nv..spec.history_len + nv + nt,
            );
        }
    }

    // Every query must see at least one key.
    for q in 0..lq {
        let row = &data[q * lk..(q + 1) * lk];
        crate::ensure!(
            row.contains(&0.0),
            "mask row {q} is fully masked (every query needs at least one visible key)"
        );
    }

    Ok(Tensor::from_vec(data, (1usize, 1usize, lq, lk), dev)?)
}

/// Convert a boolean allow-matrix to the additive form (`0` / [`MASKED`]).
pub fn bool_to_additive(allow: &[Vec<bool>], dev: &Device) -> Result<Tensor> {
    let lq = allow.len();
    let lk = allow.first().map(|r| r.len()).unwrap_or(0);
    let mut data = Vec::with_capacity(lq * lk);
    for row in allow {
        for &a in row {
            data.push(if a { 0.0 } else { MASKED });
        }
    }
    Ok(Tensor::from_vec(data, (1usize, 1usize, lq, lk), dev)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::actions::{ActionSpec, TextBlockLayout};
    use crate::tensor_utils::ToVecExt;

    fn layout_two_actions(frames: usize) -> TextBlockLayout {
        let actions = vec![
            ActionSpec {
                start_time: 0.0,
                end_time: 1.0,
                description: "a".into(),
            },
            ActionSpec {
                start_time: 1.0,
                end_time: f64::INFINITY,
                description: "b".into(),
            },
        ];
        TextBlockLayout::new(0, &actions, &[2, 2], frames, 2.0, true)
    }

    fn read_mask(t: &Tensor) -> Vec<Vec<f32>> {
        t.to_dtype(candle_core::DType::F32)
            .unwrap()
            .to_vec4::<f32>()
            .unwrap()[0][0]
            .to_vec()
    }

    #[test]
    fn history_visible_to_everything() {
        let tl = TextBlockLayout::new(0, &[], &[], 2, 2.0, false);
        let spec = MaskSpec {
            history_layout: &[],
            history_len: 5,
            video_tokens: 4,
            frames: 2,
            tokens_per_frame: 2,
            text_layout: &tl,
            causality: IntraCausality::Full,
        };
        let m = build_attention_mask(&spec, &Device::Cpu).unwrap();
        let v = read_mask(&m);
        assert_eq!(v.len(), 4); // 4 video tokens, no text
        for row in &v {
            for k in 0..5 {
                assert_eq!(row[k], 0.0, "history must be visible");
            }
        }
    }

    #[test]
    fn full_mode_is_bidirectional_intra_segment() {
        let tl = layout_two_actions(4);
        let spec = MaskSpec {
            history_layout: &[],
            history_len: 0,
            video_tokens: 8,
            frames: 4,
            tokens_per_frame: 2,
            text_layout: &tl,
            causality: IntraCausality::Full,
        };
        let m = build_attention_mask(&spec, &Device::Cpu).unwrap();
        let v = read_mask(&m);
        // video rows see all video (except action-gated text)
        for q in 0..8 {
            for k in 0..8 {
                assert_eq!(v[q][k], 0.0);
            }
        }
        // frame 0 (tokens 0,1): action a covers [0,1s) -> frames [0,2); token sees text 0..2 + separator
        assert!((v[0][8] - 0.0).abs() < f32::EPSILON);
        assert_eq!(v[0][8 + 1], 0.0);
        assert_eq!(
            v[0][8 + 2],
            MASKED,
            "action b starts at 1s; frame 0 center 0.25s is outside"
        );
        assert_eq!(v[0][8 + 3], MASKED);
        assert_eq!(v[0][8 + 4], 0.0, "separator always visible");
    }

    #[test]
    fn frame_mode_is_causal() {
        let tl = TextBlockLayout::new(0, &[], &[], 4, 2.0, false);
        let spec = MaskSpec {
            history_layout: &[],
            history_len: 0,
            video_tokens: 8,
            frames: 4,
            tokens_per_frame: 2,
            text_layout: &tl,
            causality: IntraCausality::Frame,
        };
        let m = build_attention_mask(&spec, &Device::Cpu).unwrap();
        let v = read_mask(&m);
        // frame 2 (tokens 4,5) sees frames 0,1,2
        for k in 0..6 {
            assert_eq!(v[4][k], 0.0);
        }
        assert_eq!(v[4][6], MASKED);
        assert_eq!(v[4][7], MASKED);
    }

    #[test]
    fn group_mode_parity() {
        let tl = TextBlockLayout::new(0, &[], &[], 4, 2.0, false);
        let spec = MaskSpec {
            history_layout: &[],
            history_len: 0,
            video_tokens: 8,
            frames: 4,
            tokens_per_frame: 2,
            text_layout: &tl,
            causality: IntraCausality::Groups(2),
        };
        let m = build_attention_mask(&spec, &Device::Cpu).unwrap();
        let v = read_mask(&m);
        // group 0 (frames 0,1 -> tokens 0..4) sees only group 0
        for q in 0..4 {
            for k in 0..4 {
                assert_eq!(v[q][k], 0.0);
            }
            assert_eq!(v[q][4], MASKED);
        }
        // group 1 sees both groups
        for q in 4..8 {
            for k in 0..8 {
                assert_eq!(v[q][k], 0.0);
            }
        }
    }

    #[test]
    fn action_query_sees_only_its_window() {
        let tl = layout_two_actions(4);
        let spec = MaskSpec {
            history_layout: &[],
            history_len: 0,
            video_tokens: 8,
            frames: 4,
            tokens_per_frame: 2,
            text_layout: &tl,
            causality: IntraCausality::Full,
        };
        let m = build_attention_mask(&spec, &Device::Cpu).unwrap();
        let v = read_mask(&m);
        // action b (text token 2) covers frames [2,4) -> video tokens 4..8
        assert_eq!(v[8 + 2][4], 0.0);
        assert_eq!(v[8 + 2][7], 0.0);
        assert_eq!(v[8 + 2][0], MASKED);
        assert_eq!(v[8 + 2][3], MASKED);
        // action tokens still see the whole text block
        assert_eq!(v[8 + 2][8 + 4], 0.0);
    }

    #[test]
    fn every_row_has_a_visible_key() {
        // An action with an empty window must not produce a dead row: it still
        // sees text + history.
        let actions = vec![ActionSpec {
            start_time: 10.0,
            end_time: 20.0,
            description: "x".into(),
        }];
        let tl = TextBlockLayout::new(0, &actions, &[2], 2, 2.0, true);
        let spec = MaskSpec {
            history_layout: &[],
            history_len: 3,
            video_tokens: 4,
            frames: 2,
            tokens_per_frame: 2,
            text_layout: &tl,
            causality: IntraCausality::Full,
        };
        let m = build_attention_mask(&spec, &Device::Cpu).unwrap();
        let v = read_mask(&m);
        for (q, row) in v.iter().enumerate() {
            assert!(
                row.iter().any(|&x| x.abs() < f32::EPSILON),
                "row {q} fully masked"
            );
        }
    }
}
