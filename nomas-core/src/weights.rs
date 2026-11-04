//! Checkpoint loading — safetensors + key diagnostics.
//!
//! `nomas-core` uses a stable, documented key naming convention (see
//! [`expected_keys`]). Real Hy-Nomas checkpoints need a one-time key remap;
//! [`CheckpointReport`] surfaces exactly which keys matched so integrators can
//! write the adapter with full visibility.

use std::collections::{HashMap, HashSet};

use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;

use crate::config::{Activation, NomasConfig};
use crate::error::{Error, Result};
use crate::model::NomasDiT;
use crate::modules::Weights;

/// Diagnostics of a checkpoint load.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CheckpointReport {
    /// Checkpoint keys consumed by the model.
    pub matched: Vec<String>,
    /// Model keys absent from the checkpoint.
    pub missing: Vec<String>,
    /// Checkpoint keys the model does not consume.
    pub unexpected: Vec<String>,
}

impl CheckpointReport {
    /// Whether the checkpoint fully covered the model.
    pub fn is_complete(&self) -> bool {
        self.missing.is_empty()
    }
}

/// Every parameter key the model expects for `cfg`, in a stable order.
pub fn expected_keys(cfg: &NomasConfig) -> Vec<String> {
    let mut keys = Vec::new();
    fn lin(keys: &mut Vec<String>, prefix: &str, bias: bool) {
        keys.push(format!("{prefix}.weight"));
        if bias {
            keys.push(format!("{prefix}.bias"));
        }
    }
    fn ffn(keys: &mut Vec<String>, prefix: &str, act: Activation) {
        match act {
            Activation::Silu => {
                lin(keys, &format!("{prefix}.gate"), true);
                lin(keys, &format!("{prefix}.up"), true);
                lin(keys, &format!("{prefix}.down"), true);
            }
            Activation::GeluErf => {
                lin(keys, &format!("{prefix}.fc1"), true);
                lin(keys, &format!("{prefix}.fc2"), true);
            }
        }
    }
    lin(&mut keys, "patch_embed.proj", true);
    lin(&mut keys, "text_proj", true);
    keys.push("text_separator".into());
    lin(&mut keys, "time_embed.mlp1", true);
    lin(&mut keys, "time_embed.mlp2", true);
    lin(&mut keys, "cam_embed.mlp1", true);
    lin(&mut keys, "cam_embed.mlp2", true);
    for i in 0..cfg.n_layers {
        let p = format!("blocks.{i}");
        keys.push(format!("{p}.norm1.weight"));
        keys.push(format!("{p}.norm2.weight"));
        lin(&mut keys, &format!("{p}.attn.q"), true);
        lin(&mut keys, &format!("{p}.attn.k"), true);
        lin(&mut keys, &format!("{p}.attn.v"), true);
        lin(&mut keys, &format!("{p}.attn.out"), true);
        if cfg.qk_norm {
            keys.push(format!("{p}.attn.q_norm.weight"));
            keys.push(format!("{p}.attn.k_norm.weight"));
        }
        lin(&mut keys, &format!("{p}.modulation"), true);
        match &cfg.moe {
            Some(moe) => {
                lin(&mut keys, &format!("{p}.moe.router"), false);
                for e in 0..moe.experts {
                    ffn(
                        &mut keys,
                        &format!("{p}.moe.experts.{e}"),
                        cfg.mlp_activation,
                    );
                }
                if moe.shared_dense > 0.0 {
                    ffn(&mut keys, &format!("{p}.moe.dense"), cfg.mlp_activation);
                }
            }
            None => ffn(&mut keys, &format!("{p}.mlp"), cfg.mlp_activation),
        }
    }
    keys.push("final.norm.weight".into());
    lin(&mut keys, "final.ada", true);
    lin(&mut keys, "final.linear", true);
    keys
}

/// Load a `NomasDiT` from one or more safetensors files.
///
/// Fails with a [`Error::WeightKey`] listing the first missing key; a full
/// [`CheckpointReport`] is returned on success.
pub fn load_dit(
    paths: &[std::path::PathBuf],
    cfg: &NomasConfig,
    dtype: candle_core::DType,
    dev: &Device,
) -> Result<(NomasDiT, CheckpointReport)> {
    let mut tensors: HashMap<String, Tensor> = HashMap::new();
    for p in paths {
        let loaded = candle_core::safetensors::load(p, dev)
            .map_err(|e| Error::Msg(format!("loading {}: {e}", p.display())))?;
        tensors.extend(loaded);
    }
    let expected: HashSet<String> = expected_keys(cfg).into_iter().collect();
    let present: HashSet<String> = tensors.keys().cloned().collect();
    let mut missing: Vec<String> = expected.difference(&present).cloned().collect();
    if !missing.is_empty() {
        missing.sort();
        return Err(Error::WeightKey(format!(
            "checkpoint is missing {} model keys, first few: {:?}",
            missing.len(),
            &missing[..missing.len().min(5)]
        )));
    }
    let unexpected: Vec<String> = present.difference(&expected).cloned().collect();
    // Cast all tensors to the target dtype.
    let tensors = tensors
        .into_iter()
        .map(|(k, v)| v.to_dtype(dtype).map(|v| (k, v)))
        .collect::<candle_core::Result<HashMap<_, _>>>()?;
    let matched: Vec<String> = expected.intersection(&present).cloned().collect();
    let vb = VarBuilder::from_tensors(tensors, dtype, dev);
    let dit = NomasDiT::from_source(&Weights::loaded(vb), cfg)?;
    let mut report = CheckpointReport {
        matched,
        missing: Vec::new(),
        unexpected,
    };
    report.matched.sort();
    report.unexpected.sort();
    Ok((dit, report))
}

/// Build a deterministically initialized model (tests, examples, demos).
pub fn random_dit(
    cfg: &NomasConfig,
    seed: u64,
    dtype: candle_core::DType,
    dev: &Device,
) -> Result<NomasDiT> {
    NomasDiT::from_source(&Weights::random(seed, dtype, dev), cfg)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expected_keys_cover_every_block() {
        let mut cfg = NomasConfig::tiny();
        cfg.n_layers = 3;
        let keys = expected_keys(&cfg);
        for i in 0..3 {
            assert!(keys.contains(&format!("blocks.{i}.attn.q.weight")));
            assert!(keys.contains(&format!("blocks.{i}.mlp.down.bias")));
            assert!(keys.contains(&format!("blocks.{i}.modulation.weight")));
        }
        assert!(keys.contains(&"patch_embed.proj.weight".to_string()));
        assert!(keys.contains(&"final.linear.weight".to_string()));
    }

    #[test]
    fn moe_keys() {
        let mut cfg = NomasConfig::tiny();
        cfg.moe = Some(crate::config::MoeConfig {
            experts: 2,
            top_k: 1,
            shared_dense: 0.0,
            seed: 1,
        });
        let keys = expected_keys(&cfg);
        assert!(keys.contains(&"blocks.0.moe.router.weight".to_string()));
        assert!(keys.contains(&"blocks.0.moe.experts.1.gate.weight".to_string()));
        assert!(!keys.contains(&"blocks.0.moe.experts.2.gate.weight".to_string()));
    }
}
