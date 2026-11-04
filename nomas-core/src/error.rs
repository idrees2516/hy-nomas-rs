//! Error type shared across the crate.

use thiserror::Error;

/// Errors produced by `nomas-core`.
#[derive(Error, Debug)]
pub enum Error {
    /// Propagated tensor/backend errors from `candle`.
    #[error("tensor error: {0}")]
    Candle(#[from] candle_core::Error),

    /// Filesystem errors while loading checkpoints.
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    /// Safetensors deserialization errors.
    #[error("safetensors error: {0}")]
    Safetensors(#[from] safetensors::SafeTensorError),

    /// A checkpoint key could not be mapped onto the model.
    #[error("weight key error: {0}")]
    WeightKey(String),

    /// Invalid or inconsistent configuration.
    #[error("invalid config: {0}")]
    InvalidConfig(String),

    /// The lifelong KV cache was used in a state that does not support the
    /// requested operation (e.g. rolling back a `recent` SWA cache, or
    /// committing two segments with the same index).
    #[error("cache state error: {0}")]
    CacheState(String),

    /// Geometry (frame/token counts, patch sizes, capacities) disagreement.
    #[error("geometry error: {0}")]
    Geometry(String),

    /// The requested feature maps to a GPU-only kernel path upstream and has no
    /// portable implementation here.
    #[error("unsupported: {0}")]
    Unsupported(&'static str),

    /// Free-form error message.
    #[error("{0}")]
    Msg(String),
}

impl Error {
    /// Build a free-form [`Error::Msg`].
    pub fn msg<S: Into<String>>(s: S) -> Self {
        Error::Msg(s.into())
    }
}

/// Convenience alias used throughout the crate.
pub type Result<T> = std::result::Result<T, Error>;

#[macro_export]
macro_rules! bail {
    ($($arg:tt)*) => {
        return Err($crate::Error::Msg(format!($($arg)*)))
    };
}

#[macro_export]
macro_rules! ensure {
    ($cond:expr, $($arg:tt)*) => {
        #[allow(clippy::neg_cmp_op_on_partial_ord)]
        if !($cond) {
            return Err($crate::Error::Msg(format!($($arg)*)));
        }
    };
}
