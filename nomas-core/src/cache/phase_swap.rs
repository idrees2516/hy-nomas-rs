//! Phase-swapped CPU offload — port of `PhaseSwappedKVCache`.
//!
//! Between denoising phases (e.g. while the PixelUMM decoder runs), the
//! working set is *parked* on the host and the device-side copy is dropped;
//! the next denoising phase unparks it. On a CPU build this is a bookkeeping
//! move; with the `cuda` feature enabled it performs real device→host→device
//! transfers, exactly like the upstream pipeline that offloads the DiT, text
//! encoder and VAE in turns.

use candle_core::{Device, Tensor};

use crate::config::{CacheStorage, HistorySampling};
use crate::Result;

use super::history::HistoryCache;
use super::{CacheMetrics, LifetimeKvCache, SegmentKv, SegmentLayout};

struct Parked {
    k: Tensor,
    v: Tensor,
    layout: Vec<SegmentLayout>,
    valid_len: usize,
    bytes: usize,
}

/// Wrapper that parks/unparks the working set of an inner [`HistoryCache`].
pub struct PhaseSwappedCache {
    inner: HistoryCache,
    parked: Option<Parked>,
}

impl PhaseSwappedCache {
    /// Wrap an inner cache.
    pub fn new(inner: HistoryCache) -> Self {
        Self {
            inner,
            parked: None,
        }
    }

    /// Park the working set on the host, releasing device memory.
    pub fn park(&mut self) -> Result<()> {
        if self.parked.is_some() {
            return Ok(());
        }
        let (k, v) = self.inner.working_set()?;
        let layout = self.inner.working_layout();
        let valid_len = self.inner.valid_len();
        let host = Device::Cpu;
        let k = k.to_device(&host)?.contiguous()?;
        let v = v.to_device(&host)?.contiguous()?;
        let bytes = {
            let per = |t: &Tensor| t.dims().iter().product::<usize>() * t.dtype().size_in_bytes();
            per(&k) + per(&v)
        };
        self.inner.release_working_set();
        self.inner.note_d2h(bytes);
        self.parked = Some(Parked {
            k,
            v,
            layout,
            valid_len,
            bytes,
        });
        Ok(())
    }

    /// Unpark the working set back onto the cache device.
    pub fn unpark(&mut self) -> Result<()> {
        let parked = match self.parked.take() {
            Some(p) => p,
            None => return Ok(()),
        };
        let dev = self.inner.device().clone();
        let k = parked.k.to_device(&dev)?;
        let v = parked.v.to_device(&dev)?;
        self.inner
            .restore_working_set(k, v, parked.layout, parked.valid_len);
        self.inner.note_h2d(parked.bytes);
        Ok(())
    }

    /// Whether the working set is currently parked.
    pub fn is_parked(&self) -> bool {
        self.parked.is_some()
    }

    /// Bytes moved by the last park.
    pub fn parked_bytes(&self) -> usize {
        self.parked.as_ref().map(|p| p.bytes).unwrap_or(0)
    }
}

impl LifetimeKvCache for PhaseSwappedCache {
    fn storage_format(&self) -> CacheStorage {
        self.inner.storage_format()
    }

    fn sampling(&self) -> HistorySampling {
        self.inner.sampling()
    }

    fn commit_segment(&mut self, seg: SegmentKv) -> Result<()> {
        self.inner.commit_segment(seg)
    }

    fn working_set(&self) -> Result<(Tensor, Tensor)> {
        if self.parked.is_some() {
            return Err(crate::Error::CacheState(
                "working set is parked; call unpark() before attention".into(),
            ));
        }
        self.inner.working_set()
    }

    fn working_layout(&self) -> Vec<SegmentLayout> {
        match &self.parked {
            Some(p) => p.layout.clone(),
            None => self.inner.working_layout(),
        }
    }

    fn valid_len(&self) -> usize {
        match &self.parked {
            Some(p) => p.valid_len,
            None => self.inner.valid_len(),
        }
    }

    fn capacity(&self) -> usize {
        self.inner.capacity()
    }

    fn n_segments(&self) -> usize {
        self.inner.n_segments()
    }

    fn total_frames(&self) -> usize {
        self.inner.total_frames()
    }

    fn tokens_per_frame(&self) -> usize {
        self.inner.tokens_per_frame()
    }

    fn rollback_last_segment(&mut self) -> Result<()> {
        self.inner.rollback_last_segment()
    }

    fn metrics(&self) -> CacheMetrics {
        self.inner.metrics()
    }

    fn park(&mut self) -> Result<()> {
        PhaseSwappedCache::park(self)
    }

    fn unpark(&mut self) -> Result<()> {
        PhaseSwappedCache::unpark(self)
    }
}
