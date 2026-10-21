//! Peak Metal memory, sampled at the points where a forward pass holds the most.
//!
//! Candle pools Metal buffers and releases unused ones only when a later allocation misses the
//! pool, so a reading taken after a pass can miss its largest temporaries (the attention score
//! buffers). The encoder calls `sample` right after allocating them and after every block.

use std::sync::atomic::{AtomicUsize, Ordering};

use candle_core::Device;

static PEAK_BYTES: AtomicUsize = AtomicUsize::new(0);

/// Bytes the Metal device has allocated for this process (all live buffers, including
/// Candle's pool), or `None` off Metal.
pub fn allocated_bytes(device: &Device) -> Option<usize> {
    match device {
        Device::Metal(metal) => Some(metal.current_allocated_size()),
        _ => None,
    }
}

/// Folds the current allocation into the running peak.
pub fn sample(device: &Device) {
    if let Some(bytes) = allocated_bytes(device) {
        PEAK_BYTES.fetch_max(bytes, Ordering::Relaxed);
    }
}

/// The peak since the last reset, then starts a new window at the current allocation.
pub fn take_peak(device: &Device) -> Option<usize> {
    let now = allocated_bytes(device)?;
    Some(PEAK_BYTES.swap(now, Ordering::Relaxed).max(now))
}
