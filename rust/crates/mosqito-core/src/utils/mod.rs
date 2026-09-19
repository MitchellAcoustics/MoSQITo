//! Small utilities shared across metrics: signal segmentation, level/frequency
//! conversions, and the threshold-in-quiet table.

pub mod conversion;
pub mod ltq;
pub mod time_segmentation;

pub use conversion::{amp2db, bark2freq, db2amp, freq2bark, spectrum2dba};
pub use ltq::{ltq, LtqReference};
pub use time_segmentation::time_segmentation;
