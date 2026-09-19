//! Small utilities shared across metrics: signal segmentation and level
//! conversions.

pub mod conversion;
pub mod time_segmentation;

pub use conversion::amp2db;
pub use time_segmentation::time_segmentation;
