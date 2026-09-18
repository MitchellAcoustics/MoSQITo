//! ECMA-418-2 (2nd Ed, 2022) §5.1.4.1 Eq. 9: the 53 auditory filter bank
//! centre frequencies.

const AF_F0: f64 = 81.9289;
const C: f64 = 0.1618;

pub fn auditory_filters_centre_freq() -> [f64; 53] {
    std::array::from_fn(|band_number| {
        let z = (band_number + 1) as f64 * 0.5;
        (AF_F0 / C) * (C * z).sinh()
    })
}
