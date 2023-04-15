use image::Rgb;
use label::prelude::TRAINING_SET_LEN;
use std::collections::BTreeSet;

pub fn color_valid_rgb_hex(s: &str) -> Result<Rgb<u8>, &'static str> {
    const ERR: &str = "十六进制RGB颜色格式错误";
    fn ck(s: &str) -> Option<Rgb<u8>> {
        let r = u8::from_str_radix(&s[0..=1], 16).ok()?;
        let g = u8::from_str_radix(&s[2..=3], 16).ok()?;
        let b = u8::from_str_radix(&s[4..=5], 16).ok()?;
        Some(Rgb::from([r, g, b]))
    }
    match s.len() {
        6 => ck(s).ok_or(ERR),
        7 if s.as_bytes()[0] == b'#' => ck(&s[1..]).ok_or(ERR),
        _ => Err(ERR),
    }
}

pub fn ranges_to_integers(s: &str) -> Result<BTreeSet<usize>, &'static str> {
    // 从字符串中提取整数集合。该集合应该是(0..=130)的子集；若为空集，则视为全选。
    const ERR: &str = "整数范围格式错误或越界";
    let mut set = BTreeSet::<usize>::new();
    for ranges in s.split(',') {
        let mut d_iter = ranges.split('-');
        let d1: usize = d_iter.next().ok_or(ERR)?.parse().map_err(|_| ERR)?;
        if d1 >= TRAINING_SET_LEN {
            return Err(ERR);
        }
        let d2 = d_iter.next();
        if let Some(d2) = d2 {
            let d2: usize = d2.parse().map_err(|_| ERR)?;
            if !(d1..TRAINING_SET_LEN).contains(&d2) {
                return Err(ERR);
            }
            set.extend(d1..=d2);
            if d_iter.next().is_some() {
                return Err(ERR);
            }
        } else {
            set.insert(d1);
        }
    }
    Ok(set)
}

pub mod rgb {
    use image::Rgb;

    #[inline]
    pub fn black() -> Rgb<u8> {
        Rgb::from([0x00, 0x00, 0x00])
    }

    #[inline]
    pub fn white() -> Rgb<u8> {
        Rgb::from([0xFF, 0xFF, 0xFF])
    }

    #[inline]
    pub fn yellow() -> Rgb<u8> {
        Rgb::from([0xFF, 0xFF, 0x00])
    }

    pub fn purple() -> Rgb<u8> {
        // Rgb::from([0x8B, 0x75, 0x00])
        Rgb::from([0xFF, 0x00, 0xFF])
    }
}
