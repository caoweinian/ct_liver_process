use super::{LIVER_HEIGHT, LIVER_WIDTH};
use json::JsonValue;
use std::cmp::min;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct Pos {
    pub h: usize,
    pub w: usize,
}

impl Pos {
    #[inline]
    pub fn new(h: usize, w: usize) -> Self {
        Self { h, w }
    }

    #[inline]
    pub fn is_row_last(&self) -> bool {
        self.w + 1 == LIVER_WIDTH
    }

    #[inline]
    pub fn is_image_boundary(&self) -> bool {
        self.w == 0 || self.h == 0 || self.w + 1 == LIVER_WIDTH || self.h + 1 == LIVER_HEIGHT
    }

    #[inline]
    pub fn to_tuple(self) -> (usize, usize) {
        (self.h, self.w)
    }

    #[inline]
    pub fn in_bounds(&self) -> bool {
        self.h < LIVER_HEIGHT && self.w < LIVER_WIDTH
    }

    // (h lb, h ub), (w lb, w, ub). closed
    pub fn window_3x3(self) -> ((usize, usize), (usize, usize)) {
        let h_lb = self.h.saturating_sub(1);
        let h_ub = min(LIVER_HEIGHT, self.h.saturating_add(2));
        let w_lb = self.w.saturating_sub(1);
        let w_ub = min(LIVER_WIDTH, self.w.saturating_add(2));
        ((h_lb, h_ub), (w_lb, w_ub))
    }
}

impl From<(usize, usize)> for Pos {
    #[inline]
    fn from(pos: (usize, usize)) -> Self {
        Pos::new(pos.0, pos.1)
    }
}

impl From<Pos> for JsonValue {
    #[inline]
    fn from(p: Pos) -> Self {
        json::array![p.h, p.w]
    }
}

#[cfg(test)]
mod tests {
    use crate::prep_ffi::pos::Pos;

    #[test]
    fn test_window_3x3() {
        let p = Pos::new(4, 5);
        let ((a, b), (c, d)) = p.window_3x3();
        assert_eq!(&[a, b, c, d], &[3, 6, 4, 7]);
    }
}

// impl Add<Pos> for Pos {
//     type Output = Self;
//
//     #[inline]
//     fn add(self, rhs: Pos) -> Self::Output {
//         (self.h + rhs.h, self.w + rhs.w).into()
//     }
// }
