use std::ops::Sub;

/// 代表一个坐标(height, width)索引，不负责边界检查。
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
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
    pub fn to_tuple(self) -> (usize, usize) {
        (self.h, self.w)
    }
}

impl From<(usize, usize)> for Pos {
    #[inline]
    fn from(pos: (usize, usize)) -> Self {
        Pos::new(pos.0, pos.1)
    }
}

impl Sub<Pos> for Pos {
    type Output = Vector2d;

    fn sub(self, rhs: Pos) -> Self::Output {
        let x = self.h as i64 - rhs.h as i64;
        let y = self.w as i64 - rhs.w as i64;
        Vector2d(x as i32, y as i32)
    }
}

#[derive(Copy, Clone, Eq, PartialEq)]
pub struct Vector2d(i32, i32);

impl Vector2d {
    #[inline]
    pub fn new(x: i32, y: i32) -> Self {
        Self(x, y)
    }

    #[inline]
    pub fn dot(self, other: Vector2d) -> i32 {
        self.0 * other.0 + self.1 * other.1
    }

    #[inline]
    pub fn is_not_axis_direction(self) -> bool {
        self.0 != 0 && self.1 != 0
    }

    #[inline]
    pub fn is_axis_direction(self) -> bool {
        !self.is_not_axis_direction()
    }
}
