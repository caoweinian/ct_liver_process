use super::improc::LiverCtSlice;
use super::pos::Pos;
use super::{LIVER_HEIGHT, LIVER_SIZE};

pub struct LiverCtSliceIter<'a> {
    orig: &'a LiverCtSlice,
    cursor: usize,
}

impl<'a> LiverCtSliceIter<'a> {
    #[inline]
    pub fn new(orig: &'a LiverCtSlice) -> Self {
        Self { orig, cursor: 0 }
    }
}

impl<'a> Iterator for LiverCtSliceIter<'a> {
    type Item = &'a u8;

    fn next(&mut self) -> Option<Self::Item> {
        if self.cursor != LIVER_SIZE {
            let target = unsafe { &*self.orig.data.add(self.cursor) };
            self.cursor += 1;
            Some(target)
        } else {
            None
        }
    }
}

pub struct LiverCtSliceIterMut<'a> {
    orig: &'a mut LiverCtSlice,
    cursor: usize,
}

impl<'a> LiverCtSliceIterMut<'a> {
    #[inline]
    pub fn new(orig: &'a mut LiverCtSlice) -> Self {
        Self { orig, cursor: 0 }
    }
}

impl<'a> Iterator for LiverCtSliceIterMut<'a> {
    type Item = &'a mut u8;

    fn next(&mut self) -> Option<Self::Item> {
        if self.cursor != LIVER_SIZE {
            let target = unsafe { &mut *self.orig.data.add(self.cursor) };
            self.cursor += 1;
            Some(target)
        } else {
            None
        }
    }
}

pub struct PosIter {
    p: Pos,
}

impl PosIter {
    #[inline]
    pub fn new() -> Self {
        Self { p: Pos::new(0, 0) }
    }
}

impl Iterator for PosIter {
    type Item = Pos;

    fn next(&mut self) -> Option<Self::Item> {
        if self.p.h == LIVER_HEIGHT {
            return None;
        }
        let ret_pos = self.p;
        if self.p.is_row_last() {
            self.p.w = 0;
            self.p.h += 1;
        } else {
            self.p.w += 1;
        }
        Some(ret_pos)
    }
}

#[cfg(test)]
mod tests {
    use super::PosIter;
    use crate::{LIVER_HEIGHT, LIVER_WIDTH};

    #[test]
    fn test_pos_iter() {
        let mut it = PosIter::new();
        for h in 0..LIVER_HEIGHT {
            for w in 0..LIVER_WIDTH {
                assert_eq!(it.next(), Some((h, w).into()));
            }
        }
        assert!(it.next().is_none());
    }

    #[test]
    fn test_wrapping() {
        let i = 0_usize;
        assert_eq!(i.wrapping_sub(1), usize::MAX);
    }
}
