use super::improc::LiverSlice;
use super::pos::Pos;

pub struct PosIter {
    cur_h: usize,
    cur_w: usize,
    h: usize,
    w: usize,
}

impl PosIter {
    #[inline]
    pub fn new(h_len: usize, w_len: usize) -> Self {
        Self {
            cur_h: 0,
            cur_w: 0,
            h: h_len,
            w: w_len,
        }
    }
}

impl Iterator for PosIter {
    type Item = Pos;

    fn next(&mut self) -> Option<Self::Item> {
        if self.cur_h == self.h {
            return None;
        }
        let ret_pos = Pos::new(self.cur_h, self.cur_w);
        if self.cur_w + 1 == self.w {
            self.cur_w = 0;
            self.cur_h += 1;
        } else {
            self.cur_w += 1;
        }
        Some(ret_pos)
    }
}

impl From<&LiverSlice> for PosIter {
    #[inline]
    fn from(s: &LiverSlice) -> Self {
        PosIter::new(s.h_len, s.w_len)
    }
}

pub struct LiverSliceIter<'a> {
    orig: &'a LiverSlice,
    cursor: usize,
    len: usize,
}

impl<'a> LiverSliceIter<'a> {
    #[inline]
    pub fn new(orig: &'a LiverSlice) -> Self {
        Self {
            orig,
            cursor: 0,
            len: orig.w_len * orig.h_len,
        }
    }
}

impl<'a> Iterator for LiverSliceIter<'a> {
    type Item = &'a u8;

    fn next(&mut self) -> Option<Self::Item> {
        if self.cursor != self.len {
            let v = unsafe { &*self.orig.data.add(self.cursor) };
            self.cursor += 1;
            Some(v)
        } else {
            None
        }
    }
}

pub struct LiverSliceIterMut<'a> {
    orig: &'a mut LiverSlice,
    cursor: usize,
    len: usize,
}

impl<'a> LiverSliceIterMut<'a> {
    #[inline]
    pub fn new(orig: &'a mut LiverSlice) -> Self {
        let len = orig.h_len * orig.w_len; // bypass borrow ck
        Self {
            orig,
            cursor: 0,
            len,
        }
    }
}

impl<'a> Iterator for LiverSliceIterMut<'a> {
    type Item = &'a mut u8;

    fn next(&mut self) -> Option<Self::Item> {
        if self.cursor != self.len {
            let v = unsafe { &mut *self.orig.data.add(self.cursor) };
            self.cursor += 1;
            Some(v)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::PosIter;

    fn test_pos_iter_with(h: usize, w: usize) {
        let mut it = PosIter::new(h, w);
        match (h, w) {
            (0, 0) => assert_eq!(it.next(), None),
            (h, w) => {
                for a in 0..h {
                    for b in 0..w {
                        assert_eq!(it.next(), Some((a, b).into()));
                    }
                }
                assert_eq!(it.next(), None);
            }
        }
        assert_eq!(it.next(), None);
    }

    #[test]
    fn test_pos_iter_zero() {
        test_pos_iter_with(0, 0);
    }

    #[test]
    fn test_pos_iter_lits() {
        test_pos_iter_with(512, 512);
    }
}
