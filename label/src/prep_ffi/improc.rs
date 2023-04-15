use json::JsonValue;
use ndarray::Array2;
use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::mpsc::channel;
use threadpool::ThreadPool;

use super::iter::{LiverCtSliceIter, LiverCtSliceIterMut, PosIter};
use super::pos::Pos;

pub const LIVER_HEIGHT: usize = 512;
pub const LIVER_WIDTH: usize = 512;
pub const LIVER_SIZE: usize = LIVER_HEIGHT * LIVER_WIDTH;

macro_rules! slice_at {
    ($data: expr, $index: expr) => {
        LiverCtSlice::new(unsafe { $data.as_mut_ptr().add(LIVER_SIZE * $index) })
    };
}

pub fn unify(data: &mut [u8]) {
    let z_len = data.len() / LIVER_SIZE;
    let logical_cpus = num_cpus::get();
    if logical_cpus == 1 {
        for index in 0..z_len {
            let mut s = slice_at!(data, index);
            s.unify_image();
        }
    } else {
        let pool = ThreadPool::new(logical_cpus);
        for index in 0..z_len {
            let mut s = slice_at!(data, index);
            pool.execute(move || s.unify_image());
        }
        pool.join();
    }
}

pub fn make_labels(data: &mut [u8], base_dir: PathBuf) {
    let z_len = data.len() / LIVER_SIZE;

    let labels: Vec<_> = match num_cpus::get() {
        1 => (0..z_len)
            .map(|z| slice_at!(data, z).make_label(z))
            .collect(),
        cpus => {
            let pool = ThreadPool::new(cpus);
            let (tx, rx) = channel();
            for index in 0..z_len {
                let tx = tx.clone();
                let mut s = slice_at!(data, index);
                pool.execute(move || tx.send(s.make_label(index)).expect("send error"));
            }
            rx.iter().take(z_len).collect()
        }
    };
    write_labels_to_disk(base_dir, labels);
}

fn write_labels_to_disk(mut base_dir: PathBuf, labels: Vec<(usize, JsonValue)>) {
    for (seq, j) in labels {
        base_dir.push(format!("{seq}.json"));
        std::fs::write(base_dir.as_path(), j.dump()).expect("write error");
        base_dir.pop();
    }
}

#[repr(u8)]
#[derive(Copy, Clone, Eq, PartialEq)]
enum LiTSLabel {
    /// 背景（黑色）
    Background = 0b_0000_0000,
    /// 边界（灰色）；或可以说是肝脏外部或肝脏边界，属于肝脏的一部分。
    Boundary = 0b_1000_0000,
    /// 肿瘤（灰白色）
    Tumor = 0b_1100_0000,
    /// 肝脏内部（白色）
    Liver = 0b_1111_1111,
}

impl LiTSLabel {
    #[inline]
    pub const fn to_u8(self) -> u8 {
        self as u8
    }
}

impl TryFrom<u8> for LiTSLabel {
    type Error = ();

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0b_0000_0000 => Ok(Self::Background),
            0b_1111_1111 => Ok(Self::Liver),
            _ => Err(()), // other cases are not needed
        }
    }
}

#[repr(transparent)]
pub struct LiverCtSlice {
    pub(crate) data: *mut u8,
}

unsafe impl Send for LiverCtSlice {}

type Area = Vec<Pos>;
type Areas = Vec<Area>;

impl LiverCtSlice {
    #[inline]
    pub fn new(pointer: *mut u8) -> Self {
        Self { data: pointer }
    }

    #[allow(dead_code)]
    fn count(&self, label: LiTSLabel) -> usize {
        self.iter().copied().filter(|v| *v == label.to_u8()).count()
    }

    #[inline]
    unsafe fn is_boundary(&self, p: Pos) -> bool {
        *self.uget(p) == LiTSLabel::Boundary.to_u8()
    }

    #[allow(dead_code)]
    #[inline]
    unsafe fn is_equal(&self, p: Pos, label: LiTSLabel) -> bool {
        *self.uget(p) == label.to_u8()
    }

    #[inline]
    unsafe fn uget(&self, p: Pos) -> &u8 {
        &*(self.data.add(p.h * LIVER_WIDTH + p.w))
    }

    #[inline]
    unsafe fn uget_mut(&mut self, p: Pos) -> &mut u8 {
        &mut *(self.data.add(p.h * LIVER_WIDTH + p.w))
    }

    #[inline]
    fn get(&self, p: Pos) -> Option<&u8> {
        p.in_bounds().then(|| unsafe { self.uget(p) })
    }

    #[allow(dead_code)]
    #[inline]
    fn get_mut(&mut self, p: Pos) -> Option<&mut u8> {
        p.in_bounds().then(|| unsafe { self.uget_mut(p) })
    }

    #[inline]
    fn iter(&self) -> LiverCtSliceIter {
        LiverCtSliceIter::new(self)
    }

    #[inline]
    fn iter_mut(&mut self) -> LiverCtSliceIterMut {
        LiverCtSliceIterMut::new(self)
    }

    fn all_inside(&self, positions: &[Pos]) -> bool {
        positions.iter().all(|p| !p.is_image_boundary())
    }

    fn fill_tumor_with_liver(&mut self) {
        self.iter_mut()
            .filter(|p| **p == LiTSLabel::Tumor.to_u8())
            .for_each(|p| *p = LiTSLabel::Liver.to_u8());
    }

    fn area_group(&self, pred: fn(u8) -> bool, map: &mut Array2<u8>) -> Areas {
        let mut ans: Areas = Vec::with_capacity(1);
        let mut bfs_q: VecDeque<Pos> = VecDeque::with_capacity(4);

        // map中元素值为0，代表未曾遍历过；1代表遍历过。
        for pos in PosIter::new() {
            unsafe {
                match *map.uget(pos.to_tuple()) {
                    0 => {
                        if !pred(*self.uget(pos)) {
                            continue;
                        }
                        bfs_q.push_back(pos);
                        let mut one_area: Area = Vec::with_capacity(1);
                        while !bfs_q.is_empty() {
                            let cur_pos = bfs_q.pop_front().unwrap();
                            let (cur_h, cur_w) = cur_pos.to_tuple();
                            let p = map.uget_mut((cur_h, cur_w));
                            if *p == 1 {
                                continue;
                            }
                            *p = 1;
                            one_area.push(cur_pos);
                            if cur_h > 0
                                && pred(*self.uget((cur_h - 1, cur_w).into()))
                                && *map.uget((cur_h - 1, cur_w)) == 0
                            {
                                bfs_q.push_back((cur_h - 1, cur_w).into());
                            }
                            if cur_h + 1 < LIVER_HEIGHT
                                && pred(*self.uget((cur_h + 1, cur_w).into()))
                                && *map.uget((cur_h + 1, cur_w)) == 0
                            {
                                bfs_q.push_back((cur_h + 1, cur_w).into());
                            }
                            if cur_w > 0
                                && pred(*self.uget((cur_h, cur_w - 1).into()))
                                && *map.uget((cur_h, cur_w - 1)) == 0
                            {
                                bfs_q.push_back((cur_h, cur_w - 1).into());
                            }
                            if cur_w + 1 < LIVER_WIDTH
                                && pred(*self.uget((cur_h, cur_w + 1).into()))
                                && *map.uget((cur_h, cur_w + 1)) == 0
                            {
                                bfs_q.push_back((cur_h, cur_w + 1).into());
                            }
                        }
                        ans.push(one_area);
                    }
                    1 => continue,
                    _ => unreachable!(),
                }
            }
        }
        ans
    }

    #[inline]
    fn liver_area_group(&self, map: &mut Array2<u8>) -> Areas {
        self.area_group(|u| u == LiTSLabel::Liver.to_u8(), map)
    }

    #[inline]
    fn background_area_group(&self, map: &mut Array2<u8>) -> Areas {
        self.area_group(|u| u == LiTSLabel::Background.to_u8(), map)
    }

    #[inline]
    fn liver_boundary_area_group(&self, map: &mut Array2<u8>) -> Areas {
        self.area_group(
            |u| u == LiTSLabel::Liver.to_u8() || u == LiTSLabel::Boundary.to_u8(),
            map,
        )
    }

    fn non_max_merging(&mut self, mut groups: Areas) -> Option<Area> {
        let index = groups.iter().enumerate().max_by_key(|v| v.1.len())?.0;
        for (_, v) in groups.iter().enumerate().filter(|(idx, _)| *idx != index) {
            v.iter().copied().for_each(|p| unsafe {
                *self.uget_mut(p) = LiTSLabel::Background.to_u8();
            });
        }
        Some(std::mem::take(&mut groups[index]))
    }

    fn draw_boundary(&mut self, liver_set: &[Pos]) {
        let targets: Vec<_> = liver_set
            .iter()
            .copied()
            .filter(|p| self.is_liver_boundary(*p))
            .collect();
        targets.iter().for_each(|p| unsafe {
            *self.uget_mut(*p) = LiTSLabel::Boundary.to_u8();
        });
        self.fine_tuning_boundary(targets.as_slice());
    }

    fn fine_tuning_boundary(&mut self, candidates: &[Pos]) {
        for pos in candidates.iter().copied() {
            if !self.is_neighbor4_containing(pos, LiTSLabel::Liver) {
                unsafe {
                    *self.uget_mut(pos) = LiTSLabel::Background.to_u8();
                }
            }
        }
    }

    fn fine_tuning_boundary_one(&mut self, pos: Pos) -> bool {
        if !self.is_neighbor4_containing(pos, LiTSLabel::Liver) {
            unsafe {
                *self.uget_mut(pos) = LiTSLabel::Background.to_u8();
            }
            true
        } else {
            false
        }
    }

    fn draw(&mut self, bg_set: &[Pos], label: LiTSLabel) {
        bg_set.iter().copied().for_each(|p| unsafe {
            *self.uget_mut(p) = label.to_u8();
        })
    }

    #[inline]
    fn is_neighbor4_containing(&self, pos: Pos, label: LiTSLabel) -> bool {
        let (h, w) = pos.to_tuple();
        let t = label.to_u8();
        matches!(self.get((h.wrapping_sub(1), w).into()), Some(&v) if v == t)
            || matches!(self.get((h + 1, w).into()), Some(&v) if v == t)
            || matches!(self.get((h, w.wrapping_sub(1)).into()), Some(&v) if v == t)
            || matches!(self.get((h, w + 1).into()), Some(&v) if v == t)
    }

    fn is_liver_boundary(&self, pos: Pos) -> bool {
        if pos.is_image_boundary() {
            return true;
        }
        let (h, w) = pos.to_tuple();
        let neighbors = unsafe {
            [
                *self.uget((h - 1, w).into()),
                *self.uget((h + 1, w).into()),
                *self.uget((h, w - 1).into()),
                *self.uget((h, w + 1).into()),
            ]
        };
        neighbors.iter().any(|v| *v == LiTSLabel::Liver.to_u8())
            && neighbors
                .iter()
                .any(|v| *v == LiTSLabel::Background.to_u8())
    }

    fn unique(&mut self, map: &mut Array2<u8>) {
        loop {
            map.fill(0);
            let liver_areas = self.liver_area_group(map);
            if liver_areas.is_empty() {
                return;
            }
            // let max_len = liver_areas.iter().map(|a| a.len()).max().unwrap();
            let max_val = liver_areas.iter().max_by_key(|a| a.len()).unwrap();
            for area in liver_areas.iter() {
                if !std::ptr::eq(max_val, area) {
                    self.draw(area.as_slice(), LiTSLabel::Boundary);
                }
            }
            let mut filled = false;
            for p in PosIter::new() {
                if unsafe { self.is_boundary(p) } {
                    filled |= self.fine_tuning_boundary_one(p);
                }
            }
            if !filled {
                return;
            }
        }
    }

    fn bfs(&mut self, start: Pos) -> Vec<JsonValue> {
        let mut q = VecDeque::with_capacity(128);
        let mut ans = Vec::with_capacity(128);
        ans.push(start.into());
        unsafe {
            *self.uget_mut(start) = LiTSLabel::Background.to_u8();
        }
        let ((h_lb, h_ub), (w_lb, w_ub)) = start.window_3x3();
        'outer: for h_cursor in h_lb..h_ub {
            for w_cursor in w_lb..w_ub {
                let p_cursor = (h_cursor, w_cursor).into();
                if unsafe { self.is_boundary(p_cursor) } {
                    q.push_back(p_cursor);
                    break 'outer;
                }
            }
        }

        'outer: while !q.is_empty() {
            for _ in 0..q.len() {
                let p0 = q.pop_front().unwrap();
                if unsafe { !self.is_boundary(p0) } {
                    continue;
                }
                ans.push(p0.into());
                let ((h_lb, h_ub), (w_lb, w_ub)) = p0.window_3x3();
                unsafe {
                    *self.uget_mut(p0) = LiTSLabel::Background.to_u8();
                }
                for h_cursor in h_lb..h_ub {
                    for w_cursor in w_lb..w_ub {
                        let p_cursor = (h_cursor, w_cursor).into();
                        if unsafe { self.is_boundary(p_cursor) } {
                            q.push_back(p_cursor);
                            continue 'outer;
                        }
                    }
                }
            }
        }
        ans.shrink_to_fit();
        ans
    }
}

impl LiverCtSlice {
    fn unify_image(&mut self) {
        // 肿瘤部分被视为肝脏区域
        self.fill_tumor_with_liver();
        let mut map = Array2::<u8>::zeros((LIVER_HEIGHT, LIVER_WIDTH));

        // 找到所有肝脏区域
        let liver_areas = self.liver_area_group(&mut map);

        // 填充所有非最大肝脏区域（填为背景），并获得最大肝脏区域
        let biggest_liver = match self.non_max_merging(liver_areas) {
            Some(liver) => liver,
            None => return,
        };

        map.fill(0);
        // 找到所有背景区域，将封闭背景区域填充为肝脏区域
        let bg_areas = self.background_area_group(&mut map);
        for area in bg_areas.iter() {
            if self.all_inside(area.as_slice()) {
                self.draw(area.as_slice(), LiTSLabel::Liver);
            }
        }

        // 填充肝脏边缘（候选），并细化
        self.draw_boundary(biggest_liver.as_slice());

        map.fill(0);
        // 找到所有独立区域（边缘+内点），只留最大的，其余填充为背景
        let hole_areas = self.liver_boundary_area_group(&mut map);
        self.non_max_merging(hole_areas);

        // 找到所有独立区域（内点）。若只有一个则结束，否则将非最大区域填充为背景，并整图细化，循环
        self.unique(&mut map);
    }

    fn make_label(&mut self, seq: usize) -> (usize, JsonValue) {
        for pos in PosIter::new() {
            if unsafe { self.is_boundary(pos) } {
                return (seq, JsonValue::Array(self.bfs(pos)));
            }
        }
        (seq, json::array![])
    }
}
