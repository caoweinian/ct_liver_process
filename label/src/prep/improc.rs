//! 核心图像处理算法。

use super::{LiverSliceIter, LiverSliceIterMut, Pos, PosIter};
use ndarray::Array2;
use opencv::core::{Mat, CV_8U};
use opencv::highgui::{imshow, wait_key};
use std::collections::{HashSet, VecDeque};
use std::ffi::c_void;
use std::ops::{Index, IndexMut};

/// 图像像素值。
pub mod consts {
    /// 原LiTS17数据集中，背景的像素值。
    pub const LITS_BACKGROUND: u8 = 0;

    /// 原LiTS17数据集中，肝脏的像素值。
    pub const LITS_LIVER: u8 = 1;

    /// 原LiTS17数据集中，肿瘤的像素值。
    pub const LITS_TUMOR: u8 = 2;

    /// 最终视觉友好的背景像素值。
    pub const VIS_BACKGROUND: u8 = 0b_0000_0000;

    /// 最终视觉友好的边缘像素值。
    pub const VIS_BOUNDARY: u8 = 0b_1000_0000;

    /// 最终视觉友好的肝脏像素值。
    pub const VIS_LIVER: u8 = 0b_1111_1111;

    /// 训练集大小。
    pub const TRAINING_SET_LEN: usize = 131;

    /// 测试集大小。
    pub const TESTING_SET_LEN: usize = 70;

    /// 黑色。
    pub const BLACK: u8 = 0;

    /// 灰色。
    pub const GRAY: u8 = 128;

    /// 白色。
    pub const WHITE: u8 = 255;
}

use consts::*;

/// 一张肝脏图像切片。
pub struct LiverSlice {
    pub(crate) data: *mut u8,
    pub(crate) h_len: usize,
    pub(crate) w_len: usize,
    map: Array2<u8>,
}

type Area = Vec<Pos>;
type Areas = Vec<Area>;

impl LiverSlice {
    /// 初始化一张(`h_len` * `w_len`)的肝脏二维图像切片。
    ///
    /// # Safety
    ///
    /// 这张切片必须满足如下性质：
    ///
    /// 1. `pointer`指向的位置保存了至少`(h_len * w_len) * mem::sizeof::<u8>()`个有效的元素。
    /// 2. `pointer`指向的`(h_len * w_len)`个字节像素的取值只有0、1、2。
    /// 3. `pointer`所指向的图像边缘像素值为2。
    ///
    /// 如果上述任意一个条件不能满足，则程序行为未定义。
    #[inline]
    pub unsafe fn new_unchecked(pointer: *mut u8, h_len: usize, w_len: usize) -> Self {
        let map = Array2::<u8>::zeros((h_len, w_len));
        Self {
            data: pointer,
            h_len,
            w_len,
            map,
        }
    }

    /// 调试用；显示图片。
    pub fn debug_show(&self) {
        let mat = unsafe {
            Mat::new_rows_cols_with_data(
                self.h_len as i32,
                self.w_len as i32,
                CV_8U,
                self.data as *mut c_void,
                self.h_len,
            )
        }
        .unwrap();
        imshow("debug_show", &mat).unwrap();
        wait_key(0).unwrap();
    }

    /// 计算图像中目前值为`label`的像素个数。
    pub fn count(&self, label: u8) -> usize {
        self.iter().copied().filter(|&v| v == label).count()
    }

    /// 将图像中所有值为`old`的像素替换为`new`，返回因此而被修改的像素值个数。
    pub fn fill(&mut self, old: u8, new: u8) -> usize {
        let mut cnt = 0;
        unsafe {
            self.iter_mut().filter(|v| **v == old).for_each(|v| {
                cnt += 1;
                *v = new
            });
        }
        cnt
    }

    /// 将`set`位置的所有坐标对应的像素值全部置为`label`。
    ///
    /// # Safety
    ///
    /// 必须保证`set`中的所有坐标都在图像范围内，否则行为未定义。
    pub unsafe fn fill_batch(&mut self, set: &[Pos], label: u8) {
        self.fill_batch_from(set.iter().copied(), label);
    }

    /// 将`set`迭代器的所有坐标对应的像素值全部置为`label`。
    ///
    /// # Safety
    ///
    /// 必须保证`set`中的所有坐标都在图像范围内，否则行为未定义。
    pub unsafe fn fill_batch_from<I: Iterator<Item = Pos>>(&mut self, set: I, label: u8) {
        set.for_each(|p| *self.uget_mut(p) = label);
    }

    /// 判断`pos`位置的像素是否是（可视化）边缘像素。
    ///
    /// # Safety
    ///
    /// 必须保证`pos`在图像范围内，否则行为未定义。
    #[inline]
    pub unsafe fn is_vis_boundary(&self, pos: Pos) -> bool {
        *self.uget(pos) == VIS_BOUNDARY
    }

    /// 判断整图的像素是否都是（可视化）背景像素。
    pub fn is_all_vis_background(&self) -> bool {
        self.iter().all(|&p| p == VIS_BACKGROUND)
    }

    /// 判断`pos`位置的像素是否是（可视化）背景像素。
    ///
    /// # Safety
    ///
    /// 必须保证`pos`在图像范围内，否则行为未定义。
    #[inline]
    pub unsafe fn is_vis_background(&self, pos: Pos) -> bool {
        *self.uget(pos) == VIS_BACKGROUND
    }

    /// 判断`pos`位置的像素是否是（可视化）肝脏像素。
    ///
    /// # Safety
    ///
    /// 必须保证`pos`在图像范围内，否则行为未定义。
    #[inline]
    pub unsafe fn is_vis_liver(&self, pos: Pos) -> bool {
        *self.uget(pos) == VIS_LIVER
    }

    /// 获得图像中的所有肝脏像素位置。
    pub fn vis_liver_pixels(&self) -> Vec<Pos> {
        PosIter::from(self)
            .filter(|p| unsafe { *self.uget(*p) == VIS_LIVER })
            .collect()
    }

    /// 获得图像中的所有肝脏像素位置。
    pub fn vis_boundary_pixels(&self) -> Vec<Pos> {
        PosIter::from(self)
            .filter(|p| unsafe { *self.uget(*p) == VIS_BOUNDARY })
            .collect()
    }

    /// 获得图像中的所有背景像素位置。
    pub fn vis_background_pixels(&self) -> Vec<Pos> {
        PosIter::from(self)
            .filter(|p| unsafe { *self.uget(*p) == VIS_BACKGROUND })
            .collect()
    }

    /// 判断`pos`位置的像素是否等于`pixel`。
    ///
    /// # Safety
    ///
    /// 如果`pos`像素越界，则程序panic。
    #[inline]
    pub fn pixel_eq(&self, pos: Pos, pixel: u8) -> bool {
        self.check(pos)
            .then(|| unsafe { self.pixel_eq_unchecked(pos, pixel) })
            .unwrap()
    }

    /// 判断`pos`位置的像素是否不等于`pixel`。
    ///
    /// # Safety
    ///
    /// 如果`pos`像素越界，则程序panic。
    #[inline]
    pub fn pixel_ne(&self, pos: Pos, pixel: u8) -> bool {
        !self.pixel_eq(pos, pixel)
    }

    /// 判断`pos`位置的像素是否等于`pixel`，不进行越界检查。
    ///
    /// # Safety
    ///
    /// 如果`pos`像素越界，则程序行为未定义。
    #[inline]
    pub unsafe fn pixel_eq_unchecked(&self, pos: Pos, pixel: u8) -> bool {
        *self.uget(pos) == pixel
    }

    /// 判断`pos`位置的像素是否不等于`pixel`，不进行越界检查。
    ///
    /// # Safety
    ///
    /// 如果`pos`像素越界，则程序行为未定义。
    #[inline]
    pub unsafe fn pixel_ne_unchecked(&self, pos: Pos, pixel: u8) -> bool {
        !self.pixel_eq_unchecked(pos, pixel)
    }

    /// 获得`pos`位置对应像素的不可变引用，不进行越界检查。
    ///
    /// # Safety
    ///
    /// 如果`pos`像素越界，则程序行为未定义。
    #[inline]
    pub unsafe fn uget(&self, p: Pos) -> &u8 {
        &*(self.data.add(p.h * self.w_len + p.w))
    }

    /// 获得`pos`位置对应像素的可变引用，不进行越界检查。
    ///
    /// # Safety
    ///
    /// 如果`pos`像素越界，则程序行为未定义。
    #[inline]
    pub unsafe fn uget_mut(&mut self, p: Pos) -> &mut u8 {
        &mut *(self.data.add(p.h * self.w_len + p.w))
    }

    /// 获得`pos`位置对应像素的不可变引用。如果越界则返回`None`。
    #[inline]
    pub fn get(&self, p: Pos) -> Option<&u8> {
        // println!("{}", self.check(p));
        self.check(p).then(|| unsafe { self.uget(p) })
    }

    /// 获得`pos`位置对应像素的可变引用。如果越界则返回`None`。
    #[inline]
    pub fn get_mut(&mut self, p: Pos) -> Option<&mut u8> {
        self.check(p).then(|| unsafe { self.uget_mut(p) })
    }

    /// 检查是否越界。若越界则返回`false`，否则返回`true`。
    #[inline]
    fn check(&self, pos: Pos) -> bool {
        // println!("{}, {}, {}, {}", pos.h, self.h_len, pos.w, self.w_len);
        pos.h < self.h_len && pos.w < self.w_len
    }

    /// 消除肿瘤像素（因为在我们的科研任务中不需要这部分），并将图像的旧（LiTS）像素迁移到新（可视化0/128/255）像素。如果原图为全背景图则返回false，否则返回true（意味着已更改）。
    pub fn migrate_pixels(&mut self) -> bool {
        let mut modified = false;
        for pos in unsafe { self.iter_mut() } {
            match *pos {
                LITS_LIVER | LITS_TUMOR => {
                    modified = true;
                    *pos = VIS_LIVER
                }
                _ => (),
            }
        }
        modified
    }

    /// 获得一个迭代器，其可以按行优先顺序枚举所有像素的像素值（不可变）。
    #[inline]
    pub fn iter(&self) -> LiverSliceIter {
        LiverSliceIter::new(self)
    }

    /// 获得一个迭代器，其可以按行优先顺序枚举所有像素的像素值（可变）。
    ///
    /// # Safety
    ///
    /// 如果用户将其中的像素更改为可视化像素(`VIS_LIVER`, `VIS_BOUNDARY`, `VIS_BACKGROUND`)以外的像素，则程序行为未定义。
    #[inline]
    pub unsafe fn iter_mut(&mut self) -> LiverSliceIterMut {
        LiverSliceIterMut::new(self)
    }

    /// 按照4-相邻规则从候选区域`it`中获取所有区域。两个像素p1和p2属于一个区域，当且仅当存在一条从p1到p2的4-相邻路径，且路径上的所有点（包括p1和p2）都满足谓词`pixel_pred`。
    ///
    /// # Safety
    ///
    /// 必须保证`it`所指向的元素最多出现一次，且都在图像范围内，否则程序行为未定义。
    pub unsafe fn area_group_from_local_immut<I: Iterator<Item = Pos>>(
        &self,
        it: I,
        pixel_pred: fn(u8) -> bool,
    ) -> Areas {
        let mut ans: Areas = Vec::with_capacity(1);
        let mut bfs_q: VecDeque<Pos> = VecDeque::with_capacity(4);
        let mut visited: HashSet<Pos> = HashSet::with_capacity(64);

        for pos in it {
            if visited.contains(&pos) || !pixel_pred(*self.uget(pos)) {
                continue;
            }
            bfs_q.push_back(pos);
            let mut one_area: Area = Vec::with_capacity(1);
            while !bfs_q.is_empty() {
                let cur_pos = bfs_q.pop_front().unwrap();
                if visited.contains(&cur_pos) {
                    continue;
                }
                let (cur_h, cur_w) = cur_pos.to_tuple();
                visited.insert(cur_pos);
                one_area.push(cur_pos);
                if cur_h > 0
                    && pixel_pred(*self.uget((cur_h - 1, cur_w).into()))
                    && !visited.contains(&(cur_h - 1, cur_w).into())
                {
                    bfs_q.push_back((cur_h - 1, cur_w).into());
                }
                if cur_h + 1 < self.h_len
                    && pixel_pred(*self.uget((cur_h + 1, cur_w).into()))
                    && !visited.contains(&(cur_h + 1, cur_w).into())
                {
                    bfs_q.push_back((cur_h + 1, cur_w).into());
                }
                if cur_w > 0
                    && pixel_pred(*self.uget((cur_h, cur_w - 1).into()))
                    && !visited.contains(&(cur_h, cur_w - 1).into())
                {
                    bfs_q.push_back((cur_h, cur_w - 1).into());
                }
                if cur_w + 1 < self.w_len
                    && pixel_pred(*self.uget((cur_h, cur_w + 1).into()))
                    && !visited.contains(&(cur_h, cur_w + 1).into())
                {
                    bfs_q.push_back((cur_h, cur_w + 1).into());
                }
            }
            ans.push(one_area);
        }
        ans
    }

    /// 按照4-相邻规则从候选区域`it`中获取所有区域。两个像素p1和p2属于一个区域，当且仅当存在一条从p1到p2的4-相邻路径，且路径上的所有点（包括p1和p2）都满足谓词`pixel_pred`。
    ///
    /// 注意该函数会修改`self.map`。用户需要确保`self.map`已被清空为全零（可通过`self.clear_map()`），否则该函数功能可能不正确。
    ///
    /// # Safety
    ///
    /// 必须保证`it`所指向的元素最多出现一次，且都在图像范围内，否则程序行为未定义。
    pub unsafe fn area_group_from_local<I: Iterator<Item = Pos>>(
        &mut self,
        it: I,
        pixel_pred: fn(u8) -> bool,
    ) -> Areas {
        let mut ans: Areas = Vec::with_capacity(1);
        let mut bfs_q: VecDeque<Pos> = VecDeque::with_capacity(4);

        for pos in it {
            match *self.map.uget(pos.to_tuple()) {
                0 => {
                    if !pixel_pred(*self.uget(pos)) {
                        continue;
                    }
                    bfs_q.push_back(pos);
                    let mut one_area: Area = Vec::with_capacity(1);
                    while !bfs_q.is_empty() {
                        let cur_pos = bfs_q.pop_front().unwrap();
                        let (cur_h, cur_w) = cur_pos.to_tuple();
                        let p: &mut u8 = self.map.uget_mut((cur_h, cur_w));
                        if *p == 1 {
                            continue;
                        }
                        *p = 1;
                        one_area.push(cur_pos);
                        if cur_h > 0
                            && pixel_pred(*self.uget((cur_h - 1, cur_w).into()))
                            && *self.map.uget((cur_h - 1, cur_w)) == 0
                        {
                            bfs_q.push_back((cur_h - 1, cur_w).into());
                        }
                        if cur_h + 1 < self.h_len
                            && pixel_pred(*self.uget((cur_h + 1, cur_w).into()))
                            && *self.map.uget((cur_h + 1, cur_w)) == 0
                        {
                            bfs_q.push_back((cur_h + 1, cur_w).into());
                        }
                        if cur_w > 0
                            && pixel_pred(*self.uget((cur_h, cur_w - 1).into()))
                            && *self.map.uget((cur_h, cur_w - 1)) == 0
                        {
                            bfs_q.push_back((cur_h, cur_w - 1).into());
                        }
                        if cur_w + 1 < self.w_len
                            && pixel_pred(*self.uget((cur_h, cur_w + 1).into()))
                            && *self.map.uget((cur_h, cur_w + 1)) == 0
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
        ans
    }

    /// 按照4-相邻规则获取所有区域。两个像素p1和p2属于一个区域，当且仅当存在一条从p1到p2的4-相邻路径，且路径上的所有点（包括p1和p2）都满足谓词`pixel_pred`。
    ///
    /// 注意该函数会修改`self.map`。用户需要确保`self.map`已被清空为全零（可通过`self.clear_map()`），否则该函数功能可能不正确。
    #[inline]
    pub fn area_group(&mut self, pixel_pred: fn(u8) -> bool) -> Areas {
        let pos_iter = PosIter::from(&*self);
        unsafe { self.area_group_from_local(pos_iter, pixel_pred) }
    }

    /// 按照4-相邻规则获取所有可视化肝脏区域。
    ///
    /// 注意该函数会修改`self.map`。用户需要确保`self.map`已被清空为全零（可通过`self.clear_map()`），否则该函数功能可能不正确。
    pub fn vis_liver_area_group(&mut self) -> Areas {
        self.area_group(|u| u == VIS_LIVER)
    }

    /// 按照4-相邻规则获取所有可视化背景区域。
    ///
    /// 注意该函数会修改`self.map`。用户需要确保`self.map`已被清空为全零（可通过`self.clear_map()`），否则该函数功能可能不正确。
    pub fn vis_background_area_group(&mut self) -> Areas {
        self.area_group(|u| u == VIS_BACKGROUND)
    }

    /// 按照4-相邻规则获取所有(可视化肝脏-可视化边缘)复合区域。
    ///
    /// 注意该函数会修改`self.map`。用户需要确保`self.map`已被清空为全零（可通过`self.clear_map()`），否则该函数功能可能不正确。
    pub fn vis_liver_boundary_area_group(&mut self) -> Areas {
        self.area_group(|u| u == VIS_LIVER || u == VIS_BOUNDARY)
    }

    /// 获取`groups`中第一个最大的区域（并返回），同时将其它区域都填充为像素`fill_with_pixel`。如果`groups`为空，则返回None。
    pub fn non_max_filling(&mut self, mut groups: Areas, fill_with_pixel: u8) -> Option<Area> {
        let index = groups.iter().enumerate().max_by_key(|v| v.1.len())?.0;
        for (_, v) in groups.iter().enumerate().filter(|(idx, _)| *idx != index) {
            v.iter().copied().for_each(|p| unsafe {
                *self.uget_mut(p) = fill_with_pixel;
            })
        }
        Some(std::mem::take(&mut groups[index]))
    }

    /// 清除图遍历信息记录缓存。
    #[inline]
    pub fn clear_map(&mut self) {
        self.map.fill(0_u8);
    }

    /// 判断图中`pos`像素的4-邻域是否包含像素`label`。
    ///
    /// 如果`pos`越界，则返回值无意义。
    #[inline]
    pub fn is_n4_containing(&self, pos: Pos, label: u8) -> bool {
        let (h, w) = pos.to_tuple();
        matches!(self.get((h.wrapping_sub(1), w).into()), Some(&v) if v == label)
            || matches!(self.get((h + 1, w).into()), Some(&v) if v == label)
            || matches!(self.get((h, w.wrapping_sub(1)).into()), Some(&v) if v == label)
            || matches!(self.get((h, w + 1).into()), Some(&v) if v == label)
    }

    /// 判断图中`pos`像素的8-邻域是否包含像素`label`。
    ///
    /// 如果`pos`越界，则返回值无意义。
    #[inline]
    pub fn is_n8_containing(&self, pos: Pos, label: u8) -> bool {
        let (h, w) = pos.to_tuple();
        self.is_n4_containing(pos, label)
            || matches!(self.get((h.wrapping_sub(1), w.wrapping_sub(1)).into()), Some(&v) if v == label)
            || matches!(self.get((h.wrapping_sub(1), w + 1).into()), Some(&v) if v == label)
            || matches!(self.get((h + 1, w.wrapping_sub(1)).into()), Some(&v) if v == label)
            || matches!(self.get((h + 1, w + 1).into()), Some(&v) if v == label)
    }

    /// 获得图中`pos`像素的4-邻域像素坐标。
    ///
    /// 如果`pos`越界，则返回值无意义。
    pub fn n4_positions(&self, pos: Pos) -> Vec<Pos> {
        let (h, w) = pos.to_tuple();
        let p1: Pos = (h.wrapping_sub(1), w).into();
        let p2: Pos = (h + 1, w).into();
        let p3: Pos = (h, w.wrapping_sub(1)).into();
        let p4: Pos = (h, w + 1).into();
        [p1, p2, p3, p4]
            .into_iter()
            .filter(|p| self.check(*p))
            .collect()
    }

    /// 统计图中`pos`像素4-邻域中像素值等于`label`的个数。
    ///
    /// 如果`pos`越界，则返回值无意义。
    pub fn n4_count(&self, pos: Pos, label: u8) -> usize {
        let (h, w) = pos.to_tuple();
        let p1: Pos = (h.wrapping_sub(1), w).into();
        let p2: Pos = (h + 1, w).into();
        let p3: Pos = (h, w.wrapping_sub(1)).into();
        let p4: Pos = (h, w + 1).into();
        [p1, p2, p3, p4]
            .into_iter()
            .filter_map(|p| self.get(p).cloned())
            .filter(|u| *u == label)
            .count()
    }

    /// 获得图中`pos`像素的8-邻域像素坐标。
    ///
    /// 如果`pos`越界，则返回值无意义。
    pub fn n8_positions(&self, pos: Pos) -> Vec<Pos> {
        let (h, w) = pos.to_tuple();
        let p1: Pos = (h.wrapping_sub(1), w).into();
        let p2: Pos = (h + 1, w).into();
        let p3: Pos = (h, w.wrapping_sub(1)).into();
        let p4: Pos = (h, w + 1).into();
        let p5: Pos = (h.wrapping_sub(1), w.wrapping_sub(1)).into();
        let p6: Pos = (h.wrapping_sub(1), w + 1).into();
        let p7: Pos = (h + 1, w.wrapping_sub(1)).into();
        let p8: Pos = (h + 1, w + 1).into();
        [p1, p2, p3, p4, p5, p6, p7, p8]
            .into_iter()
            .filter(|p| self.check(*p))
            .collect()
    }

    /// 描绘一个肝脏区域的边缘，然后进行平滑化。返回被侵蚀为背景的边缘像素个数。
    #[inline]
    pub fn draw_liver_boundary(&mut self, liver_set: &[Pos]) -> usize {
        let targets: Vec<Pos> = liver_set
            .iter()
            .copied()
            .filter(|p| self.is_n4_containing(*p, VIS_BACKGROUND))
            .collect();
        targets.iter().for_each(|p| unsafe {
            *self.uget_mut(*p) = VIS_BOUNDARY;
        });
        self.shrink_boundary_pixels(targets.as_slice())
    }

    /// 肝脏边缘平滑化。返回被侵蚀为背景的边缘像素个数。
    fn shrink_boundary_pixels(&mut self, candidates: &[Pos]) -> usize {
        let mut eroded = 0;
        for pos in candidates.iter().copied() {
            if !self.is_n4_containing(pos, VIS_LIVER) {
                unsafe {
                    *self.uget_mut(pos) = VIS_BACKGROUND;
                }
                eroded += 1;
            }
        }
        eroded
    }

    /// 最后的循环，保证最终能够得到一张单一连接的规范化肝脏+边缘。返回循环次数。
    #[deprecated(
        since = "0.3.0",
        note = "该方法命名有误——不需要执行循环；请使用`unique()`"
    )]
    pub fn unique_loop(&mut self) -> usize {
        for loop_times in 1.. {
            self.clear_map();
            let liver_areas = self.vis_liver_area_group();
            if liver_areas.is_empty() {
                return loop_times - 1;
            }
            let max_val = liver_areas.iter().max_by_key(|a| a.len()).unwrap();
            for area in liver_areas.iter() {
                if !std::ptr::eq(max_val, area) {
                    unsafe {
                        self.fill_batch(area.as_slice(), VIS_BOUNDARY);
                    }
                }
            }
            let mut filled = false;
            for p in PosIter::from(&*self) {
                unsafe {
                    if self.is_vis_boundary(p) {
                        filled |= self.try_erase_redundant_boundary(p);
                    }
                }
            }
            if !filled {
                return loop_times;
            }
        }
        unreachable!()
    }

    /// 最终细化为一张单一连接的规范化肝脏+边缘。
    pub fn unique(&mut self) {
        self.clear_map();
        let liver_areas = self.vis_liver_area_group();
        if liver_areas.is_empty() {
            return;
        }
        let max_val = liver_areas.iter().max_by_key(|a| a.len()).unwrap();
        for area in liver_areas.iter() {
            if !std::ptr::eq(max_val, area) {
                unsafe {
                    self.fill_batch(area.as_slice(), VIS_BOUNDARY);
                }
            }
        }
        for p in PosIter::from(&*self) {
            unsafe {
                if self.is_vis_boundary(p) {
                    self.try_erase_redundant_boundary(p);
                }
            }
        }
    }

    /// `pos`代表一个可视化边缘像素。若其4-邻域不包含可视化肝脏像素，就将其涂改为可视化边缘像素。
    unsafe fn try_erase_redundant_boundary(&mut self, pos: Pos) -> bool {
        if !self.is_n4_containing(pos, VIS_LIVER) {
            *self.uget_mut(pos) = VIS_BACKGROUND;
            true
        } else {
            false
        }
    }

    /// 判断`pos`是否在图像的边缘位置。
    ///
    /// 注意该函数不会检查`pos`是否越界（越界时，返回值无意义）。函数仅保证当`pos`在图像范围内的正确性。
    pub fn is_at_image_edge(&self, pos: Pos) -> bool {
        pos.w == 0 || pos.h == 0 || pos.w + 1 == self.w_len || pos.h + 1 == self.h_len
    }

    /// 判断`positions`的所有像素是否都在图像内部。如果都在内部则返回true，否则返回false。
    pub fn all_within(&self, positions: &[Pos]) -> bool {
        positions.iter().all(|p| !self.is_at_image_edge(*p))
    }
}

impl Index<Pos> for LiverSlice {
    type Output = u8;

    fn index(&self, index: Pos) -> &Self::Output {
        self.get(index).unwrap()
    }
}

impl Index<(usize, usize)> for LiverSlice {
    type Output = u8;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        &self[Pos::from(index)]
    }
}

impl IndexMut<Pos> for LiverSlice {
    fn index_mut(&mut self, index: Pos) -> &mut Self::Output {
        self.get_mut(index).unwrap()
    }
}

impl IndexMut<(usize, usize)> for LiverSlice {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        &mut self[Pos::from(index)]
    }
}
