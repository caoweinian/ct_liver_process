use crate::subcmd_impls::{AREA, WIDTH};
use clap::Args;
use label::prelude::{
    LiverSlice, LITS_BACKGROUND, TRAINING_SET_LEN, VIS_BACKGROUND, VIS_BOUNDARY, VIS_LIVER,
};
use label::prep::Pos;
use ndarray::Array3;
use std::collections::HashSet;
use std::path::PathBuf;

#[derive(Args, Debug)]
pub struct Verify {
    /// 数据集根目录。
    #[arg(long = "base-dir", short = 'D')]
    base_dir: PathBuf,
}

impl Verify {
    pub fn run(&mut self) {
        self.base_dir
            .extend(["medical_liver", "LiTS", "lits_train", "label_out"]);

        let mut raw_npy_dir = self.base_dir.clone();
        raw_npy_dir.push("raw_npy");
        assert!(raw_npy_dir.is_dir());

        let mut uni_npy_dir = self.base_dir.clone();
        uni_npy_dir.push("unified_npy");
        assert!(uni_npy_dir.is_dir());

        Program::new(raw_npy_dir, uni_npy_dir).run();
    }
}

struct Program {
    raw_npy_dir: PathBuf,
    uni_npy_dir: PathBuf,
    file_seq: usize,
    slice_seq: usize,
}

impl Program {
    #[inline]
    pub fn new(raw_npy_dir: PathBuf, uni_npy_dir: PathBuf) -> Self {
        Self {
            raw_npy_dir,
            uni_npy_dir,
            file_seq: 0,
            slice_seq: 0,
        }
    }

    pub fn run(&mut self) {
        while self.file_seq != TRAINING_SET_LEN {
            println!("正在文件`{}.npy`中验证性质...", self.file_seq);
            self.run_seq();
            self.file_seq += 1;
        }
    }

    #[inline]
    fn run_seq(&mut self) {
        self.check_raw();
        self.check_unified();
    }

    #[inline]
    fn print_failed_info_pos(&self, fn_name: &str, pos: Pos) {
        eprintln!(
            "`{fn_name}` failed: in {file_seq}.npy, slice {slice_seq}, position ({h}, {w}).",
            file_seq = self.file_seq,
            slice_seq = self.slice_seq,
            h = pos.h,
            w = pos.w,
        );
    }

    #[inline]
    fn print_failed_into(&self, fn_name: &str) {
        eprintln!(
            "`{fn_name}` failed: in {}.npy, slice {}.",
            self.file_seq, self.slice_seq
        );
    }

    fn check_raw(&mut self) -> bool {
        // 检验raw_npy目录下文件的边缘都是背景
        let mut ok = true;
        let filename = format!("{}.npy", self.file_seq);
        self.raw_npy_dir.push(filename.as_str());

        let liver_3d: Array3<u8> = ndarray_npy::read_npy(self.raw_npy_dir.as_path()).unwrap();
        let mut liver_vec: Vec<u8> = liver_3d.into_raw_vec();

        let slice_len = liver_vec.len() / AREA;
        println!("\t检验边缘像素中...");

        for slice_seq in 0..slice_len {
            self.slice_seq = slice_seq;
            let offset = slice_seq * AREA;
            let ct = unsafe {
                LiverSlice::new_unchecked(liver_vec.as_mut_ptr().add(offset), WIDTH, WIDTH)
            };
            for j in 0..WIDTH {
                if ct[(0, j)] != LITS_BACKGROUND {
                    self.print_failed_info_pos("check_raw", (0, j).into());
                    ok = false;
                }
                if ct[(WIDTH - 1, j)] != LITS_BACKGROUND {
                    self.print_failed_info_pos("check_raw", (WIDTH - 1, j).into());
                    ok = false;
                }
            }
            for i in 0..WIDTH {
                if ct[(i, 0)] != LITS_BACKGROUND {
                    self.print_failed_info_pos("check_raw", (i, 0).into());
                    ok = false;
                }
                if ct[(i, WIDTH - 1)] != LITS_BACKGROUND {
                    self.print_failed_info_pos("check_raw", (i, WIDTH - 1).into());
                    ok = false;
                }
            }
        }
        self.raw_npy_dir.pop();
        ok
    }

    fn check_unified(&mut self) -> bool {
        // 检验unified_npy目录下文件的各种性质
        let mut ok = true;
        let filename = format!("{}.npy", self.file_seq);
        self.uni_npy_dir.push(filename.as_str());

        let liver_3d: Array3<u8> = ndarray_npy::read_npy(self.uni_npy_dir.as_path()).unwrap();
        let mut liver_vec: Vec<u8> = liver_3d.into_raw_vec();

        let slice_len = liver_vec.len() / AREA;
        println!("\t检验算法正确性中...");

        for slice_seq in 0..slice_len {
            self.slice_seq = slice_seq;
            let offset = slice_seq * AREA;
            let mut ct = unsafe {
                LiverSlice::new_unchecked(liver_vec.as_mut_ptr().add(offset), WIDTH, WIDTH)
            };
            ok &= self.check_liver_background_not_n4(&ct);
            ok &= self.check_boundary_n4_contains_liver_and_background(&ct);
            ok &= self.check_at_most_1_liver_boundary_compound_area(&mut ct);
            ok &= self.check_at_most_1_liver_area(&mut ct);
            ok &= self.check_exact_1_background_area(&mut ct);
            ok &= self.check_boundary_n8_has_exact_2_boundaries(&ct);
            ok &= self.check_boundary_n8_angle(&ct);
            ok &= self.check_boundary_circle_and_full(&ct);
        }

        self.uni_npy_dir.pop();
        ok
    }

    fn check_liver_background_not_n4(&self, ct: &LiverSlice) -> bool {
        let mut ok = true;
        for pos in ct.vis_liver_pixels() {
            if ct.is_n4_containing(pos, VIS_BACKGROUND) {
                self.print_failed_info_pos("check_liver_background_not_n4", pos);
                ok = false;
            }
        }
        for pos in ct.vis_background_pixels() {
            if ct.is_n4_containing(pos, VIS_LIVER) {
                self.print_failed_info_pos("check_liver_background_not_n4", pos);
                ok = false;
            }
        }
        ok
    }

    fn check_boundary_n4_contains_liver_and_background(&self, ct: &LiverSlice) -> bool {
        let mut ok = true;
        for pos in ct.vis_boundary_pixels() {
            if !ct.is_n4_containing(pos, VIS_LIVER) || !ct.is_n4_containing(pos, VIS_BACKGROUND) {
                self.print_failed_info_pos("check_boundary_n4_contains_liver_and_background", pos);
                ok = false;
            }
        }
        ok
    }

    fn check_at_most_1_liver_boundary_compound_area(&self, ct: &mut LiverSlice) -> bool {
        let area = ct.vis_liver_boundary_area_group();
        ct.clear_map(); // invariants
        if area.len() > 1 {
            self.print_failed_into("check_at_most_1_liver_boundary_compound_area");
            return false;
        }
        true
    }

    fn check_at_most_1_liver_area(&self, ct: &mut LiverSlice) -> bool {
        let area = ct.vis_liver_area_group();
        ct.clear_map(); // invariants
        if area.len() > 1 {
            self.print_failed_into("check_at_most_1_liver_area");
            return false;
        }
        true
    }

    fn check_exact_1_background_area(&self, ct: &mut LiverSlice) -> bool {
        let area = ct.vis_background_area_group();
        ct.clear_map(); // invariants
        if area.len() != 1 {
            self.print_failed_into("check_at_most_1_background_area");
            return false;
        }
        true
    }

    fn check_boundary_n8_has_exact_2_boundaries(&self, ct: &LiverSlice) -> bool {
        let mut ok = true;
        for pos in ct.vis_boundary_pixels() {
            let n8 = ct.n8_positions(pos);
            if n8
                .into_iter()
                .filter(|n| *ct.get(*n).unwrap() == VIS_BOUNDARY)
                .count()
                != 2
            {
                self.print_failed_info_pos("check_boundary_n8_has_exact_2_boundaries", pos);
                ok = false;
            }
        }
        ok
    }

    fn check_boundary_n8_angle(&self, ct: &LiverSlice) -> bool {
        let mut ok = true;
        for pos in ct.vis_boundary_pixels() {
            let n8 = ct.n8_positions(pos);
            let mut it = n8
                .into_iter()
                .filter(|n| *ct.get(*n).unwrap() == VIS_BOUNDARY);
            let p1 = it.next().unwrap();
            let p2 = it.next().unwrap();
            assert!(it.next().is_none());
            let v1 = p1 - pos;
            let v2 = p2 - pos;
            let dot = v1.dot(v2);
            if dot > 0 || (dot == 0 && (v1.is_axis_direction() || v2.is_axis_direction())) {
                ok = false;
                self.print_failed_info_pos("check_boundary_n8_angle", pos);
            }
        }
        ok
    }

    fn check_boundary_circle_and_full(&self, ct: &LiverSlice) -> bool {
        let pixels = ct.vis_boundary_pixels();
        if pixels.is_empty() {
            return true;
        }

        let mut marked: HashSet<Pos> = HashSet::with_capacity(pixels.len());
        let p0 = pixels[0];
        marked.insert(p0);
        let mut cur_pos = p0;

        'outer: loop {
            let n8 = ct.n8_positions(cur_pos);
            for neighbor in n8 {
                if unsafe { *ct.uget(neighbor) == VIS_BOUNDARY } && !marked.contains(&neighbor) {
                    marked.insert(neighbor);
                    cur_pos = neighbor;
                    continue 'outer;
                }
            }
            break 'outer;
        }

        if marked.len() != pixels.len() {
            self.print_failed_into("check_boundary_circle_and_full");
            return false;
        }
        true
    }
}
