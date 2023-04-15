use crate::subcmd_impls::{AREA, WIDTH};
use clap::Args;
use label::prelude::{LiverSlice, Pos, TRAINING_SET_LEN};
use label::prep::improc::consts::{BLACK, GRAY, WHITE};
use ndarray::Array3;
use std::collections::HashSet;
use std::mem;
use std::path::PathBuf;

#[derive(Args, Debug)]
pub struct ErodedCoefficient {
    /// 数据集根目录。
    #[arg(long = "base-dir", short = 'D')]
    base_dir: PathBuf,
}

impl ErodedCoefficient {
    pub fn run(&mut self) {
        // [label_out/raw_npy]
        self.base_dir.extend([
            "medical_liver",
            "LiTS",
            "lits_train",
            "label_out",
            "unique_npy",
        ]);
        assert!(self.base_dir.is_dir());

        Program::new(mem::take(&mut self.base_dir)).run();
    }
}

struct Program {
    unique_npy_dir: PathBuf,
    foreground_count: u64,
    hrvoje_eroded: u64,
    ours_eroded: u64,
    seq: usize,
}

impl Program {
    #[inline]
    pub fn new(unique_npy_dir: PathBuf) -> Self {
        Self {
            unique_npy_dir,
            foreground_count: 0,
            hrvoje_eroded: 0,
            ours_eroded: 0,
            seq: 0,
        }
    }

    pub fn run(&mut self) {
        while self.seq != TRAINING_SET_LEN {
            println!("处理目录`{}`...", self.seq);
            self.run_seq();
            self.seq += 1;
        }
        self.summary();
    }

    fn summary(&self) {
        println!("----------------------------------------------------------");
        Self::summary_item(self.hrvoje_eroded, self.foreground_count, "hrvoje");
        Self::summary_item(self.ours_eroded, self.foreground_count, "ours");
        println!("----------------------------------------------------------");
    }

    #[inline]
    fn summary_item(total: u64, count: u64, algo_name: &str) {
        println!(
            "{algo_name}算法\n\t总腐蚀像素: {total}\n\t腐蚀系数: {}",
            total as f64 / count as f64
        );
    }

    fn run_seq(&mut self) {
        let filename = format!("{}.npy", self.seq);
        self.unique_npy_dir.push(filename.as_str());

        let liver_3d: Array3<u8> = ndarray_npy::read_npy(self.unique_npy_dir.as_path()).unwrap();
        let mut liver_vec: Vec<u8> = liver_3d.into_raw_vec();
        let slice_len = liver_vec.len() / AREA;
        for i in 0..slice_len {
            let offset = i * AREA;
            let ct = unsafe {
                LiverSlice::new_unchecked(liver_vec.as_mut_ptr().add(offset), WIDTH, WIDTH)
            };
            if ct.is_all_vis_background() {
                continue;
            }
            self.foreground_count += 1;

            let mut img_hrvoje = liver_vec[offset..offset + AREA].to_vec();
            let mut img_ours = img_hrvoje.clone();

            let hrvoje_ct =
                unsafe { LiverSlice::new_unchecked(img_hrvoje.as_mut_ptr(), WIDTH, WIDTH) };
            let hrvoje_cnt = Self::count_hrvoje(hrvoje_ct);
            self.hrvoje_eroded += hrvoje_cnt;

            let ours_ct = unsafe { LiverSlice::new_unchecked(img_ours.as_mut_ptr(), WIDTH, WIDTH) };
            let ours_cnt = Self::count_ours(ours_ct);
            self.ours_eroded += ours_cnt;

            if ours_cnt > hrvoje_cnt {
                println!(
                    "Note: in `{}.npy`, slice {i}: hrvoje_cnt = {hrvoje_cnt}, ours = {ours_cnt}",
                    self.seq
                );
            }
        }
        self.unique_npy_dir.pop();
    }

    fn count_hrvoje(mut ct: LiverSlice) -> u64 {
        let mut fg: HashSet<Pos> = ct.vis_liver_pixels().into_iter().collect();
        // boundaries是粗略边缘
        let boundaries = fg
            .iter()
            .cloned()
            .filter(|p| ct.is_n4_containing(*p, BLACK))
            .collect::<Vec<_>>();
        let mut cnt: u64 = boundaries.len() as u64;

        // 形态学腐蚀
        for pos in boundaries.iter().cloned() {
            unsafe {
                *ct.uget_mut(pos) = BLACK;
            }
            fg.remove(&pos);
        }

        loop {
            // `bd`的过滤条件是hrvoje算法的核心逻辑
            let bd = fg
                .iter()
                .cloned()
                .filter(|p| ct.n4_count(*p, WHITE) < 2)
                .collect::<Vec<_>>();
            if bd.is_empty() {
                break;
            }
            cnt += bd.len() as u64;
            for pos in bd.iter().cloned() {
                unsafe {
                    *ct.uget_mut(pos) = BLACK;
                }
                fg.remove(&pos);
            }
        }

        // 注意从上面的循环跳出后，根据hrvoje算法的特性，可能生成多个孔洞。需要考虑去除问题。
        let areas = unsafe { ct.area_group_from_local_immut(fg.iter().cloned(), |u| u == WHITE) };
        if areas.is_empty() {
            return cnt;
        }
        let index = areas
            .iter()
            .enumerate()
            .max_by_key(|v| v.1.len())
            .unwrap()
            .0;
        for (_, v) in areas.iter().enumerate().filter(|(idx, _)| *idx != index) {
            cnt += v.len() as u64;
            v.iter().copied().for_each(|p| unsafe {
                *ct.uget_mut(p) = BLACK;
                fg.remove(&p);
            });
        }
        let contours = fg
            .iter()
            .cloned()
            .flat_map(|p| ct.n4_positions(p))
            .filter(|p| unsafe { *ct.uget(*p) == BLACK })
            .collect::<HashSet<_>>();
        cnt -= contours.len() as u64;
        cnt
    }

    fn count_ours(mut ct: LiverSlice) -> u64 {
        let mut cnt = 0;
        let mut g_max0: HashSet<Pos> = ct.vis_liver_pixels().into_iter().collect();
        let e_set = g_max0
            .iter()
            .cloned()
            .filter(|p| ct.is_n4_containing(*p, BLACK))
            .collect::<Vec<_>>();
        unsafe {
            ct.fill_batch(e_set.as_slice(), GRAY);
            for pos in e_set {
                if !ct.is_n4_containing(pos, WHITE) {
                    *ct.uget_mut(pos) = BLACK;
                    cnt += 1;
                    g_max0.remove(&pos);
                }
            }
            let mut s_g01 =
                ct.area_group_from_local_immut(g_max0.iter().cloned(), |u| u == WHITE || u == GRAY);
            if s_g01.is_empty() {
                return cnt;
            }
            let index = s_g01
                .iter()
                .enumerate()
                .max_by_key(|v| v.1.len())
                .unwrap()
                .0;
            for (_, v) in s_g01.iter().enumerate().filter(|(idx, _)| *idx != index) {
                cnt += v.len() as u64;
                v.iter().copied().for_each(|p| *ct.uget_mut(p) = BLACK);
            }
            let s_g01 = mem::take(&mut s_g01[index]);
            let s_g0_1 = ct.area_group_from_local_immut(s_g01.iter().cloned(), |u| u == WHITE);
            ct.non_max_filling(s_g0_1, GRAY);
            for pos in s_g01.iter().cloned() {
                if *ct.uget(pos) == GRAY && !ct.is_n4_containing(pos, WHITE) {
                    *ct.uget_mut(pos) = BLACK;
                    cnt += 1;
                }
            }
        }
        cnt
    }
}
