use crate::subcmd_impls::{AREA, WIDTH};
use clap::Args;
use image::GrayImage;
use label::prelude::{AccTimer, LiverSlice, Pos, LITS_BACKGROUND, VIS_BACKGROUND};
use label::prep::improc::consts::{BLACK, GRAY, WHITE};
use label::prep::TRAINING_SET_LEN;
use ndarray::{Array3, ArrayView3, Dim, Shape};
use opencv::core::{no_array, Mat, Point, Scalar, CV_8U};
use opencv::imgproc::{canny, draw_contours, find_contours, CHAIN_APPROX_SIMPLE, RETR_EXTERNAL};
use opencv::prelude::MatTraitConstManual;
use opencv::types::VectorOfVectorOfPoint;
use std::collections::HashSet;
use std::ffi::c_void;
use std::fs;
use std::mem;
use std::path::PathBuf;

cfg_if::cfg_if! {
    if #[cfg(target_os = "macos")] {
        use opencv::imgproc::LINE_8;
    } else if #[cfg(target_os = "linux")] {
        use opencv::core::LINE_8;
    }
}
#[derive(Args, Debug)]
pub struct AlgoBench {
    /// 数据集根目录。
    #[arg(long = "base-dir", short = 'D')]
    base_dir: PathBuf,
    /// 是否将npy文件写到数据集目录中。
    #[arg(long)]
    save_npy: bool,
    /// 是否将可视化结果写到数据集目录中。
    #[arg(long)]
    save_png: bool,
}

impl AlgoBench {
    pub fn run(&mut self) {
        // 注意在每个算法的输出中，将轮廓标记为白色，将背景标记为黑色，将在原图与输出图中均为前景的标记为灰色（内部）。
        // 注意这种颜色配置与之前`rnpy2unpy`过程不一致。
        self.base_dir.extend([
            "medical_liver",
            "LiTS",
            "lits_train",
            "label_out",
            "algo_result",
        ]);
        fs::create_dir_all(self.base_dir.as_path()).unwrap();
        // assert!(self.base_dir.is_dir());
        Program::new(mem::take(&mut self.base_dir), self.save_npy, self.save_png).run();
    }
}

struct Program {
    unique_npy_dir: PathBuf,

    npy_canny_dir: PathBuf,
    npy_suzuki_dir: PathBuf,
    npy_hrvoje_dir: PathBuf,
    npy_ours_dir: PathBuf,

    v2d_canny_dir: PathBuf,
    v2d_suzuki_dir: PathBuf,
    v2d_hrvoje_dir: PathBuf,
    v2d_ours_dir: PathBuf,

    timers: Vec<AccTimer>,

    save_npy: bool,
    save_png: bool,
}

impl Program {
    #[inline]
    pub fn new(base_dir: PathBuf, save_npy: bool, save_png: bool) -> Self {
        let (npy_canny_dir, npy_suzuki_dir, npy_hrvoje_dir, npy_ours_dir) = match save_npy {
            true => {
                let mut npy_canny_dir = base_dir.clone();
                npy_canny_dir.push("npy_algo_canny");
                fs::create_dir_all(npy_canny_dir.as_path()).unwrap();

                let mut npy_suzuki_dir = base_dir.clone();
                npy_suzuki_dir.push("npy_algo_suzuki");
                fs::create_dir_all(npy_suzuki_dir.as_path()).unwrap();

                let mut npy_hrvoje_dir = base_dir.clone();
                npy_hrvoje_dir.push("npy_algo_hrvoje");
                fs::create_dir_all(npy_hrvoje_dir.as_path()).unwrap();

                let mut npy_ours_dir = base_dir.clone();
                npy_ours_dir.push("npy_algo_ours");
                fs::create_dir_all(npy_ours_dir.as_path()).unwrap();

                (npy_canny_dir, npy_suzuki_dir, npy_hrvoje_dir, npy_ours_dir)
            }
            false => (
                PathBuf::new(),
                PathBuf::new(),
                PathBuf::new(),
                PathBuf::new(),
            ),
        };
        let (v2d_canny_dir, v2d_suzuki_dir, v2d_hrvoje_dir, v2d_ours_dir) = match save_png {
            true => {
                let mut v2d_canny_dir = base_dir.clone();
                v2d_canny_dir.push("v2d_algo_canny");
                fs::create_dir_all(v2d_canny_dir.as_path()).unwrap();

                let mut v2d_suzuki_dir = base_dir.clone();
                v2d_suzuki_dir.push("v2d_algo_suzuki");
                fs::create_dir_all(v2d_suzuki_dir.as_path()).unwrap();

                let mut v2d_hrvoje_dir = base_dir.clone();
                v2d_hrvoje_dir.push("v2d_algo_hrvoje");
                fs::create_dir_all(v2d_hrvoje_dir.as_path()).unwrap();

                let mut v2d_ours_dir = base_dir.clone();
                v2d_ours_dir.push("v2d_algo_ours");
                fs::create_dir_all(v2d_ours_dir.as_path()).unwrap();

                (v2d_canny_dir, v2d_suzuki_dir, v2d_hrvoje_dir, v2d_ours_dir)
            }
            false => (
                PathBuf::new(),
                PathBuf::new(),
                PathBuf::new(),
                PathBuf::new(),
            ),
        };
        let mut unique_npy_dir = base_dir;
        unique_npy_dir.push("unique_npy");
        Self {
            unique_npy_dir,
            npy_canny_dir,
            npy_suzuki_dir,
            npy_hrvoje_dir,
            npy_ours_dir,
            v2d_canny_dir,
            v2d_suzuki_dir,
            v2d_hrvoje_dir,
            v2d_ours_dir,
            timers: vec![AccTimer::new(); 4],
            save_npy,
            save_png,
        }
    }

    pub fn run(&mut self) {
        // [label_out/unique_npy] ->? [label_out/{npy, visual2d}_algo_{canny, suzuki, hrvoje, ours}]
        let mut foreground_cnt = 0_u64;
        {
            let algo_result_dirname = self.unique_npy_dir.file_name().unwrap().to_owned();
            self.unique_npy_dir.pop();
            self.unique_npy_dir.pop();
            self.unique_npy_dir.push(algo_result_dirname);
        }

        for seq in 0..TRAINING_SET_LEN {
            let filename = format!("{seq}.npy");
            println!("处理目录`{seq}`...");

            self.unique_npy_dir.push(filename.as_str());
            let liver_3d: Array3<u8> =
                ndarray_npy::read_npy(self.unique_npy_dir.as_path()).unwrap();
            self.unique_npy_dir.pop();

            let mut liver_vec: Vec<u8> = liver_3d.into_raw_vec();
            let slice_len = liver_vec.len() / AREA;
            for z in 0..slice_len {
                let offset = z * AREA;
                let ct = unsafe {
                    LiverSlice::new_unchecked(liver_vec.as_mut_ptr().add(offset), WIDTH, WIDTH)
                };
                if ct.count(VIS_BACKGROUND) == AREA {
                    foreground_cnt += 1;
                }
            }
            let s = liver_vec.as_slice();
            self.run_canny(s, seq);
            self.run_suzuki(s, seq);
            self.run_hrvoje(s, seq);
            self.run_ours(s, seq);
        }
        self.summary(foreground_cnt);
    }

    fn run_canny(&mut self, ct: &[u8], dir_seq: usize) {
        let needs_save = self.save_npy || self.save_png;
        let slice_len = ct.len() / AREA;
        let mut array: Vec<u8> = if needs_save {
            Vec::with_capacity(ct.len())
        } else {
            vec![]
        };
        for png_seq in 0..slice_len {
            let offset = png_seq * AREA;
            let mut orig_img: Vec<u8> = ct[offset..offset + AREA].to_vec();
            if orig_img.iter().all(|&u| u == LITS_BACKGROUND) {
                if needs_save {
                    array.extend(orig_img.into_iter());
                }
                continue;
            }
            self.timers[0].start();
            // `template`指向`orig_img`，其中没有内存分配。
            // 为公平起见，需要计算一次`template`复制的时间，因此调用`clone()`。
            let template = unsafe {
                Mat::new_rows_cols_with_data(
                    WIDTH as i32,
                    WIDTH as i32,
                    CV_8U,
                    orig_img.as_mut_ptr() as *mut c_void,
                    WIDTH,
                )
            }
            .unwrap()
            .clone();
            let mut contours =
                Mat::new_size_with_default(template.size().unwrap(), CV_8U, Scalar::from(255))
                    .unwrap();
            canny(&template, &mut contours, 50.0, 150.0, 3, false).unwrap();
            self.timers[0].elapsed();
            if needs_save {
                array.extend(
                    contours
                        .to_vec_2d()
                        .unwrap()
                        .iter()
                        .flat_map(|v: &Vec<u8>| v.iter())
                        .cloned(),
                );
            }
        }
        if self.save_npy {
            let pat = &mut self.npy_canny_dir as *mut PathBuf;
            let shape = Shape::from(Dim([slice_len, WIDTH, WIDTH]));
            let data_ptr = unsafe { ArrayView3::<u8>::from_shape_ptr(shape, array.as_mut_ptr()) };
            self.save_npy(pat, data_ptr, dir_seq);
        }
        if self.save_png {
            let pat = &mut self.v2d_canny_dir as *mut PathBuf;
            self.save_pngs(pat, array.as_slice(), dir_seq);
        }
    }

    fn run_suzuki(&mut self, ct: &[u8], dir_seq: usize) {
        let needs_save = self.save_npy || self.save_png;
        let slice_len = ct.len() / AREA;
        let mut array: Vec<u8> = if needs_save {
            Vec::with_capacity(ct.len())
        } else {
            vec![]
        };
        for png_seq in 0..slice_len {
            let offset = png_seq * AREA;
            let mut orig_img: Vec<u8> = ct[offset..offset + AREA].to_vec();
            if orig_img.iter().all(|&u| u == LITS_BACKGROUND) {
                if needs_save {
                    array.extend(orig_img.into_iter());
                }
                continue;
            }
            self.timers[1].start();
            // `template`指向`orig_img`，其中没有内存分配。
            // 为公平起见，需要计算一次`template`复制的时间，因此调用`clone()`。
            cfg_if::cfg_if! {
                if #[cfg(target_os = "macos")] {
                    let template = unsafe {
                        Mat::new_rows_cols_with_data(
                            WIDTH as i32,
                            WIDTH as i32,
                            CV_8U,
                            orig_img.as_mut_ptr() as *mut c_void,
                            WIDTH,
                        )
                    }
                    .unwrap()
                    .clone();
                    let mut contours = VectorOfVectorOfPoint::with_capacity(1);
                    find_contours(
                        &template,
                        &mut contours,
                        RETR_EXTERNAL,
                        CHAIN_APPROX_SIMPLE,
                        Point::new(0, 0),
                    )
                    .unwrap();
                } else {
                    // for linux only; windows not supported
                    let mut template = unsafe {
                        Mat::new_rows_cols_with_data(
                            WIDTH as i32,
                            WIDTH as i32,
                            CV_8U,
                            orig_img.as_mut_ptr() as *mut c_void,
                            WIDTH,
                        )
                    }
                    .unwrap()
                    .clone();
                    let mut contours = VectorOfVectorOfPoint::with_capacity(1);
                    find_contours(
                        &mut template,
                        &mut contours,
                        RETR_EXTERNAL,
                        CHAIN_APPROX_SIMPLE,
                        Point::new(0, 0),
                    )
                    .unwrap();
                }
            }

            let mut contours_img =
                Mat::new_size_with_default(template.size().unwrap(), CV_8U, Scalar::from(0))
                    .unwrap();
            draw_contours(
                &mut contours_img,
                &contours,
                -1,
                Scalar::from(255),
                1,
                LINE_8,
                &no_array(),
                i32::MAX,
                Point::default(),
            )
            .unwrap();
            self.timers[1].elapsed();
            if needs_save {
                array.extend(
                    contours_img
                        .to_vec_2d()
                        .unwrap()
                        .iter()
                        .flat_map(|v: &Vec<u8>| v.iter())
                        .cloned(),
                );
            }
        }
        if self.save_npy {
            let pat = &mut self.npy_suzuki_dir as *mut PathBuf;
            let shape = Shape::from(Dim([slice_len, WIDTH, WIDTH]));
            let data_ptr = unsafe { ArrayView3::<u8>::from_shape_ptr(shape, array.as_mut_ptr()) };
            self.save_npy(pat, data_ptr, dir_seq);
        }
        if self.save_png {
            let pat = &mut self.v2d_suzuki_dir as *mut PathBuf;
            self.save_pngs(pat, array.as_slice(), dir_seq);
        }
    }

    fn run_hrvoje(&mut self, ct: &[u8], dir_seq: usize) {
        let needs_save = self.save_npy || self.save_png;
        let slice_len = ct.len() / AREA;
        let mut owned_ct = ct.to_vec();
        let mut array: Vec<u8> = if needs_save {
            Vec::with_capacity(ct.len())
        } else {
            vec![]
        };
        for png_seq in 0..slice_len {
            let offset = png_seq * AREA;
            let ct_slice = unsafe {
                LiverSlice::new_unchecked(owned_ct.as_mut_ptr().add(offset), WIDTH, WIDTH)
            };
            if ct_slice.is_all_vis_background() {
                if needs_save {
                    array.extend(owned_ct[offset..offset + AREA].iter().cloned());
                }
                continue;
            }
            self.timers[2].start();
            let mut algo = HrvojeAlgoOptimized(ct_slice);
            algo.run();
            self.timers[2].elapsed();
            if needs_save {
                array.extend(owned_ct[offset..offset + AREA].iter().cloned());
            }
        }
        if self.save_npy {
            let pat = &mut self.npy_hrvoje_dir as *mut PathBuf;
            let shape = Shape::from(Dim([slice_len, WIDTH, WIDTH]));
            let data_ptr = unsafe { ArrayView3::<u8>::from_shape_ptr(shape, array.as_mut_ptr()) };
            self.save_npy(pat, data_ptr, dir_seq);
        }
        if self.save_png {
            let pat = &mut self.v2d_hrvoje_dir as *mut PathBuf;
            self.save_pngs(pat, array.as_slice(), dir_seq);
        }
    }

    fn run_ours(&mut self, ct: &[u8], dir_seq: usize) {
        let needs_save = self.save_npy || self.save_png;
        let slice_len = ct.len() / AREA;
        let mut owned_ct = ct.to_vec();
        let mut array: Vec<u8> = if needs_save {
            Vec::with_capacity(ct.len())
        } else {
            vec![]
        };
        for png_seq in 0..slice_len {
            let offset = png_seq * AREA;
            let ct_slice = unsafe {
                LiverSlice::new_unchecked(owned_ct.as_mut_ptr().add(offset), WIDTH, WIDTH)
            };
            if ct_slice.is_all_vis_background() {
                if needs_save {
                    array.extend(owned_ct[offset..offset + AREA].iter().cloned());
                }
                continue;
            }
            self.timers[3].start();
            let mut algo = OursAlgoOptimized(ct_slice);
            algo.run();
            self.timers[3].elapsed();
            if needs_save {
                array.extend(owned_ct[offset..offset + AREA].iter().cloned());
            }
        }
        if self.save_npy {
            let pat = &mut self.npy_ours_dir as *mut PathBuf;
            let shape = Shape::from(Dim([slice_len, WIDTH, WIDTH]));
            let data_ptr = unsafe { ArrayView3::<u8>::from_shape_ptr(shape, array.as_mut_ptr()) };
            self.save_npy(pat, data_ptr, dir_seq);
        }
        if self.save_png {
            let pat = &mut self.v2d_ours_dir as *mut PathBuf;
            self.save_pngs(pat, array.as_slice(), dir_seq);
        }
    }

    fn save_npy(&mut self, base_path: *mut PathBuf, array: ArrayView3<u8>, dir_seq: usize) {
        let base_path = unsafe { &mut *base_path };
        let filename = format!("{dir_seq}.npy");
        base_path.push(filename.as_str());
        ndarray_npy::write_npy(base_path.as_path(), &array).unwrap();
        base_path.pop();
    }

    fn save_pngs(&mut self, base_path: *mut PathBuf, array: &[u8], dir_seq: usize) {
        let base_path = unsafe { &mut *base_path };
        let dir_name = format!("{dir_seq}");
        base_path.push(dir_name.as_str());
        fs::create_dir_all(base_path.as_path()).unwrap();

        let z_len = array.len() / AREA;
        for z in 0..z_len {
            let offset = z * AREA;
            let png_vec = array[offset..offset + AREA].to_vec();
            let png = GrayImage::from_vec(WIDTH as u32, WIDTH as u32, png_vec).unwrap();
            let png_filename = format!("{z}.png");
            base_path.push(png_filename.as_str());
            png.save(base_path.as_path()).unwrap();
            base_path.pop();
        }
        base_path.pop();
    }

    fn summary(&self, foreground_count: u64) {
        let foreground_count = foreground_count as f64;
        let mut mss = self.timers.iter().map(|t| t.get_total_ms() as f64);
        let canny_tot_ms = mss.next().unwrap();
        let suzuki_tot_ms = mss.next().unwrap();
        let hrvoje_tot_ms = mss.next().unwrap();
        let ours_tot_ms = mss.next().unwrap();
        assert!(mss.next().is_none());
        let (canny_avg_ms, suzuki_avg_ms, hrvoje_avg_ms, ours_avg_ms) = (
            canny_tot_ms / foreground_count,
            suzuki_tot_ms / foreground_count,
            hrvoje_tot_ms / foreground_count,
            ours_tot_ms / foreground_count,
        );
        println!("----------------------------------------------------------");
        Self::summary_item(canny_tot_ms, canny_avg_ms, "canny");
        Self::summary_item(suzuki_tot_ms, suzuki_avg_ms, "suzuki");
        Self::summary_item(hrvoje_tot_ms, hrvoje_avg_ms, "hrvoje");
        Self::summary_item(ours_tot_ms, ours_avg_ms, "ours");
        println!("----------------------------------------------------------");
    }

    #[inline]
    fn summary_item(tot: f64, avg: f64, algo_name: &str) {
        println!("{algo_name}算法\n\t总耗时: {tot} ms\n\t平均耗时: {avg} ms");
    }
}

struct HrvojeAlgoOptimized(LiverSlice);

impl HrvojeAlgoOptimized {
    pub fn run(&mut self) {
        // 算法流程（仅生成白色轮廓，其它像素为黑色）：
        // 1. 腐蚀一层边缘（白色改为黑色背景）
        // 2. 循环腐蚀前景（白色改为黑色背景），直到收敛
        // 3. 扩张一层前景（扩张为白色前景）
        // 4. 将内部白色前景去除
        // --------------------------------------------------------
        // 1.
        let mut fg: HashSet<Pos> = self.0.vis_liver_pixels().into_iter().collect();
        let boundaries = fg
            .iter()
            .cloned()
            .filter(|p| self.0.is_n4_containing(*p, BLACK))
            .collect::<Vec<_>>();
        for pos in boundaries.iter().cloned() {
            unsafe {
                *self.0.uget_mut(pos) = BLACK;
            }
            fg.remove(&pos);
        }
        // 2.
        loop {
            let bd = fg
                .iter()
                .cloned()
                .filter(|p| self.0.n4_count(*p, WHITE) < 2)
                .collect::<Vec<_>>();
            if bd.is_empty() {
                break;
            }
            for pos in bd.iter().cloned() {
                unsafe {
                    *self.0.uget_mut(pos) = BLACK;
                }
                fg.remove(&pos);
            }
        }
        let areas = unsafe {
            self.0
                .area_group_from_local_immut(fg.iter().cloned(), |u| u == WHITE)
        };
        if areas.is_empty() {
            return;
        }
        let index = areas
            .iter()
            .enumerate()
            .max_by_key(|v| v.1.len())
            .unwrap()
            .0;
        for (_, v) in areas.iter().enumerate().filter(|(idx, _)| *idx != index) {
            v.iter().copied().for_each(|p| unsafe {
                *self.0.uget_mut(p) = BLACK;
                fg.remove(&p);
            });
        }
        // 3 & 4
        let contours = fg
            .iter()
            .cloned()
            .flat_map(|p| self.0.n4_positions(p))
            .filter(|p| unsafe { *self.0.uget(*p) == BLACK })
            .collect::<HashSet<_>>();
        unsafe {
            self.0.fill_batch_from(fg.iter().copied(), BLACK);
            self.0.fill_batch_from(contours.iter().copied(), WHITE);
        }
    }
}

struct OursAlgoOptimized(LiverSlice);

impl OursAlgoOptimized {
    pub fn run(&mut self) {
        // 算法数据流（这里的Step与论文中的步骤不一致；以这里的编号为准）：
        // Step4，记录Gmax0集合和E集合。
        // Step5，Gmax0不变，E不变。其中Gmax0中一些从0变1；E中所有从0变1。
        // Step6，把E中变为2的从Gmax0 <del>和E中</del> 去除。
        // Step7，从Gmax0中提取Sg01，只留面积最大的那个，其它变2。
        // Step8-9，从Sg01中提取Sg0^1, 令K=Sg01-Sg0^1变1。
        // Step10，将K中符合要求的1变2。
        // --------------------------------------------------------
        // Step4.
        let mut g_max0: HashSet<Pos> = self.0.vis_liver_pixels().into_iter().collect();
        let e_set = g_max0
            .iter()
            .cloned()
            .filter(|p| self.0.is_n4_containing(*p, BLACK))
            .collect::<Vec<_>>();
        unsafe {
            // Step5. & Step6.
            self.0.fill_batch(e_set.as_slice(), GRAY);
            for pos in e_set {
                if !self.0.is_n4_containing(pos, WHITE) {
                    *self.0.uget_mut(pos) = BLACK;
                    g_max0.remove(&pos);
                }
            }
            // Step7.
            let s_g01 = self
                .0
                .area_group_from_local_immut(g_max0.iter().cloned(), |u| u == WHITE || u == GRAY);
            let s_g01 = match self.0.non_max_filling(s_g01, BLACK) {
                None => return,
                Some(a) => a,
            };
            // 此时`s_g01`是单个极大0-1区域。
            // Step8. & Step9. & Step10.
            let s_g0_1 = self
                .0
                .area_group_from_local_immut(s_g01.iter().cloned(), |u| u == WHITE);
            self.0.non_max_filling(s_g0_1, GRAY);
            for pos in s_g01.iter().cloned() {
                if *self.0.uget(pos) == GRAY && !self.0.is_n4_containing(pos, WHITE) {
                    *self.0.uget_mut(pos) = BLACK;
                }
            }
            // 后处理
            for pos in s_g01.iter().cloned() {
                match *self.0.uget(pos) {
                    WHITE => *self.0.uget_mut(pos) = BLACK,
                    GRAY => *self.0.uget_mut(pos) = WHITE,
                    _ => (),
                }
            }
        }
    }
}
