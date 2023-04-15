use super::{AREA, WIDTH};
use clap::Args;
use label::prelude::{AccTimer, LiverSlice, TRAINING_SET_LEN, VIS_BACKGROUND, VIS_LIVER};
use ndarray::Array3;
use std::fs;
use std::path::PathBuf;

#[derive(Args, Debug)]
pub struct Rnpy2unpy {
    /// 数据集根目录。
    #[arg(long = "base-dir", short = 'D')]
    base_dir: PathBuf,
    /// 是否启用性能测试。
    #[arg(long, short = 'B')]
    bench: bool,
}

impl Rnpy2unpy {
    pub fn run(&mut self) {
        // [label_out/raw_npy] -> [label_out/unified_npy]
        self.base_dir
            .extend(["medical_liver", "LiTS", "lits_train", "label_out"]);

        let mut raw_npy_dir = self.base_dir.clone();
        raw_npy_dir.push("raw_npy");
        assert!(raw_npy_dir.is_dir());

        let mut unified_npy_dir = self.base_dir.clone();
        unified_npy_dir.push("unified_npy");
        fs::create_dir_all(unified_npy_dir.as_path()).unwrap();

        if self.bench {
            Bench::new(raw_npy_dir, unified_npy_dir).run();
        } else {
            NoBench::new(raw_npy_dir, unified_npy_dir).run();
        }
    }
}

struct Bench {
    raw_npy_dir: PathBuf,
    uni_npy_dir: PathBuf,
    seq: usize,
    all_img: usize,         // 总共图像个数
    normal_img: usize,      // 非背景图像个数
    timer_cpu: AccTimer,    // 记录图像处理总用时
    timer_normal: AccTimer, // 记录非背景图像处理总用时
}

impl Bench {
    #[inline]
    pub fn new(raw_npy_dir: PathBuf, uni_npy_dir: PathBuf) -> Self {
        Self {
            raw_npy_dir,
            uni_npy_dir,
            seq: 0,
            all_img: 0,
            normal_img: 0,
            timer_cpu: AccTimer::new(),
            timer_normal: AccTimer::new(),
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
        let cpu_ms = self.timer_cpu.get_total_ms();
        let normal_ms = self.timer_normal.get_total_ms();
        let trivial_ms = cpu_ms - normal_ms;
        let avg_cpu = cpu_ms as f64 / self.all_img as f64;
        let avg_normal = normal_ms as f64 / self.normal_img as f64;
        let avg_trivial = trivial_ms as f64 / (self.all_img - self.normal_img) as f64;

        println!("----------------------------------------------------------");
        println!("全部完成。\n\n切片总数: {}", self.all_img);
        println!("背景切片总数: {}", self.all_img - self.normal_img);
        println!("非背景切片总数: {}", self.normal_img);
        println!("总CPU时间(自加载后始，包括系统调度时间): {cpu_ms} ms");
        println!(
            "处理背景切片总CPU时间(自加载后始，包括系统调度时间): {} ms",
            cpu_ms - normal_ms
        );
        println!("处理非背景切片总CPU时间(自加载后始，包括系统调度时间): {normal_ms} ms");
        println!("每张切片平均处理时长: {avg_cpu} ms每张");
        println!("背景切片平均处理时长: {avg_trivial} ms每张");
        println!("非背景切片平均处理时长: {avg_normal} ms每张");
        println!("----------------------------------------------------------");
    }

    fn run_seq(&mut self) {
        let filename = format!("{}.npy", self.seq);
        self.raw_npy_dir.push(filename.as_str());
        self.uni_npy_dir.push(filename.as_str());

        let liver_3d: Array3<u8> = ndarray_npy::read_npy(self.raw_npy_dir.as_path()).unwrap();
        let mut liver_vec: Vec<u8> = liver_3d.into_raw_vec();

        assert_eq!(liver_vec.len() % AREA, 0);
        let slice_len = liver_vec.len() / AREA;
        println!("\t切片个数: {slice_len}");
        self.all_img += slice_len;

        for i in 0..slice_len {
            let offset = i * AREA;
            let mut ct = unsafe {
                LiverSlice::new_unchecked(liver_vec.as_mut_ptr().add(offset), WIDTH, WIDTH)
            };

            self.timer_cpu.start();

            // 肿瘤部分被视为肝脏区域
            let modified = ct.migrate_pixels();

            if modified {
                self.normal_img += 1;
                self.timer_normal.start();

                // 填充所有非最大肝脏区域（填为背景），并获得最大肝脏区域
                let liver_areas = ct.vis_liver_area_group();
                let biggest_liver = ct.non_max_filling(liver_areas, VIS_BACKGROUND).unwrap();
                ct.clear_map();

                // 找到所有背景区域，将封闭背景区域填充为肝脏区域
                let bg_areas = ct.vis_background_area_group();
                for area in bg_areas.iter() {
                    if ct.all_within(area.as_slice()) {
                        unsafe {
                            ct.fill_batch(area.as_slice(), VIS_LIVER);
                        }
                    }
                }

                // 填充肝脏边缘（候选），并细化
                ct.draw_liver_boundary(biggest_liver.as_slice());
                ct.clear_map();

                // 找到所有独立区域（边缘+内点），只留最大的，其余填充为背景
                let hole_areas = ct.vis_liver_boundary_area_group();
                ct.non_max_filling(hole_areas, VIS_BACKGROUND);

                // 找到所有独立区域（内点）。若只有一个则结束，否则将非最大区域填充为背景，并整图细化，循环
                ct.unique();

                self.timer_normal.elapsed();
            }
            self.timer_cpu.elapsed();
        }

        let array = Array3::<u8>::from_shape_vec((slice_len, WIDTH, WIDTH), liver_vec).unwrap();
        ndarray_npy::write_npy(self.uni_npy_dir.as_path(), &array).unwrap();

        self.uni_npy_dir.pop();
        self.raw_npy_dir.pop();
    }
}

struct NoBench {
    raw_npy_dir: PathBuf,
    uni_npy_dir: PathBuf,
    seq: usize,
}

impl NoBench {
    #[inline]
    pub fn new(raw_npy_dir: PathBuf, uni_npy_dir: PathBuf) -> Self {
        Self {
            raw_npy_dir,
            uni_npy_dir,
            seq: 0,
        }
    }

    pub fn run(&mut self) {
        while self.seq != TRAINING_SET_LEN {
            println!("处理目录`{}`...", self.seq);
            self.run_seq();
            self.seq += 1;
        }
    }

    fn run_seq(&mut self) {
        let filename = format!("{}.npy", self.seq);
        self.raw_npy_dir.push(filename.as_str());
        self.uni_npy_dir.push(filename.as_str());

        let liver_3d: Array3<u8> = ndarray_npy::read_npy(self.raw_npy_dir.as_path()).unwrap();
        let mut liver_vec: Vec<u8> = liver_3d.into_raw_vec();

        assert_eq!(liver_vec.len() % AREA, 0);
        let slice_len = liver_vec.len() / AREA;
        println!("\t切片个数: {slice_len}");

        for i in 0..slice_len {
            let offset = i * AREA;
            let mut ct = unsafe {
                LiverSlice::new_unchecked(liver_vec.as_mut_ptr().add(offset), WIDTH, WIDTH)
            };

            // 肿瘤部分被视为肝脏区域
            let modified = ct.migrate_pixels();

            if modified {
                // 填充所有非最大肝脏区域（填为背景），并获得最大肝脏区域
                let liver_areas = ct.vis_liver_area_group();
                let biggest_liver = ct.non_max_filling(liver_areas, VIS_BACKGROUND).unwrap();
                ct.clear_map();

                // 找到所有背景区域，将封闭背景区域填充为肝脏区域
                let bg_areas = ct.vis_background_area_group();
                for area in bg_areas.iter() {
                    if ct.all_within(area.as_slice()) {
                        unsafe {
                            ct.fill_batch(area.as_slice(), VIS_LIVER);
                        }
                    }
                }

                // 填充肝脏边缘（候选），并细化
                ct.draw_liver_boundary(biggest_liver.as_slice());
                ct.clear_map();

                // 找到所有独立区域（边缘+内点），只留最大的，其余填充为背景
                let hole_areas = ct.vis_liver_boundary_area_group();
                ct.non_max_filling(hole_areas, VIS_BACKGROUND);

                // 整图唯一化。
                ct.unique();
            }
        }

        let array = Array3::<u8>::from_shape_vec((slice_len, WIDTH, WIDTH), liver_vec).unwrap();
        ndarray_npy::write_npy(self.uni_npy_dir.as_path(), &array).unwrap();

        self.uni_npy_dir.pop();
        self.raw_npy_dir.pop();
    }
}
