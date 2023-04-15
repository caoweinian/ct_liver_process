use crate::subcmd_impls::{AREA, WIDTH};
use clap::Args;
use image::GrayImage;
use label::prelude::{LiverSlice, TRAINING_SET_LEN, VIS_BACKGROUND, VIS_LIVER};
use ndarray::Array3;
use std::fs;
use std::mem;
use std::path::PathBuf;

#[derive(Args, Debug)]
pub struct Rnpy2unique {
    /// 数据集根目录。
    #[arg(long = "base-dir", short = 'D')]
    base_dir: PathBuf,
}

impl Rnpy2unique {
    pub fn run(&mut self) {
        // [label_out/raw_npy] -> [label_out/{unique_npy, visual2d_unique}]
        self.base_dir
            .extend(["medical_liver", "LiTS", "lits_train", "label_out"]);

        let mut uniq_npy_dir = self.base_dir.clone();
        uniq_npy_dir.push("unique_npy");
        fs::create_dir_all(uniq_npy_dir.as_path()).unwrap();

        let mut v2d_unique = self.base_dir.clone();
        v2d_unique.push("visual2d_unique");
        fs::create_dir_all(v2d_unique.as_path()).unwrap();

        self.base_dir.push("raw_npy");
        assert!(self.base_dir.is_dir());
        Program::new(mem::take(&mut self.base_dir), uniq_npy_dir, v2d_unique).run();
    }
}

struct Program {
    base_dir: PathBuf,
    uniq_npy_dir: PathBuf,
    v2d_unique_dir: PathBuf,
    seq: usize,
}

impl Program {
    #[inline]
    pub fn new(base_dir: PathBuf, uniq_npy_dir: PathBuf, v2d_unique_dir: PathBuf) -> Self {
        Self {
            base_dir,
            uniq_npy_dir,
            v2d_unique_dir,
            seq: 0,
        }
    }

    pub fn run(&mut self) {
        while self.seq != TRAINING_SET_LEN {
            println!("处理目录`{}`", self.seq);
            self.run_seq();
            self.seq += 1;
        }
    }

    fn run_seq(&mut self) {
        let filename = format!("{}.npy", self.seq);
        self.base_dir.push(filename.as_str());
        self.uniq_npy_dir.push(filename.as_str());

        let liver_3d: Array3<u8> = ndarray_npy::read_npy(self.base_dir.as_path()).unwrap();
        let mut liver_vec: Vec<u8> = liver_3d.into_raw_vec();

        assert_eq!(liver_vec.len() % AREA, 0);
        let slice_len = liver_vec.len() / AREA;
        println!("\t切片个数: {slice_len}\n\t唯一化中...");

        for i in 0..slice_len {
            let offset = i * AREA;
            let mut ct = unsafe {
                LiverSlice::new_unchecked(liver_vec.as_mut_ptr().add(offset), WIDTH, WIDTH)
            };
            if !ct.migrate_pixels() {
                continue;
            }
            let liver_areas = ct.vis_liver_area_group();
            ct.non_max_filling(liver_areas, VIS_BACKGROUND);
            ct.clear_map();
            let bg_areas = ct.vis_background_area_group();
            for area in bg_areas.iter() {
                if ct.all_within(area.as_slice()) {
                    unsafe {
                        ct.fill_batch(area.as_slice(), VIS_LIVER);
                    }
                }
            }
        }
        println!("\t写入npy中...");
        let array =
            Array3::<u8>::from_shape_vec((slice_len, WIDTH, WIDTH), liver_vec.clone()).unwrap();
        ndarray_npy::write_npy(self.uniq_npy_dir.as_path(), &array).unwrap();
        self.uniq_npy_dir.pop();
        self.base_dir.pop();

        self.v2d_unique_dir.push(format!("{}", self.seq));
        fs::create_dir_all(self.v2d_unique_dir.as_path()).unwrap();
        println!("\t写入png中...");

        for i in 0..slice_len {
            let offset = i * AREA;
            let buf = unsafe {
                std::slice::from_raw_parts(liver_vec.as_mut_ptr().add(offset), AREA).into()
            };
            let png = GrayImage::from_vec(WIDTH as u32, WIDTH as u32, buf).unwrap();
            self.v2d_unique_dir.push(format!("{i}.png"));
            png.save(self.v2d_unique_dir.as_path()).unwrap();
            self.v2d_unique_dir.pop();
        }

        self.v2d_unique_dir.pop();
    }
}
