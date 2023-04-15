use super::{AREA, WIDTH};
use clap::Args;
use image::GrayImage;
use label::prelude::{LITS_LIVER, LITS_TUMOR, TRAINING_SET_LEN, VIS_LIVER};
use ndarray::Array3;
use std::fs;
use std::path::PathBuf;

#[derive(Args, Debug)]
pub struct Rnpy2png {
    /// 数据集根目录。
    #[arg(long = "base-dir", short = 'D')]
    base_dir: PathBuf,
}

impl Rnpy2png {
    pub fn run(&mut self) {
        // [label_out/raw_npy] -> [label_out/visual2d_raw]
        self.base_dir
            .extend(["medical_liver", "LiTS", "lits_train", "label_out"]);

        let mut raw_npy_dir = self.base_dir.clone();
        raw_npy_dir.push("raw_npy");
        assert!(raw_npy_dir.is_dir());

        let mut v2d_raw_dir = self.base_dir.clone();
        v2d_raw_dir.push("visual2d_raw");
        fs::create_dir_all(v2d_raw_dir.as_path()).unwrap();

        Program::new(raw_npy_dir, v2d_raw_dir).run();
    }
}

struct Program {
    raw_npy_dir: PathBuf,
    v2d_raw_dir: PathBuf,
    seq: usize,
}

impl Program {
    #[inline]
    pub fn new(raw_npy_dir: PathBuf, v2d_raw_dir: PathBuf) -> Self {
        Self {
            raw_npy_dir,
            v2d_raw_dir,
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
        self.v2d_raw_dir.push(format!("{}", self.seq));
        fs::create_dir_all(self.v2d_raw_dir.as_path()).unwrap();

        let liver_3d: Array3<u8> = ndarray_npy::read_npy(self.raw_npy_dir.as_path()).unwrap();
        let mut liver_vec: Vec<u8> = liver_3d.into_raw_vec();

        let slice_len = liver_vec.len() / AREA;
        println!("\t切片个数: {slice_len}");

        for i in 0..slice_len {
            let offset = i * AREA;
            let mut buf: Vec<u8> = unsafe {
                std::slice::from_raw_parts(liver_vec.as_mut_ptr().add(offset), AREA).into()
            };
            for pixel in buf.iter_mut() {
                if *pixel == LITS_TUMOR || *pixel == LITS_LIVER {
                    *pixel = VIS_LIVER;
                }
            }
            let png = GrayImage::from_vec(WIDTH as u32, WIDTH as u32, buf).unwrap();
            self.v2d_raw_dir.push(format!("{i}.png"));
            png.save(self.v2d_raw_dir.as_path()).unwrap();
            self.v2d_raw_dir.pop();
        }

        self.v2d_raw_dir.pop();
        self.raw_npy_dir.pop();
    }
}
