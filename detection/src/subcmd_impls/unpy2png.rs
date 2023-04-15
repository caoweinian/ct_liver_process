use super::utils::rgb;
use super::{AREA, WIDTH};
use clap::Args;
use image::{GrayImage, Rgb, RgbImage};
use label::prelude::{TRAINING_SET_LEN, VIS_BACKGROUND, VIS_BOUNDARY, VIS_LIVER};
use ndarray::Array3;
use std::fs;
use std::path::PathBuf;

#[derive(Args, Debug)]
pub struct Unpy2png {
    /// 数据集根目录。
    #[arg(long = "base-dir", short = 'D')]
    base_dir: PathBuf,
    /// 使用RGB彩色图像输出，并指定颜色（默认灰色灰度图）。
    #[arg(long, short, value_parser = super::utils::color_valid_rgb_hex)]
    color: Option<Rgb<u8>>,
    #[arg(long = "output-dir")]
    output_dir: Option<PathBuf>,
}

impl Unpy2png {
    pub fn run(&mut self) {
        // [label_out/unified_npy] -> [label_out/visual2d_unified]
        self.base_dir
            .extend(["medical_liver", "LiTS", "lits_train", "label_out"]);

        let mut uni_npy_dir = self.base_dir.clone();
        uni_npy_dir.push("unified_npy");
        assert!(uni_npy_dir.is_dir());

        let mut v2d_uni_dir = self.base_dir.clone();
        if let Some(ref p) = self.output_dir {
            v2d_uni_dir.push(p.as_path());
        } else {
            v2d_uni_dir.push("visual2d_unified");
        }
        fs::create_dir_all(v2d_uni_dir.as_path()).unwrap();

        Program::new(uni_npy_dir, v2d_uni_dir, self.color).run();
    }
}

struct Program {
    uni_npy_dir: PathBuf,
    v2d_uni_dir: PathBuf,
    seq: usize,
    color: Option<Rgb<u8>>,
}

impl Program {
    #[inline]
    pub fn new(uni_npy_dir: PathBuf, v2d_uni_dir: PathBuf, color: Option<Rgb<u8>>) -> Self {
        Self {
            uni_npy_dir,
            v2d_uni_dir,
            seq: 0,
            color,
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
        self.uni_npy_dir.push(filename.as_str());
        self.v2d_uni_dir.push(format!("{}", self.seq));
        fs::create_dir_all(self.v2d_uni_dir.as_path()).unwrap();

        let liver_3d: Array3<u8> = ndarray_npy::read_npy(self.uni_npy_dir.as_path()).unwrap();
        let mut liver_vec: Vec<u8> = liver_3d.into_raw_vec();

        let slice_len = liver_vec.len() / AREA;
        println!("\t切片个数: {slice_len}");

        for i in 0..slice_len {
            let offset = i * AREA;

            if let Some(color) = self.color {
                let base_ptr = unsafe { liver_vec.as_ptr().add(offset) };
                let mut png = RgbImage::new(WIDTH as u32, WIDTH as u32);
                let mut inner_offset = 0;
                for h in 0..(WIDTH as u32) {
                    for w in 0..(WIDTH as u32) {
                        let pixel = unsafe { base_ptr.add(inner_offset).read() };
                        *png.get_pixel_mut(w, h) = match pixel {
                            VIS_LIVER => rgb::white(),
                            VIS_BOUNDARY => color,
                            VIS_BACKGROUND => rgb::black(),
                            _ => unreachable!(),
                        };
                        inner_offset += 1;
                    }
                }
                self.v2d_uni_dir.push(format!("{i}.png"));
                png.save(self.v2d_uni_dir.as_path()).unwrap();
            } else {
                let buf: Vec<u8> = unsafe {
                    std::slice::from_raw_parts(liver_vec.as_mut_ptr().add(offset), AREA).into()
                };
                let png = GrayImage::from_vec(WIDTH as u32, WIDTH as u32, buf).unwrap();
                self.v2d_uni_dir.push(format!("{i}.png"));
                png.save(self.v2d_uni_dir.as_path()).unwrap();
            }
            self.v2d_uni_dir.pop();
        }

        self.v2d_uni_dir.pop();
        self.uni_npy_dir.pop();
    }
}
