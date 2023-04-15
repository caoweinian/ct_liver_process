use super::{AREA, WIDTH};
use clap::Args;
use image::GrayImage;
use label::prelude::TRAINING_SET_LEN;
use ndarray::Array3;
use std::fs;
use std::path::PathBuf;

#[derive(Args, Debug)]
pub struct CtWindow {
    /// 数据集根目录。
    #[arg(long = "base-dir", short = 'D')]
    base_dir: PathBuf,
    /// CT窗位。
    #[arg(long = "centre", value_parser = ct_centre_legal_range)]
    ct_centre: f32,
    /// CT窗宽。
    #[arg(long = "width", value_parser = ct_width_legal_range)]
    ct_width: f32,
}

fn ct_centre_legal_range(s: &str) -> Result<f32, String> {
    let centre: f32 = s
        .parse()
        .map_err(|_| format!("`{s}` is not a legal CT centre value"))?;
    if centre.is_nan() || centre.is_infinite() {
        return Err(format!("`{s}` is not a legal CT centre value"));
    }
    if centre.abs() >= 10000.0 {
        return Err(format!(
            "CT centre value should be in range (-10000, 10000), but got `{centre}`"
        ));
    }
    Ok(centre)
}

fn ct_width_legal_range(s: &str) -> Result<f32, String> {
    let width: f32 = s
        .parse()
        .map_err(|_| format!("`{s}` is not a legal CT width value"))?;
    if width.is_nan() || width.is_infinite() {
        return Err(format!("`{s}` is not a legal CT width value"));
    }
    if width <= 0.0 {
        return Err(format!(
            "CT width value must be positive, but got `{width}`"
        ));
    }
    Ok(width)
}

impl CtWindow {
    pub fn run(&mut self) {
        // [scan_out/raw_npy] -> [scan_out/visual2d_raw]
        self.base_dir
            .extend(["medical_liver", "LiTS", "lits_train", "scan_out"]);

        let mut raw_npy_dir = self.base_dir.clone();
        raw_npy_dir.push("raw_npy");
        assert!(raw_npy_dir.is_dir());

        let mut v2d_raw_dir = self.base_dir.clone();
        v2d_raw_dir.push("visual2d_raw");
        fs::create_dir_all(v2d_raw_dir.as_path()).unwrap();

        let offset = self.ct_width / 2.0;
        Program::new(
            raw_npy_dir,
            v2d_raw_dir,
            self.ct_centre - offset,
            self.ct_centre + offset,
        )
        .run();
    }
}

struct Program {
    raw_npy_dir: PathBuf,
    v2d_raw_dir: PathBuf,
    seq: usize,
    lower_bound: f32,
    upper_bound: f32,
    scale: f32,
}

impl Program {
    #[inline]
    pub fn new(
        raw_npy_dir: PathBuf,
        v2d_raw_dir: PathBuf,
        lower_bound: f32,
        upper_bound: f32,
    ) -> Self {
        Self {
            raw_npy_dir,
            v2d_raw_dir,
            seq: 0,
            lower_bound,
            upper_bound,
            scale: 256.0 / (upper_bound - lower_bound),
        }
    }

    pub fn run(&mut self) {
        println!(
            "CT窗口规范化范围: [{:.2}, {:.2}]",
            self.lower_bound, self.upper_bound
        );
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

        let scan_3d: Array3<f32> = ndarray_npy::read_npy(self.raw_npy_dir.as_path()).unwrap();
        let mut scan_vec = scan_3d.into_raw_vec();

        let slice_len = scan_vec.len() / AREA;
        println!("\t切片个数: {slice_len}");

        for i in 0..slice_len {
            let offset = i * AREA;
            let f32_slice =
                unsafe { std::slice::from_raw_parts(scan_vec.as_mut_ptr().add(offset), AREA) };
            let buf: Vec<u8> = f32_slice
                .iter()
                .copied()
                .map(|f| self.normalize(f))
                .collect();

            let png = GrayImage::from_vec(WIDTH as u32, WIDTH as u32, buf).unwrap();
            self.v2d_raw_dir.push(format!("{i}.png"));
            png.save(self.v2d_raw_dir.as_path()).unwrap();
            self.v2d_raw_dir.pop();
        }

        self.v2d_raw_dir.pop();
        self.raw_npy_dir.pop();
    }

    #[inline]
    fn normalize(&self, ct: f32) -> u8 {
        if ct >= self.upper_bound {
            255_u8
        } else if ct <= self.lower_bound {
            0_u8
        } else {
            ((ct - self.lower_bound) * self.scale).floor() as u8
        }
    }
}
