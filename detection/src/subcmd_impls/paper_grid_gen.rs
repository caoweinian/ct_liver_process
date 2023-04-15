use crate::subcmd_impls::utils::rgb;
use clap::Args;
use image::io::Reader as ImageReader;
use image::{GenericImage, GrayImage, Luma, RgbImage};
use label::prep::improc::consts::{BLACK, GRAY, WHITE};
use std::cmp::{max, min};
use std::fs;
use std::path::PathBuf;
use walkdir::WalkDir;

#[derive(Args, Debug)]
pub struct PaperGridGen {
    /// 输入目录。
    #[arg(long = "input-dir", short)]
    in_dir: PathBuf,
    /// 输出目录。
    #[arg(long = "output-dir", short)]
    out_dir: PathBuf,
    /// 主物体显示窗口补齐像素大小，默认为5。
    #[arg(long, default_value_t = 5)]
    out_padding: u32,
    /// 输入图像中单个像素在输出图像中的像素宽度，默认为6。
    #[arg(long, default_value_t = 6)]
    scale: u32,
    /// 输出图像中每个前景格的黑色边缘补齐宽度，默认为1。
    #[arg(long, default_value_t = 1)]
    in_padding: u32,
}

impl PaperGridGen {
    pub fn run(&mut self) {
        // [...input-dir/*.png] -> [...output-dir/*.png]
        assert!(self.in_dir.is_dir());
        fs::create_dir_all(self.out_dir.as_path()).unwrap();

        for entry in WalkDir::new(self.in_dir.as_path()).max_depth(1) {
            let entry = entry.unwrap();
            let path = entry.path();
            if !path.is_file() {
                continue;
            }
            let Some(s) = path.extension() else { continue; };
            if s.to_str().unwrap() != "png" {
                continue;
            }
            let filename = path.file_name().unwrap();
            let png = ImageReader::open(path).unwrap().decode().unwrap();
            let mut png = png.into_luma8();
            let (x, y, w, h) = self.get_window(&png);
            let png = png.sub_image(x, y, w, h).to_image();
            let pretty = self.gray2pretty(&png);
            self.out_dir.push(filename);
            pretty.save(self.out_dir.as_path()).unwrap();
            self.out_dir.pop();
        }
    }

    fn get_window(&self, png: &GrayImage) -> (u32, u32, u32, u32) {
        // (x, y, width, height)
        // x是横向增加，y是纵向增加
        let (width, height) = (png.width(), png.height());
        let (mut x_min, mut x_max) = (width, 0_u32);
        let (mut y_min, mut y_max) = (height, 0_u32);

        for y in 0..height {
            for x in 0..width {
                if *png.get_pixel(x, y) != Luma([BLACK]) {
                    x_min = min(x_min, x);
                    x_max = max(x_max, x);
                    y_min = min(y_min, y);
                    y_max = max(y_max, y);
                }
            }
        }
        let x_base = x_min.saturating_sub(self.out_padding);
        let y_base = y_min.saturating_sub(self.out_padding);
        let x_len = min(x_max - x_min + 2 * self.out_padding, width);
        let y_len = min(y_max - y_min + 2 * self.out_padding, width);
        (x_base, y_base, x_len, y_len)
    }

    fn gray2pretty(&self, png: &GrayImage) -> RgbImage {
        let (width, height) = (png.width(), png.height());
        let mut rgb = RgbImage::new(width * self.scale, height * self.scale);
        for h in 0..height {
            for w in 0..width {
                let h_base = h * self.scale;
                let w_base = w * self.scale;
                let gray_pixel: u8 = png.get_pixel(w, h).0[0];
                if gray_pixel == BLACK {
                    for h_pos in h_base..h_base + self.scale {
                        for w_pos in w_base..w_base + self.scale {
                            *rgb.get_pixel_mut(w_pos, h_pos) = rgb::black();
                        }
                    }
                } else {
                    // 上条带
                    for k in 0..self.in_padding {
                        for w_pos in w_base..w_base + self.scale {
                            *rgb.get_pixel_mut(w_pos, h_base + k) = rgb::black();
                        }
                    }
                    // 下条带
                    for k in self.scale - self.in_padding..self.scale {
                        for w_pos in w_base..w_base + self.scale {
                            *rgb.get_pixel_mut(w_pos, h_base + k) = rgb::black();
                        }
                    }
                    // 左条带
                    for k in 0..self.in_padding {
                        for h_pos in h_base..h_base + self.scale {
                            *rgb.get_pixel_mut(w_base + k, h_pos) = rgb::black();
                        }
                    }
                    // 右条带
                    for k in self.scale - self.in_padding..self.scale {
                        for h_pos in h_base..h_base + self.scale {
                            *rgb.get_pixel_mut(w_base + k, h_pos) = rgb::black();
                        }
                    }

                    let fill = match gray_pixel {
                        WHITE => rgb::yellow(),
                        GRAY => rgb::purple(),
                        _ => unreachable!(),
                    };
                    for h_pos in h_base + 1..=(h_base + self.scale - 2 * self.in_padding) {
                        for w_pos in w_base + 1..=(w_base + self.scale - 2 * self.in_padding) {
                            *rgb.get_pixel_mut(w_pos, h_pos) = fill;
                        }
                    }
                }
            }
        }
        rgb
    }
}
