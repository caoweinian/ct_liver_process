use super::WIDTH;
use crate::subcmd_impls::utils::rgb;
use clap::Args;
use image::io::Reader as ImageReader;
use image::{GrayImage, Rgb, RgbImage};
use label::prelude::{LiverSlice, VIS_BACKGROUND, VIS_BOUNDARY, VIS_LIVER};
use std::fs;
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

#[derive(Args, Debug)]
pub struct PngBatch {
    /// 输入目录。
    #[arg(long = "input-dir", short)]
    in_dir: PathBuf,
    /// 输出目录。
    #[arg(long = "output-dir", short)]
    out_dir: PathBuf,
    /// 使用RGB彩色图像输出，并指定颜色（默认灰色灰度图）。
    #[arg(long, short, value_parser = super::utils::color_valid_rgb_hex)]
    color: Option<Rgb<u8>>,
}

impl PngBatch {
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

            if let Some(s) = path.extension() {
                if s.to_str().unwrap() == "png" {
                    let png = ImageReader::open(path).unwrap().decode().unwrap();
                    let png_vec = png.as_bytes().to_vec();
                    let filename = path.file_name().unwrap();
                    self.out_dir.push(filename);
                    self.process(png_vec, self.out_dir.as_path());
                    self.out_dir.pop();
                }
            }
        }
    }

    fn process<P: AsRef<Path>>(&self, mut img: Vec<u8>, save_to: P) {
        let mut ct = unsafe { LiverSlice::new_unchecked(img.as_mut_ptr(), WIDTH, WIDTH) };
        let liver_areas = ct.vis_liver_area_group();
        if !liver_areas.is_empty() {
            let biggest_liver = ct.non_max_filling(liver_areas, VIS_BACKGROUND).unwrap();
            ct.clear_map();
            let bg_areas = ct.vis_background_area_group();
            for area in bg_areas.iter() {
                if ct.all_within(area.as_slice()) {
                    unsafe {
                        ct.fill_batch(area.as_slice(), VIS_LIVER);
                    }
                }
            }
            ct.draw_liver_boundary(biggest_liver.as_slice());
            ct.clear_map();
            let hole_areas = ct.vis_liver_boundary_area_group();
            ct.non_max_filling(hole_areas, VIS_BACKGROUND);
            ct.unique();
        }
        self.save(img, save_to);
        // GrayImage::from_vec(WIDTH as u32, WIDTH as u32, img).unwrap()
    }

    fn save<P: AsRef<Path>>(&self, img: Vec<u8>, path: P) {
        if let Some(color) = self.color {
            let mut png = RgbImage::new(WIDTH as u32, WIDTH as u32);
            let mut inner_offset = 0;
            for h in 0..(WIDTH as u32) {
                for w in 0..(WIDTH as u32) {
                    let pixel = img[inner_offset];
                    *png.get_pixel_mut(w, h) = match pixel {
                        VIS_LIVER => rgb::white(),
                        VIS_BOUNDARY => color,
                        VIS_BACKGROUND => rgb::black(),
                        _ => unreachable!(),
                    };
                    inner_offset += 1;
                }
            }
            png.save(path).unwrap();
        } else {
            let png = GrayImage::from_vec(WIDTH as u32, WIDTH as u32, img).unwrap();
            png.save(path).unwrap();
        }
    }
}
