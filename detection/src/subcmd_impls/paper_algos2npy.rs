use crate::subcmd_impls::{AREA, WIDTH};
use clap::Args;
use image::GrayImage;
use label::prep::improc::consts::{BLACK, GRAY, WHITE};
use label::prep::TRAINING_SET_LEN;
use ndarray::Array3;
use std::collections::BTreeSet;
use std::path::PathBuf;
use std::{fs, mem};

#[derive(Args, Debug)]
pub struct PaperAlgos2Npy {
    /// 数据集根目录。
    #[arg(long = "base-dir", short = 'D')]
    base_dir: PathBuf,
    /// 需要转化的目录范围。
    #[arg(long, short, value_parser = super::utils::ranges_to_integers)]
    ranges: Option<BTreeSet<usize>>,
}

impl PaperAlgos2Npy {
    pub fn run(&mut self) {
        // [...label_out/algo_result/npy*] -> [.../label_out/paper_result/{npy*, v2d*}]
        self.base_dir
            .extend(["medical_liver", "LiTS", "lits_train", "label_out"]);
        assert!(self.base_dir.is_dir());

        let mut base_dir = mem::take(&mut self.base_dir);
        let mut algo_result_dir = base_dir.clone();
        algo_result_dir.push("algo_result");
        fs::create_dir_all(algo_result_dir.as_path()).unwrap();

        let mut paper_result_dir = base_dir.clone();
        paper_result_dir.push("paper_result");
        fs::create_dir_all(paper_result_dir.as_path()).unwrap();

        base_dir.push("unique_npy");

        let ranges: Vec<usize> = match self.ranges.take() {
            Some(r) => r.into_iter().collect(),
            _ => (0..TRAINING_SET_LEN).collect(),
        };

        Program::new(base_dir, algo_result_dir, paper_result_dir, ranges).run();
    }
}

struct Program {
    dirs: Vec<PathBuf>,
    ranges: Vec<usize>,
}

impl Program {
    #[inline]
    pub fn new(
        unique_npy_dir: PathBuf,
        algo_result_dir: PathBuf,
        paper_result_dir: PathBuf,
        ranges: Vec<usize>,
    ) -> Self {
        // dirs:
        // 0 -> [unique_npy_dir] (in)
        // 1-4 -> [algo_result/npy_algo_{canny, suzuki, hrvoje, ours}] (in)
        // 5-8 -> [paper_result/npy_algo_{canny, suzuki, hrvoje, ours}] (out)
        // 9-12 -> [paper_result/v2d_algo_{canny, suzuki, hrvoje, ours}] (out)
        let npy_paths = [
            "npy_algo_canny",
            "npy_algo_suzuki",
            "npy_algo_hrvoje",
            "npy_algo_ours",
        ];
        let v2d_paths = [
            "v2d_algo_canny",
            "v2d_algo_suzuki",
            "v2d_algo_hrvoje",
            "v2d_algo_ours",
        ];
        let [p1, p2, p3, p4] = Self::prepare_dir(algo_result_dir, npy_paths, true);
        let [p5, p6, p7, p8] = Self::prepare_dir(paper_result_dir.clone(), npy_paths, false);
        let [p9, p10, p11, p12] = Self::prepare_dir(paper_result_dir, v2d_paths, false);
        Self {
            dirs: vec![
                unique_npy_dir,
                p1,
                p2,
                p3,
                p4,
                p5,
                p6,
                p7,
                p8,
                p9,
                p10,
                p11,
                p12,
            ],
            ranges,
        }
    }

    #[inline]
    fn prepare_dir(base: PathBuf, dir_names: [&str; 4], needs_exists: bool) -> [PathBuf; 4] {
        let mut p1 = base.clone();
        p1.push(dir_names[0]);
        let mut p2 = base.clone();
        p2.push(dir_names[1]);
        let mut p3 = base.clone();
        p3.push(dir_names[2]);
        let mut p4 = base;
        p4.push(dir_names[3]);
        let ret = [p1, p2, p3, p4];
        if needs_exists {
            assert!(ret.iter().all(|p| p.is_dir()));
        } else {
            ret.iter()
                .for_each(|p| fs::create_dir_all(p.as_path()).unwrap());
        }
        ret
    }

    pub fn run(&mut self) {
        for i in 0..self.ranges.len() {
            let index = self.ranges[i];
            println!("处理目录`{index}`...");
            self.run_seq(index);
        }
    }

    fn run_seq(&mut self, seq: usize) {
        for npy_input_index in 1_usize..=4_usize {
            let npy_output_index = npy_input_index + 4;
            let v2d_output_index = npy_input_index + 8;
            let npy_filename = format!("{seq}.npy");
            let seq_filename = format!("{seq}");
            self.dirs[0].push(npy_filename.as_str());
            self.dirs[npy_input_index].push(npy_filename.as_str());

            let unique_array: Array3<u8> = ndarray_npy::read_npy(self.dirs[0].as_path()).unwrap();
            let algo_output: Array3<u8> =
                ndarray_npy::read_npy(self.dirs[npy_input_index].as_path()).unwrap();

            let result = Self::generate(&unique_array, &algo_output);
            let npy_output_dir = &mut self.dirs[npy_output_index];
            (*npy_output_dir).push(npy_filename.as_str());
            ndarray_npy::write_npy(npy_output_dir.as_path(), &result).unwrap();
            npy_output_dir.pop();

            let v2d_output_dir = &mut self.dirs[v2d_output_index];
            v2d_output_dir.push(seq_filename.as_str());
            fs::create_dir_all(v2d_output_dir.as_path()).unwrap();

            let result_buf = result.into_raw_vec();
            let z_len = result_buf.len() / AREA;
            for i in 0..z_len {
                let offset = i * AREA;
                let png_filename = format!("{i}.png");
                v2d_output_dir.push(png_filename.as_str());
                let this_buf = result_buf[offset..offset + AREA].to_vec();
                let png = GrayImage::from_vec(WIDTH as u32, WIDTH as u32, this_buf).unwrap();
                png.save(v2d_output_dir.as_path()).unwrap();
                v2d_output_dir.pop();
            }
            v2d_output_dir.pop();

            self.dirs[npy_input_index].pop();
            self.dirs[0].pop();
        }
    }

    fn generate(unique_array: &Array3<u8>, algo_output: &Array3<u8>) -> Array3<u8> {
        let unique_slice = unique_array.as_slice().unwrap();
        let algo_slice = algo_output.as_slice().unwrap();
        let mut ans: Vec<u8> = Vec::with_capacity(unique_array.len());
        let z_len = unique_slice.len() / AREA;
        for i in 0..z_len {
            let offset = i * AREA;
            let u_s = &unique_slice[offset..offset + AREA];
            let a_s = &algo_slice[offset..offset + AREA];
            Self::generate_one(u_s, a_s, &mut ans);
        }
        Array3::<u8>::from_shape_vec((z_len, WIDTH, WIDTH), ans).unwrap()
    }

    fn generate_one(unique_slice: &[u8], algo_slice: &[u8], out: &mut Vec<u8>) {
        for i in 0..unique_slice.len() {
            let unique_pixel = unique_slice[i];
            let algo_pixel = algo_slice[i];
            match (unique_pixel, algo_pixel) {
                (_, WHITE) => out.push(GRAY),
                (BLACK, _) => out.push(BLACK),
                _ => out.push(WHITE),
            }
        }
    }
}
