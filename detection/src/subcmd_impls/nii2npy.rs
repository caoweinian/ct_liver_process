use clap::{Args, ValueEnum};
use label::prelude::TRAINING_SET_LEN;
use nifti::{IntoNdArray, NiftiObject, ReaderOptions};
use std::fs;
use std::path::PathBuf;

#[derive(Args, Debug)]
pub struct Nii2npy {
    /// 数据集根目录。
    #[arg(long = "base-dir", short = 'D')]
    base_dir: PathBuf,
    /// 处理`label`目录还是`scan`目录。
    #[arg(short, long, value_enum, default_value_t = TargetType::Label)]
    target: TargetType,
}

#[derive(ValueEnum, Clone, Copy, Debug)]
/// 表明处理标签(Ground Truth)集合还是扫描集合。
enum TargetType {
    /// 标签集合。
    Scan,
    /// 扫描集合。
    Label,
}

impl TargetType {
    #[inline]
    pub fn nii_prefix(self) -> &'static str {
        match self {
            TargetType::Scan => "volume",
            TargetType::Label => "segmentation",
        }
    }

    #[inline]
    pub fn is_scan(self) -> bool {
        matches!(self, TargetType::Scan)
    }

    #[inline]
    pub fn is_label(self) -> bool {
        matches!(self, TargetType::Label)
    }
}

impl Nii2npy {
    pub fn run(&mut self) {
        // [../{label, scan}] -> [{label_out, scan_out}/raw_npy]
        self.base_dir
            .extend(["medical_liver", "LiTS", "lits_train"]);

        let mut nii_dir = self.base_dir.clone();
        if self.target.is_scan() {
            nii_dir.push("scan");
        } else {
            nii_dir.push("label");
        }
        assert!(nii_dir.is_dir());

        let mut raw_npy_dir = self.base_dir.clone();
        if self.target.is_scan() {
            raw_npy_dir.extend(["scan_out", "raw_npy"]);
        } else {
            raw_npy_dir.extend(["label_out", "raw_npy"]);
        }
        fs::create_dir_all(raw_npy_dir.as_path()).unwrap();
        Program::new(nii_dir, raw_npy_dir, self.target).run();
    }
}

struct Program {
    nii_dir: PathBuf,
    raw_npy_dir: PathBuf,
    target: TargetType,
    seq: usize,
}

impl Program {
    #[inline]
    pub fn new(nii_dir: PathBuf, raw_npy_dir: PathBuf, target: TargetType) -> Self {
        Self {
            nii_dir,
            raw_npy_dir,
            target,
            seq: 0,
        }
    }

    pub fn run(&mut self) {
        while self.seq != TRAINING_SET_LEN {
            println!(
                "处理文件 `{}-{}.nii`...",
                self.target.nii_prefix(),
                self.seq
            );
            self.run_seq();
            self.seq += 1;
        }
    }

    fn run_seq(&mut self) {
        let nii_filename = format!("{}-{}.nii", self.target.nii_prefix(), self.seq);
        let npy_filename = format!("{}.npy", self.seq);
        self.nii_dir.push(nii_filename.as_str());
        self.raw_npy_dir.push(npy_filename.as_str());

        let obj = ReaderOptions::new()
            .read_file(self.nii_dir.as_path())
            .unwrap();

        // [512, 512, z] -> [z, 512, 512]
        if self.target.is_label() {
            let volume = obj.into_volume().into_ndarray::<u8>().unwrap();
            let v0 = volume.permuted_axes([2, 0, 1].as_slice());
            ndarray_npy::write_npy(self.raw_npy_dir.as_path(), &v0).unwrap();
        } else {
            let volume = obj.into_volume().into_ndarray::<f32>().unwrap();
            let v0 = volume.permuted_axes([2, 0, 1].as_slice());
            ndarray_npy::write_npy(self.raw_npy_dir.as_path(), &v0).unwrap();
        }

        self.raw_npy_dir.pop();
        self.nii_dir.pop();
    }
}
