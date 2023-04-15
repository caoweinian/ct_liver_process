use clap::{Parser, Subcommand};

#[derive(Parser, Debug)]
#[command(name = "detection")]
#[command(about = "处理`dataset/medical_liver/LiTS/`目录的工具集.")]
#[command(version, about, long_about = None)]
pub struct Cli {
    /// 子命令。
    #[command(subcommand)]
    command: Commands,
}

impl Cli {
    pub fn run_program(&mut self) {
        match self.command {
            Commands::Nii2npy(ref mut v) => v.run(),
            Commands::Rnpy2unpy(ref mut v) => v.run(),
            Commands::Rnpy2png(ref mut v) => v.run(),
            Commands::Unpy2png(ref mut v) => v.run(),
            Commands::Verify(ref mut v) => v.run(),
            Commands::CtWindow(ref mut v) => v.run(),
            Commands::PngBatch(ref mut v) => v.run(),
            Commands::ErodedCoe(ref mut v) => v.run(),
            Commands::Rnpy2unique(ref mut v) => v.run(),
            Commands::AlgoBench(ref mut v) => v.run(),
            Commands::PaperAlgos2npy(ref mut v) => v.run(),
            Commands::PaperGridGen(ref mut v) => v.run(),
        }
    }
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// 将医学图像数据格式nii文件(ground truth)转换为npy数据。
    Nii2npy(crate::subcmd_impls::nii2npy::Nii2npy),
    /// 将原数据集npy文件平滑化，在新目录生成新npy文件。
    Rnpy2unpy(crate::subcmd_impls::rnpy2unpy::Rnpy2unpy),
    /// 将原数据集npy文件转换为png文件。
    Rnpy2png(crate::subcmd_impls::rnpy2png::Rnpy2png),
    /// 将平滑化数据集npy文件转换为png文件。
    Unpy2png(crate::subcmd_impls::unpy2png::Unpy2png),
    /// 自动验证相关性质。
    Verify(crate::subcmd_impls::verify::Verify),
    /// 按照给定的CT窗位和窗宽，转化scan目录下的npy文件。
    CtWindow(crate::subcmd_impls::ct_window::CtWindow),
    /// 处理将一个目录下的二值png图像，使边缘为红色。
    PngBatch(crate::subcmd_impls::png_batch::PngBatch),
    /// 边缘侵蚀系数计算。
    ErodedCoe(crate::subcmd_impls::eroded_coefficient::ErodedCoefficient),
    /// 对原始去肿瘤npy图像中的肝脏物体唯一化（取最大的之一）保存在`unique`相关目录中。
    Rnpy2unique(crate::subcmd_impls::rnpy2unique::Rnpy2unique),
    /// 类内与类间的算法性能测试。
    AlgoBench(crate::subcmd_impls::algo_bench::AlgoBench),
    /// 生成四种算法对比的灰度图。
    PaperAlgos2npy(crate::subcmd_impls::paper_algos2npy::PaperAlgos2Npy),
    /// 生成算法效果演示大图。
    PaperGridGen(crate::subcmd_impls::paper_grid_gen::PaperGridGen),
}
