# 《二值图像实体的8邻域稀疏轮廓提取算法》稿件代码实验

本目录下的代码由该稿件的第一作者完成。为了尽可能精确地测试各个算法的底层原生执行效率，本实验使用100%纯Rust代码编写。相比于C++，Rust易于跨平台构建，方便在各个平台上一键构建运行复现。

## 兼容性

已经在MacOS、Linux(Ubuntu distribution)上成功构建并运行。

## 目录组织

本项目目录组织如下：

```text
.
├── Cargo.toml          工作空间管理文件
├── README.md           项目说明
├── dataset             数据集
├── detection           核心算法
├── label               核心数据结构、部分算法
├── pipeline.sh         流水线处理脚本(可忽略)
```

该项目由两个包`label`和`detection`组成：

- `label`包提供了对LiTS17数据集中单张512*512图像切片的操作，核心数据结构与部分非核心算法在`label/src/prep/improc.rs`中的`struct LiverSlice`中。
- `detection`包依赖于`label`，实现了稿件中描述的核心算法(具体见`detection/src/subcmd_impls/algo_bench.rs`)

## 从零开始构建

1. [安装Rust](https://www.rust-lang.org/learn/get-started)。
2. [安装适用于Rust构建的OpenCV](https://crates.io/crates/opencv)。
3. 在本目录运行命令`cargo build --release`。

构建成功后，在`target/release`目录下将生成可执行程序`detection`（不同操作系统下名字不同）。

## 数据集目录组织

本文的验证用数据集在目录`dataset`中。其目录组织如下：

```text
./dataset
└── medical_liver
    └── LiTS
        └── lits_train
            └── label
                └── segmentation-0.nii
```

原LiTS17数据集中共有131组患者数据，其ground truth标注文件分别命名为`segmentation-0.nii`, `segmentation-1.nii`, ..., `segmentation-130.nii`。 为使目录不至于过大，该目录只包含了一个文件`segmentation-0.nii`。在编译软件时，需要将`label/src/prep/improc.rs`中`TRAINING_SET_LEN`的值从131改为1，然后再运行`cargo build --release`。

> 全部131组标签可在[AI Studio](https://aistudio.baidu.com/aistudio/datasetdetail/10273)下载；只需要下载`lits_train.zip`。

## 从头开始复现的方法

复现分为两部分，一是运行时间，二是轮廓的侵蚀系数。先在本目录下依次运行以下命令：

```bash
cargo build --release;
cargo run --release -p detection -- nii2npy -D ./dataset;
```

此时会在`lits_train`下生成`label_out`目录，其中的`raw_npy`即保存了三值图像（0为背景像素，1为肝脏像素，2为肿瘤像素）的原生标签，npy格式的文件可以被`numpy`直接读取，便于深度学习语义分割之用。

然后运行：

```bash
cargo run --release -p detection -- rnpy2unique -D ./dataset;
```

此时会在`lits_train`下生成目录`unique_npy`和`visual2d_unique`。其中`unique_npy`目录下的npy是经过预处理（肿瘤像素变为前景像素，然后将前景唯一化、背景唯一化）后生成的对应文件；`visual2d_unique`目录下是该npy对应的可视化二值图像。

测试算法运行速度的命令为：

```bash
cargo run --release -p detection -- algo-bench -D ./dataset;
```

> 在测试输出中显示的`hrvoje算法`即为原文提到的Leventic算法。

测试侵蚀系数的命令为：

```bash
cargo run --release -p detection -- eroded-coe -D ./dataset;
```
