"""
LiTS数据集预处理过程。
"""

import os
import sys
import enum
import platform
import ctypes
import time
import json
from collections import deque

import numpy as np
import nibabel as nib
from consts import TrainingSetLen, LocalDirectory
from PIL import Image


import matplotlib.pyplot as plt
import matplotlib

def _mkdir_r_must(d: str) -> None:
    if not os.path.exists(d):
        os.makedirs(d, exist_ok=True)
    elif not os.path.isdir(d):
        raise NotADirectoryError(f'{d} is not a directory')


class LiTSLabel(enum.Enum):
    """
    LiTS数据集的标签。在原始数据集中，背景为0，肝脏为1，肿瘤为2。本类描述的是重处理后的标签赋值。
    """
    Background = 0b_0000_0000
    Boundary = 0b_1000_0000
    Tumor = 0b_1100_0000
    Liver = 0b_1111_1111


# Prep stands for Preprocess
class PrepLabel:
    def __init__(self, base_dir: str = LocalDirectory):
        self.base_dir = base_dir
        self.label_in_dir = os.path.join(self.base_dir, 'lits_train', 'label')
        self.label_out_base_dir = os.path.join(self.base_dir, 'lits_train', 'label_out')
        self.label_out_visual_raw_dir = os.path.join(self.label_out_base_dir, 'visual2d_raw')
        self.label_out_raw_npy_dir = os.path.join(self.label_out_base_dir, 'raw_npy')
        self.label_out_visual_unified_dir = os.path.join(self.label_out_base_dir, 'visual2d_unified')
        self.label_out_unified_npy_dir = os.path.join(self.label_out_base_dir, 'unified_npy')
        self.label_out_json_dir = os.path.join(self.label_out_base_dir, 'seq_label')

    @staticmethod
    def _dll_path():
        abs_path = os.path.join(os.path.abspath(os.curdir), 'lib')
        sys_name = platform.uname()[0]
        if sys_name == 'Darwin':
            path = os.path.join(abs_path, 'liblabel.dylib')
        elif sys_name == 'Linux':
            path = os.path.join(abs_path, 'liblabel.so')
        else:
            path = os.path.join(abs_path, 'liblabel.dll')
        return path

    @staticmethod
    def _get_nii_name(index: int) -> str:
        return f'segmentation-{index}.nii'

    def count_labels(self) -> None:
        """
        初步探索整体训练集的像素分布情况。
        :return: None
        """
        background, liver, tumor = 0, 0, 0
        for no in range(TrainingSetLen):
            filename = PrepLabel._get_nii_name(no)
            print(f'Processing {filename}...')
            nii_path = os.path.join(self.label_in_dir, filename)
            raw_ct = nib.load(nii_path)
            ct0: np.ndarray = raw_ct.dataobj[:, :, :]
            # assert ((ct0 <= 2) | (ct0 >= 0)).sum() == ct0.size  # met
            if ct0.dtype != np.uint8:
                ct0 = ct0.astype(np.uint8)
            b_cnt = (ct0 == LiTSLabel.Background.value).sum()
            l_cnt = (ct0 == LiTSLabel.Liver.value).sum()
            t_cnt = (ct0 == LiTSLabel.Tumor.value).sum()
            print(f'\tBackground pixel: {b_cnt}\n\tLiver pixel: {l_cnt}\n\tTumor pixel: {t_cnt}')
            background += b_cnt
            liver += l_cnt
            tumor += t_cnt
        # 15020370189, 332764489, 18465194
        print(f'\n-----\nTotal:\n\tBackground pixel: {background}\n\tLiver pixel: {liver}\n\tTumor pixel: {tumor}')

    def count_boundary_neighbors(self) -> None:
        """
        验证在经过处理后的每张肝脏CT图像中，每个肝脏边缘像素的8-邻域中有且仅有两个肝脏边缘像素。
        :return: None
        """
        for num in range(TrainingSetLen):
            filename = f'{num}.npy'
            print(f'Testing {filename}...')
            npy_path = os.path.join(self.label_out_unified_npy_dir, filename)
            ct0: np.ndarray = np.load(npy_path)
            singular_count = 0

            for z in range(ct0.shape[0]):
                img_z: np.ndarray = ct0[z, :, :]
                for h, w in np.argwhere(img_z == LiTSLabel.Boundary.value):
                    new_val = (img_z[h - 1: h + 2, w - 1: w + 2] == LiTSLabel.Boundary.value).sum() - 1
                    if new_val != 2:
                        print(f'\tFind singular value {new_val} in slice {z}, pos {(h, w)}')
                        singular_count += 1
            if singular_count == 0:
                print('-----\nNo singular value detected. Fine!')
            elif singular_count == 1:
                print(f'-----\nFind {singular_count} singular value. Check it!')
            else:
                print(f'-----\nFind {singular_count} singular values. Check them!')

    def statistics(self) -> None:
        """
        可视化预处理后的肝脏边缘像素数统计。
        :return: None
        """
        num_having_boundary = 0
        total = 0
        num_array = []
        for num in range(TrainingSetLen):
            filename = f'{num}.npy'
            print(f'Processing {filename}...', end='')
            sys.stdout.flush()
            npy_path = os.path.join(self.label_out_unified_npy_dir, filename)
            ct0: np.ndarray = np.load(npy_path)

            for z in range(ct0.shape[0]):
                img_z: np.ndarray = ct0[z, :, :]
                boundary_cnt = (img_z == LiTSLabel.Boundary.value).sum()
                if boundary_cnt > 0:
                    num_having_boundary += 1
                    total += boundary_cnt
                    num_array.append(boundary_cnt)
            print(f'cur avg: {total / num_having_boundary}')
        # font: 设置中文
        # facecolor: 长条形的颜色
        # edgecolor: 长条形边框的颜色
        # alpha: 透明度
        matplotlib.rcParams['font.family'] = ['Heiti TC']
        matplotlib.rcParams['axes.unicode_minus'] = False
        data = np.array(num_array)
        plt.hist(data, bins=400, density=False, facecolor='tab:blue', edgecolor='tab:orange', alpha=0.7)
        plt.xlabel('边缘像素数')
        plt.ylabel('频数')
        plt.title('频数分布图')
        plt.show()

    def raw_npy_to_unified(self) -> None:
        """
        预处理`label_raw_npy`目录下的所有npy文件并保存到`label_unified_npy`目录。保证每张512 * 512的CT切片图中：
        1. 所有肿瘤区域被修改为肝脏区域；
        2. 至多只有一个肝脏区域（按4-邻域连通标准）。选择面积最大区域的保存；
        3. 使用128灰度值标注出肝脏边缘（若存在；按4-邻域连通标准）。
        :return: None
        """
        _mkdir_r_must(self.label_out_unified_npy_dir)
        lib_label = ctypes.cdll.LoadLibrary(PrepLabel._dll_path())  # ffi -> rust exports
        unify_liver_area = lib_label.unify_liver_area

        for i in range(TrainingSetLen):
            t0 = time.time()
            print(f'Unifying task {i}...', end='')
            sys.stdout.flush()
            filename = f'{i}.npy'
            src_path = os.path.join(self.label_out_raw_npy_dir, filename)
            src_cstr = ctypes.c_char_p(src_path.encode('utf-8'))
            dest_path = os.path.join(self.label_out_unified_npy_dir, filename)
            dest_cstr = ctypes.c_char_p(dest_path.encode('utf-8'))
            result_code = unify_liver_area(src_cstr, dest_cstr)
            print(f'return code {result_code}, total {time.time() - t0:.3f}s')

    def nii_to_npy(self, in_dir: str, out_dir: str) -> None:
        """
        将LiTS数据集的医学nii格式转换为numpy并保存为npy文件。
        :param in_dir: 输入目录，其中有若干nii文件。
        :param out_dir: 输出目录。
        :return: None
        """
        _mkdir_r_must(out_dir)
        for num in range(TrainingSetLen):
            nii_path = os.path.join(in_dir, self._get_nii_name(num))
            img: np.ndarray = nib.load(nii_path).dataobj[:, :, :]
            if img.dtype != np.uint8:
                img = img.astype(np.uint8)
            assert (img >= 0).all()
            assert (img <= 2).all()
            # img[img == 1] = LiTSLabel.Liver.value  # liver: 255
            # img[img == 2] = LiTSLabel.Tumor.value  # tumor: 192
            img = img.transpose((2, 0, 1))  # [512, 512, z] -> [z, 512, 512]
            assert img.shape[1:] == (512, 512)
            print(f'Running task {num}...\nShape: {img.shape}')
            print('Saving npy file...', end='')
            sys.stdout.flush()
            np.save(os.path.join(out_dir, f'{num}.npy'), img)
            print('done.')

    def npy_to_png(self, in_dir: str, out_dir: str) -> None:
        """
        在`out_dir`目录创建强可视化二维标签。
        :param in_dir: 输入目录，其中应当有形如`{num}.npy`的文件(0 <= num <= 130)。
        :param out_dir: 输出目录，将保存形如`{id}/{num}.png`的文件，其中`id`是数据集序号，`num`是三维肝脏切片序号。
        :return: None
        """
        _mkdir_r_must(in_dir)
        _mkdir_r_must(out_dir)

        for num in range(TrainingSetLen):
            visual_dest_dir = os.path.join(out_dir, f'{num}')
            _mkdir_r_must(visual_dest_dir)
            npy_path = os.path.join(in_dir, f'{num}.npy')
            img: np.ndarray = np.load(npy_path)
            z = img.shape[0]
            print(f'Generating png for task {num}. Totaling {z} images...', end='')
            sys.stdout.flush()
            t0 = time.time()
            # assert img.shape[1:] == (512, 512)
            for sub_num in range(z):
                img_z = Image.fromarray(img[sub_num, :, :])
                img_z.save(os.path.join(visual_dest_dir, f'{sub_num}.png'))
            print(f'Done. Consuming {time.time() - t0:.3f}s')

    @staticmethod
    def verify_only_one_cycle(in_dir: str) -> None:
        """
        假设所有肝脏图像中，边缘的8-邻域中有且仅有两个边缘。该函数用于测试每张有肝脏边缘的图像是否有且仅有一个环。
        :return: None
        """
        failed_times = 0
        for num in range(TrainingSetLen):
            npy_path = os.path.join(in_dir, f'{num}.npy')
            img: np.ndarray = np.load(npy_path)
            z = img.shape[0]
            print(f'Testing {num}...')
            for z0 in range(z):
                img_z = img[z0, :, :]
                boundary_pos: np.ndarray = np.argwhere(img_z == LiTSLabel.Boundary.value)
                if boundary_pos.size == 0:
                    continue
                b_cnt = (img_z == LiTSLabel.Boundary.value).sum()
                h, w = next(iter(boundary_pos))
                bfs_cnt = PrepLabel._bfs(img_z, h, w)
                assert bfs_cnt <= b_cnt
                if bfs_cnt < b_cnt:
                    print(f'\tError: more than one cycle in slice {z0}. {bfs_cnt} < {b_cnt}')
                    failed_times += 1
        if failed_times == 0:
            print('-----\nTests all pass.')
        else:
            print(f'-----\nFailed test(s): {failed_times}')

    @staticmethod
    def _bfs(img: np.ndarray, h: int, w: int) -> int:
        cnt = 0
        q = deque([(h, w)])
        while q:
            for i in range(len(q)):
                h0, w0 = q.popleft()
                if img[h0, w0] != LiTSLabel.Boundary.value:
                    continue
                cnt += 1
                img[h0, w0] = LiTSLabel.Background.value
                base_h, base_w = h0 - 1, w0 - 1
                for h_delta, w_delta in np.argwhere(
                        img[base_h: base_h + 3, base_w: base_w + 3] == LiTSLabel.Boundary.value):
                    q.append((base_h + h_delta, base_w + w_delta))
        return cnt

    def gen_train_label_seq(self) -> None:
        """
        为每个预处理后的标签图像生成json格式摘要。
        :return: None
        """
        _mkdir_r_must(self.label_out_json_dir)
        lib_label = ctypes.cdll.LoadLibrary(PrepLabel._dll_path())  # ffi -> rust exports
        make_sequential_labels = lib_label.make_sequential_labels

        for num in range(TrainingSetLen):
            t0 = time.time()
            print(f'Generating json label for task {num}...', end='')
            sys.stdout.flush()
            npy_path = os.path.join(self.label_out_unified_npy_dir, f'{num}.npy')
            npy_path_cstr = ctypes.c_char_p(npy_path.encode('utf-8'))
            dest_path = os.path.join(self.label_out_json_dir, str(num))
            dest_path_cstr = ctypes.c_char_p(dest_path.encode('utf-8'))
            result_code = make_sequential_labels(npy_path_cstr, dest_path_cstr)
            print(f'return code {result_code}, total {time.time() - t0:.3f}s')

    def merge_json_for_polyfit(self, dest_path: str, normalized_to_100x100: bool = True) -> None:
        """
        将预处理后的边缘集标准化。
        :param dest_path: 输出路径（相对于label_out_base_dir）
        :param normalized_to_100x100: 是否将数据点从[0, 512]标准化到[-50, 50]范围
        :return: None
        """
        output = []
        if normalized_to_100x100:
            shift_trans = np.array([[1., 0., -256.5], [0., 1., -256.5], [0., 0., 1.]], dtype=np.float64)
            rotate_trans = np.array([[0., 1., 0.], [-1., 0., 0.], [0., 0., 1.]], dtype=np.float64)
            scale_trans = np.array([[100. / 512., 0., 0.], [0., 100. / 512., 0.], [0., 0., 1.]], dtype=np.float64)
            trans = scale_trans @ rotate_trans @ shift_trans
        else:
            trans = None

        for seq in range(TrainingSetLen):
            print(f'Merging directory `{seq}`...')
            d = os.path.join(self.label_out_json_dir, str(seq))
            for json_filename in os.listdir(d):
                components = os.path.splitext(json_filename)
                if len(components) == 2 and components[1] == '.json':
                    target = os.path.join(d, json_filename)
                    j = open(target, 'r', encoding='utf-8')
                    json_value = json.load(j)
                    if json_value:
                        if normalized_to_100x100:
                            np_arr = np.array(json_value)
                            rows = np_arr.shape[0]
                            arr = np.column_stack((np_arr, np.ones(rows))).T
                            result = (trans @ arr)[:2, :].T
                            output.append(result.tolist())
                        else:
                            output.append(json_value)
                    j.close()

        out_handle = open(os.path.join(self.label_out_base_dir, dest_path), 'w', encoding='utf-8')
        json.dump(output, out_handle)
        out_handle.close()
        print('\nAll done.')

    @staticmethod
    def ct_window(in_npy: str, out_dir: str, ct_centre: int, ct_width: int):
        assert ct_width > 0
        os.makedirs(out_dir, exist_ok=True)
        arr: np.ndarray = np.load(in_npy)
        arr = arr.astype(np.float64)
        lower_bound, upper_bound = ct_centre - ct_width / 2, ct_centre + ct_width / 2
        arr -= lower_bound
        upper_bound -= lower_bound
        arr[arr < 0] = 0.
        arr[arr > upper_bound] = 255.
        arr *= 256.0 / ct_width
        arr = arr.astype(np.uint8)
        print(arr.min(), arr.max(), arr.mean())


class PrepScan:
    def __init__(self, base_dir: str = LocalDirectory):
        self.base_dir = base_dir
        self.scan_in_dir = os.path.join(self.base_dir, 'lits_train', 'scan')
        self.scan_out_base_dir = os.path.join(self.base_dir, 'lits_train', 'scan_out')
        self.scan_out_raw_npy_dir = os.path.join(self.scan_out_base_dir, 'raw_npy')

    def nii_to_npy(self) -> None:
        """
        将LiTS数据集的医学nii格式转换为numpy并保存为npy文件。
        :return: None
        """
        _mkdir_r_must(self.scan_out_base_dir)
        for num in range(TrainingSetLen):
            nii_path = os.path.join(self.scan_in_dir, self._get_nii_name(num))
            img: np.ndarray = nib.load(nii_path).dataobj[:, :, :]
            # if img.dtype != np.uint8:
            #     img = img.astype(np.uint8)
            img = img.transpose((2, 0, 1))  # [512, 512, z] -> [z, 512, 512]
            assert img.shape[1:] == (512, 512)
            print(f'Running task {num}...\nShape: {img.shape}')
            # print('Saving npy file...', end='')
            sys.stdout.flush()
            # np.save(os.path.join(self.scan_out_raw_npy_dir, f'{num}.npy'), img)
            print(img.min(), img.max(), img.mean())
            print('done.')

    @staticmethod
    def _get_nii_name(index: int) -> str:
        return f'volume-{index}.nii'

    @staticmethod
    def ct_window(in_npy: str, out_dir: str, ct_centre: float, ct_width: float):
        assert ct_width > 0.0
        arr: np.ndarray = np.load(in_npy)
        arr = arr.astype(np.float64)
        lb, ub = ct_centre - ct_width / 2.0, ct_centre + ct_width / 2.0
        arr -= lb
        ub -= lb
        arr[arr < 0.0] = 0.0
        arr[arr > ub] = 255.0
        arr[arr <= ub] *= 255.0 / ct_width
        arr = arr.astype(np.uint8)
        z_len = arr.shape[0]
        for i in range(z_len):
            img_z = Image.fromarray(arr[i, :, :])
            img_z.save(os.path.join(out_dir, f'{i}.png'))
        print(arr.min(), arr.max(), arr.mean())


if __name__ == '__main__':
    tool = PrepLabel()
    # tool.nii_to_npy(tool.label_in_dir, tool.label_out_raw_npy_dir)
    # tool.npy_to_png(tool.label_out_raw_npy_dir, tool.label_out_visual_raw_dir)
    #
    # tool.raw_npy_to_unified()
    # tool.npy_to_png(tool.label_out_unified_npy_dir, tool.label_out_visual_unified_dir)
    # tool.count_boundary_neighbors()
    #
    # tool.statistics()
    #
    # tool.verify_only_one_cycle(tool.label_out_unified_npy_dir)
    #
    # tool.gen_train_label_seq()
    #
    # tool.merge_json_for_polyfit('boundaries_100x100.json', normalized_to_100x100=True)
    # tool.merge_json_for_polyfit('boundaries.json', normalized_to_100x100=False)

    # tool = PrepScan()
    # tool.nii_to_npy()
    # tool.ct_window('/Volumes/LaCie_exFAT/dataset/medical_liver/LiTS/lits_train/scan_out/raw_npy/0.npy',
    #       '../draft/img0', 60.0, 200.0)
