from typing import Optional, Tuple, Callable
import time
import math

import torch
import matplotlib.pyplot as plt
import numpy as np


class Poly4:
    def __init__(self, points: torch.Tensor, b: Optional[torch.Tensor] = None, v3: Optional[torch.Tensor] = None,
                 epsilon: float = 1e-8, delta: float = 1.0):
        """
        初始化。一组要拟合的点从属于一个Poly4类。
        :param points: 要拟合的点集[N, 2]。
        :param b: 对称[3, 3]矩阵。
        :param v3: 非主要项p3(x, y)的系数向量，排列顺序为a30, a21, a12, a03, a20, a11, a02, a10, a01, a00.
        :param epsilon: 为了让拟合曲线更紧贴在实际数据点上而设置的超参数。
        :param delta: 当目标多项式经过一次大迭代，降低的值低于此值时，程序会提前返回（否则会迭代指定多数目后返回）。
        """
        # 初始化为全零参数。注意这不会使程序直接结束，因为 A = B^2 + `self.eps * I`。这个迭代起点是很合适的。
        if b is None:
            b = torch.zeros((3, 3), dtype=torch.float32)
        if v3 is None:
            v3 = torch.zeros(10)
        assert b.shape == (3, 3)
        assert b.dtype == torch.float32
        # assert (b.T == b).all()  # 去掉，因为浮点误差
        assert (b.abs() <= 10.0).all()  # 不要让初始参数过大

        assert v3.shape == (10,)
        assert v3.dtype == torch.float32
        # assert (v3.abs() <= 100.0).all()  # 不要让初始参数过大，除了常数项

        assert len(points.shape) == 2 and points.shape[1] == 2
        assert (points.abs() <= 50.0).all()  # LiTS数据集512 * 512分辨率的图像的宽和高都已经仿射变换到[-50.0, 50.0]区间中
        assert epsilon > 0.0  # 0.0应该不合适

        assert not b.requires_grad
        assert not v3.requires_grad

        self.B: torch.Tensor = b
        # 不变式：当访问self.A时，它目前一定是`self.B ** 2 + self.eps * I`。
        self.A: torch.Tensor = torch.empty(3, 3, dtype=torch.float32)
        self.v3: torch.Tensor = v3
        self.points: torch.Tensor = points.requires_grad_(True)
        self.eps: float = epsilon
        self.delta: float = delta
        self.poly_result: torch.Tensor = torch.tensor(float('inf'))
        self.orig_poly_result: torch.Tensor = torch.tensor(float('inf'))
        self.time_cost = None
        self.iter_time = 0  # 大迭代次数（相对于`class Minimizer`的小迭代）

    @staticmethod
    def make_p3_matrix(v3: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        """
        创建计算非主要项p3(x, y)的指数设计矩阵。
        :param v3: [10,]矩阵。
        :param points: [N, 2]矩阵，N是平面点集的个数。
        :return: [10, N]矩阵。
        """
        assert not v3.requires_grad
        assert points.requires_grad
        columns = []
        # index_sum是x+y的指数，y_index是y的指数
        for index_sum in range(3, -1, -1):
            for y_index in range(index_sum + 1):
                columns.append(points[:, 0] ** (index_sum - y_index) * points[:, 1] ** y_index)
        result = torch.stack(columns)
        assert result.shape == (10, points.shape[0])
        return result

    @staticmethod
    def make_p4_matrix(points: torch.Tensor) -> torch.Tensor:
        """
        创建计算主要项p4(x, y)的指数设计矩阵。
        :param points: [N, 2]矩阵，N是平面点集个数。
        :return: [N, 3]矩阵。
        """
        assert points.requires_grad
        col1 = points[:, 0] ** 2
        col2 = points[:, 0] * points[:, 1]
        col3 = points[:, 1] ** 2
        result = torch.stack([col1, col2, col3], dim=1)
        assert result.shape == (points.shape[0], 3)
        return result

    def evaluate_origin(self) -> torch.Tensor:
        """
        计算原多项式的加权和（对每个点的 p^2/grad^2 求和）。
        :return: 计算结果；一个单元素的torch.Tensor，并可反传梯度到`self.points`。
        """
        assert self.points.grad is None or (torch.zeros_like(self.points.grad) == self.points.grad).all()
        self.A = self.B ** 2 + self.eps * torch.eye(3)
        v_mat: torch.Tensor = Poly4.make_p4_matrix(self.points)  # 多项式四阶主要项设计矩阵
        s_mat: torch.Tensor = Poly4.make_p3_matrix(self.v3, self.points)  # 多项式非主要项（3+2+1+0阶）设计矩阵
        poly4_result: torch.Tensor = ((v_mat @ self.A) * v_mat).sum(axis=1)
        poly3_result: torch.Tensor = self.v3 @ s_mat
        point_wise_sum: torch.Tensor = poly3_result + poly4_result  # 每个点对应的多项式值
        point_wise_sum.sum().backward()
        weights: torch.Tensor = self.get_points_weights()
        return (point_wise_sum ** 2 * weights).sum()

    def evaluate(self, save: bool, weights: Optional[torch.Tensor] = None,
                 replaced_by: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> torch.Tensor:
        """
        初始化，计算加权多项式的平方和。
        :param save: 是否将计算结果保存到`self.poly_result`中。
        :param weights: 对每个点的多项式平方的加权系数。若为None，则系数为全1。
        :param replaced_by: 若非None，则在计算时以该参数代替`(self.B, self.v3)`。
        :return: 计算结果；一个单元素的torch.Tensor，并可反传梯度到`self.points`。
        """
        pts_len: int = self.points.shape[0]  # 点的个数
        if weights is None:
            weights = torch.ones(pts_len)  # 无加权也即全1加权
        else:
            assert weights.shape == (pts_len,)
        v_mat: torch.Tensor = Poly4.make_p4_matrix(self.points)  # 多项式四阶主要项设计矩阵
        if replaced_by is None:
            self.A = self.B ** 2 + self.eps * torch.eye(3)  # 满足`self.A`有意义的不变式
            s_mat: torch.Tensor = Poly4.make_p3_matrix(self.v3, self.points)  # 多项式非主要项（3+2+1+0阶）设计矩阵
            poly4_result: torch.Tensor = ((v_mat @ self.A) * v_mat).sum(axis=1)
            poly3_result: torch.Tensor = self.v3 @ s_mat
        else:
            b, v3 = replaced_by
            a: torch.Tensor = b ** 2 + self.eps * torch.eye(3)
            s_mat: torch.Tensor = Poly4.make_p3_matrix(v3, self.points)
            poly4_result: torch.Tensor = ((v_mat @ a) * v_mat).sum(axis=1)
            poly3_result: torch.Tensor = v3 @ s_mat
        assert poly3_result.shape == poly4_result.shape == (pts_len,)
        point_wise_poly_result: torch.Tensor = (poly3_result + poly4_result) ** 2  # 每个点对应多项式值的平方
        result: torch.Tensor = (point_wise_poly_result * weights).sum()  # 对每个点对应多项式值的平方进行加权求和
        if save:
            self.poly_result = result
        return result

    def clear_points_gradients(self) -> None:
        """
        将`self.points`的梯度信息清零。
        :return: None
        """
        if self.points.grad is not None:
            self.points.grad.zero_()
        else:
            print('Warning: invoke `Poly4.clear_points_gradients()` but `self.points.grad` is None. Maybe a misuse?')

    def minimize_arguments(self, weights: Optional[torch.Tensor] = None) -> None:
        """
        核心算法。最小化多项式的参数，包括`self.B`和`self.v3`。
        副作用：
            1. 无论返回值是True还是False，都会更新`self.poly_result`为更小的新值。
            2. `self.B`和`self.v3`将会更新为更优值。
            3. `self.points.grad`将保存目前取得最小`self.poly_result`值的梯度。
        :param weights: 对每个点的多项式平方的加权系数。若为None，则系数设置为全1。
        :return: None
        """
        if weights is None:
            weights = torch.ones(self.points.shape[0])
        else:
            assert weights.shape == (self.points.shape[0],)
        min_driver = Minimizer(self, delta=1e-4, weights=weights)
        min_driver.minimize()  # 该操作将满足上述三条副作用

    def get_points_weights(self) -> torch.Tensor:
        """
        从点集`self.points`中提取梯度（`self.points.grad`）并计算权重。
        :return: torch.Tensor, shape为(N,)，也即平面要拟合的点集个数。
        """
        pts_grad: torch.Tensor = self.points.grad
        assert pts_grad is not None and not (torch.zeros_like(pts_grad) == pts_grad).all()  # 梯度应当非0
        result: torch.Tensor = pts_grad[:, 0] ** 2 + pts_grad[:, 1] ** 2
        assert result.shape == (self.points.shape[0],)
        return 1.0 / result  # assign to each data point p0 a weight "1.0 / gradient^2(x0, y0)"

    def run(self, max_iter_times: int = 10) -> None:
        """
        实际运行多项式拟合工作。
        :param max_iter_times: 最大迭代次数。这里的迭代次数是指“大”迭代次数，即一次迭代意味着给多项式换了一次加权。每一个“大”迭代下有很多“小”迭代。
        :return: None
        """
        self.time_cost = time.time()
        assert max_iter_times >= 1
        # 下面这一行由`class Minimizer`执行，故不再需要。
        # self.evaluate(save=True, weights=None)  # 计算 sum p^2(x0, y0)，并保存结果；无加权
        self.minimize_arguments(weights=None)  # 第一次最小化 sum p^2(x0, y0)，不需要加权
        poly_last: float = self.poly_result.item()  # `self.minimize_arguments()`保证更新`self.poly_result`
        print(f'First `poly_last`: {poly_last}')
        assert 0.0 < poly_last < float('inf')
        points_weights: torch.Tensor = self.get_points_weights()  # 计算每个点的权重矩阵
        print(f'points_weight [:5]: {points_weights[:5]}')
        # 接下来循环优化加权多项式

        while self.iter_time != max_iter_times:
            self.iter_time += 1
            print(f'----------------------------------\nBig iteration epoch: {self.iter_time}')
            # 每次迭代要优化的都是一个全新的加权多项式，之前计算出的最小值已经没有意义
            self.poly_result: torch.Tensor = torch.tensor(float('inf'))
            self.clear_points_gradients()  # 初始化，清空上一步运算的梯度（因为已经被`points_weights`变量保存了）
            self.minimize_arguments(weights=points_weights)
            self.clear_points_gradients()

            poly_orig_val: torch.Tensor = self.evaluate_origin()  # 最终要优化的多项式
            if poly_orig_val.item() > self.orig_poly_result.item():
                print('Warning: `poly_orig_val` not decreasing.')
                break
            elif self.orig_poly_result.item() - poly_orig_val.item() > self.delta:
                self.orig_poly_result = poly_orig_val
            else:
                break

            print(f'poly_orig_val: {poly_orig_val}')
            points_weights: torch.Tensor = self.get_points_weights()  # 保存以供下一步计算
            print(f'points_weight [:5]: {points_weights[:5]}')
            poly_now: float = self.poly_result.item()
            print(f'poly_now: {poly_now}')
            if poly_now == poly_last:
                print(f'Warning: frozen big iteration epoch {self.iter_time}')
            # if poly_last - poly_now < self.delta:
            #     break
            poly_last = poly_now
        self.time_cost = time.time() - self.time_cost

    def summarize(self) -> None:
        """
        拟合结束后，输出详细拟合结果。
        :return: None
        """
        # val: float = self.poly_result.item()
        self.clear_points_gradients()
        val: float = self.evaluate_origin().item()
        print(f'----------------------------------\n原多项式误差: {val}')
        print(f'原多项式平均(每点)误差: {val / self.points.shape[0]}')
        print(f'时间总消耗: {self.time_cost:.4f}s')
        print(f'迭代总次数: {self.iter_time}')

        a40: float = self.B[0, 0] + self.eps
        a31: float = 2.0 * self.B[0, 1]
        a22: float = 2.0 * self.B[0, 2] + self.B[1, 1] + self.eps
        a13: float = 2.0 * self.B[1, 2]
        a04: float = self.B[2, 2] + self.eps
        args: torch.Tensor = torch.empty(15)
        args[0:5] = torch.tensor([a40, a31, a22, a13, a04])
        args[5:] = self.v3
        coe_list: List = args.tolist()
        print(f'系数列表: {coe_list}')

        # 绘制肝脏像素
        points_x: List[float] = self.points[:, 0].detach().numpy()
        points_y: List[float] = self.points[:, 1].detach().numpy()
        plt.scatter(points_x, points_y, s=0.5, c='red')

        # 绘制拟合曲线像素
        plt.xlim((-50.0, 50.0))
        plt.ylim((-50.0, 50.0))
        delta: float = 0.025
        xrange: np.ndarray = np.arange(-50.0, 50.0, delta)
        yrange: np.ndarray = np.arange(-50.0, 50.0, delta)
        x, y = np.meshgrid(xrange, yrange)
        fn: np.ndarray = np.zeros_like(x)
        seq: int = 0
        for index in range(4, -1, -1):
            for x_index in range(index, -1, -1):
                y_index: int = index - x_index
                fn += coe_list[seq] * (x ** x_index) * (y ** y_index)
                seq += 1
        assert seq == 15
        plt.contour(x, y, fn, [0])
        plt.show()


class Minimizer:
    Golden: float = (1.0 + math.sqrt(5)) / 2.0  # 1.618033988749895
    GoldenBigPart: float = Golden - 1.0  # 0.618033988749895
    GoldenSmallPart: float = 1 - GoldenBigPart  # 0.3819660112501051

    def __init__(self, poly4: Poly4, delta: float, weights: torch.Tensor):
        """
        初始化拟合优化驱动。该类的生命周期仅限于某一次大迭代下。
        :param poly4: 多项式参数。
        :param delta: 当某轮小迭代中，函数值的下降幅度小于该值时，迭代终止。
        :param weights: 权重向量。
        """
        self.poly4: Poly4 = poly4
        self.delta: float = delta  # ftol
        self.weights: torch.Tensor = weights
        self.poly_res_cache: torch.Tensor = self.poly4.evaluate(save=True, weights=weights)
        print(f'Minimizer.poly_res_cache(init poly value): {self.poly_res_cache.item()}')
        self.temp_B: torch.Tensor = poly4.B.clone()
        self.temp_v3: torch.Tensor = poly4.v3.clone()
        self.iter_time: int = 0  # 小迭代次数

    def minimize(self) -> None:
        """
        最小化多项式值（即最优化多项式参数）。
        :return: None
        """
        self.powell_fit()

    @staticmethod
    def _grad_none_or_zeros(t: torch.Tensor) -> bool:
        return t.grad is None or (t.grad == torch.zeros_like(t.grad)).all()

    @staticmethod
    def _sign(val: float, sign: float) -> float:
        return abs(val) if sign >= 0.0 else -abs(val)

    def powell_fit(self) -> None:
        """
        使用powell方法进行曲线拟合。
        :return: None
        """
        # 16, or x.shape[0] -> n
        # float *fret -> self.poly_res_cache
        # float (*func)(float []) -> self.evaluate_serialize()

        x: torch.Tensor = self.serialize_coefficients()  # float p[], shape (16,)
        n: int = x.shape[0]
        assert x.shape == (16,)
        directions: torch.Tensor = torch.eye(x.shape[0])  # float xi[]
        x_cur: torch.Tensor = x.clone()  # pt[]
        fn_res: torch.Tensor = self.poly_res_cache  # *fret
        fn_val_local: torch.Tensor  # fptt
        mean_direction: torch.Tensor  # xit

        while True:
            self.iter_time += 1
            print(f'\tSmall iteration epoch: {self.iter_time}')
            if self.iter_time == 1000:
                print(f'Warning: too many iterations(>= 1000) in class `Minimizer`')
            fn_val_global: torch.Tensor = fn_res  # fp
            biggest_column: int = -1  # ibig
            local_delta: float = 0.0  # del
            for i in range(n):
                fn_val_local: torch.Tensor = fn_val_global  # fptt
                # directions[:, i] -> xit[]
                x, mean_direction, fn_res = self.linear_min_val_searching(x, directions[:, i])
                if fn_val_local - fn_res > local_delta:
                    local_delta: float = fn_val_local - fn_res
                    biggest_column = i
            if 2.0 * (fn_val_global - fn_res) <= self.delta * (abs(fn_val_global) + abs(fn_res) + 1e-20):
                assert fn_res.requires_grad
                assert Minimizer._grad_none_or_zeros(self.poly4.points)
                self.poly4.poly_result = fn_res  # 实现`Poly4.minimize_arguments`副作用1
                self.poly4.B, self.poly4.v3 = Minimizer.deserialize(x)  # 实现`Poly4.minimize_arguments`副作用2
                # self.poly4.poly_result.backward()  # 实现`Poly4.minimize_arguments`副作用3
                self.poly4.evaluate_origin()  # 对新多项式进行backward，实现`Poly4.minimize_arguments`副作用3
                return
            if self.iter_time == 2000:
                raise ValueError("Too many iterations in method `Minimizer.powell_fit()`")
            extrapolation: torch.Tensor = 2.0 * x - x_cur  # ptt
            mean_direction = x - x_cur  # xit
            x_cur = x
            assert extrapolation.shape == (n,)
            fn_val_local = self.evaluate_serialize(
                ser_coe=extrapolation, save=False)
            if fn_val_local.item() < fn_val_global.item():
                fvg_f: float = fn_val_global.item()
                fr_f: float = fn_res.item()
                fvl_f: float = fn_val_local.item()
                t: float = 2.0 * (fvg_f - 2.0 * fr_f + fvl_f) * (fvg_f - fr_f - local_delta) ** 2 - local_delta * (
                        fvg_f - fvl_f) ** 2
                if t < 0.0:
                    x, mean_direction, fn_res = self.linear_min_val_searching(x, mean_direction)
                    directions[:, biggest_column] = directions[:, -1]
                    directions[:, -1] = mean_direction
            print(f'\t\tfn_res: {fn_res}')

    def serialize_coefficients(self) -> torch.Tensor:
        """
        将`self.temp_B`, `self.temp_v3`序列化为向量参数，以便于`self.powell_fit()`方法存储方向矩阵。
        :return: 16维向量`torch.Tensor`.
        """
        # -----------------------------------
        #           vec0 vec1 vec2
        # B的布局: [ vec1 vec3 vec4 ]
        #           vec2 vec4 vec5
        # -----------------------------------
        # v3的布局: (vec6, vec7, ..., vec15)
        # -----------------------------------
        vec: torch.Tensor = torch.empty(16)
        vec[0:3] = self.temp_B[0, :]
        vec[3:5] = self.temp_B[1, 1:]
        vec[5] = self.temp_B[-1, -1]
        vec[6:] = self.temp_v3
        return vec

    def evaluate_serialize(self, ser_coe: torch.Tensor, save: bool) -> torch.Tensor:
        """
        计算多项式的值，但使用临时提供的序列化向量参数而不是`self.poly4.{B, v3}`。
        :param ser_coe: 临时序列化向量，由`self.serialize_coefficients()`得到。
        :param save: 是否将计算结果保存到`self.poly4.poly_result`中。
        :return: 计算结果；单元素`torch.Tensor`
        """
        return self.poly4.evaluate(save, weights=self.weights, replaced_by=self.deserialize(ser_coe))

    @staticmethod
    def deserialize(coefficient: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """
        将序列化向量解析为主要项矩阵B和非主要项向量v3。
        :param coefficient: 序列化向量。
        :return: (B, v3).
        """
        # -----------------------------------
        #           vec0 vec1 vec2
        # B的布局: [ vec1 vec3 vec4 ]
        #           vec2 vec4 vec5
        # -----------------------------------
        # v3的布局: (vec6, vec7, ..., vec15)
        # -----------------------------------
        assert coefficient.shape == (16,)
        b: torch.Tensor = torch.empty(3, 3)
        b[0, :] = coefficient[0:3]
        b[1, 1:] = coefficient[3:5]
        b[2, 2] = coefficient[5]
        b[1, 0] = b[0, 1]
        b[2, 0:2] = b[0:2, 2]
        v3: torch.Tensor = coefficient[6:]
        assert v3.shape == (10,)
        return b, v3

    def linear_min_val_searching(self, x: torch.Tensor, direction: torch.Tensor) -> Tuple[torch.Tensor,
                                                                                          torch.Tensor, torch.Tensor]:
        """
        以x为基准点，direction为基方向探索函数的最小值。
        :param x: 基准点。
        :param direction: 探索方向（包括正向和反向）。
        :return: 三元元组，均为`torch.Tensor`：(取得极小值的点, 基准点移动的位移, 极小值)。
        """
        direction_len: torch.Tensor = direction.norm()
        assert direction_len > 1e-8
        direction_unit_vec: torch.Tensor = direction / direction_len
        assert abs(direction_unit_vec.norm() - 1.0) < 1e-3
        x_left: torch.Tensor = x - 1e-1 * direction_unit_vec
        x_right: torch.Tensor = x + 1e-1 * direction_unit_vec
        package: Tuple[torch.Tensor, torch.Tensor, torch.Tensor] = self.get_rough_interval(x_left, x_right)
        # print(package)
        min_pos, min_val = self.search_minimal(*package, method='golden')
        vec: torch.Tensor = min_pos - x
        return min_pos, vec, min_val

    def get_rough_interval(self, ax: torch.Tensor, bx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        从一个区间（ax->bx或者bx->ax）内搜索函数极小值。
        :param ax: 区间一端。
        :param bx: 区间另一端。
        :return: 三元元组，内部元素都是`torch.Tensor`。值为(新ax, 新bx, cx). 由新ax、新bx、cx确定的新区间能划界该函数的极小值。
        """
        fa: torch.Tensor = self.evaluate_serialize(ser_coe=ax, save=False)
        fb: torch.Tensor = self.evaluate_serialize(ser_coe=bx, save=False)
        if fb.item() > fa.item():
            ax, bx = bx, ax
            fa, fb = fb, fa
        cx: torch.Tensor = bx + Minimizer.Golden * (bx - ax)
        fc: torch.Tensor = self.evaluate_serialize(ser_coe=cx, save=False)
        fu: torch.Tensor
        while fb.item() > fc.item():
            r: float = (bx - ax).norm() * (fb.item() - fc.item())
            q: float = (bx - cx).norm() * (fb.item() - fa.item())
            u: torch.Tensor = bx - ((bx - cx) * q - (bx - ax) * r) / (
                    2.0 * Minimizer._sign(max(abs(q - r), 1e-20), q - r))
            threshold: torch.Tensor = bx + 100.0 * (cx - bx)  # ulim
            if torch.dot(bx - u, u - cx).item() > 0.0:
                fu = self.evaluate_serialize(ser_coe=u, save=False)
                if fu.item() < fc.item():
                    ax = bx
                    bx = u
                    break
                elif fu.item() > fb.item():
                    cx = u
                    break
                else:
                    u = cx + Minimizer.Golden * (cx - bx)
                    fu = self.evaluate_serialize(ser_coe=u, save=False)
            elif torch.dot(cx - u, u - threshold).item() > 0.0:
                fu = self.evaluate_serialize(ser_coe=u, save=False)
                if fu.item() < fc.item():
                    bx = cx
                    cx = u
                    u = cx + Minimizer.Golden * (cx - bx)

                    fb = fc
                    fc = fu
                    fu = self.evaluate_serialize(ser_coe=u, save=False)
            elif torch.dot(u - threshold, threshold - cx).item() >= 0.0:
                u = threshold
                fu = self.evaluate_serialize(ser_coe=u, save=False)
            else:
                u = cx + Minimizer.Golden * (cx - bx)
                fu = self.evaluate_serialize(ser_coe=u, save=False)
            ax = bx
            bx = cx
            cx = u

            fa = fb
            fb = fc
            fc = fu
        return ax, bx, cx

    def search_minimal(self, ax: torch.Tensor, bx: torch.Tensor, cx: torch.Tensor, method: str = 'golden') \
            -> Tuple[torch.Tensor, torch.Tensor]:
        match method:
            case 'golden':
                func: Callable = Minimizer._golden_search
            case 'brent':
                func: Callable = Minimizer._brent_method
            case _:
                raise ValueError(f'Unexpected argument `method` = {method}')
        return func(self, ax, bx, cx)

    def _golden_search(self, ax: torch.Tensor, bx: torch.Tensor, cx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        给定三个点的坐标ax, bx, cx（满足bx在ax和cx之间，f(bx) < f(ax), f(bx) < f(cx)），使用黄金分割方法搜索函数极小值点。
        :param ax: 左坐标。
        :param bx: 中坐标。
        :param cx: 右坐标。
        :return: 二元元组，均为`torch.Tensor`：(取得极小值的坐标, 极小值).
        """
        origin: torch.Tensor = ax.clone()  # 以"原点"坐标
        x0: torch.Tensor = ax.clone()
        x3: torch.Tensor = cx.clone()
        if (cx - bx).norm() > (bx - ax).norm():
            x1: torch.Tensor = bx.clone()
            x2: torch.Tensor = bx + Minimizer.GoldenSmallPart * (cx - bx)
        else:
            x2: torch.Tensor = bx.clone()
            x1: torch.Tensor = bx + Minimizer.GoldenSmallPart * (cx - bx)
        f1: torch.Tensor = self.evaluate_serialize(ser_coe=x1, save=False)
        f2: torch.Tensor = self.evaluate_serialize(ser_coe=x2, save=False)
        delta0: float = 2e-4

        while (x3 - x0).norm() > delta0 * ((x1 - origin).norm() + (x2 - origin).norm()):
            if f2.item() < f1.item():
                x0 = x1
                x1 = x2
                x2 = Minimizer.GoldenBigPart * x1 + Minimizer.GoldenSmallPart * x3
                f1 = f2
                f2 = self.evaluate_serialize(ser_coe=x2, save=False)
            else:
                x3 = x2
                x2 = x1
                x1 = Minimizer.GoldenBigPart * x2 + Minimizer.GoldenSmallPart * x0
                f2 = f1
                f1 = self.evaluate_serialize(ser_coe=x1, save=False)
        if f1.item() < f2.item():
            return x1, f1
        else:
            return x2, f2

    def _brent_method(self, ax: torch.Tensor, bx: torch.Tensor, cx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        给定三个点的坐标ax, bx, cx（满足bx在ax和cx之间，f(bx) < f(ax), f(bx) < f(cx)），使用brent方法搜索函数极小值点。
        :param ax: 左坐标。
        :param bx: 中坐标。
        :param cx: 右坐标。
        :return: 二元元组，均为`torch.Tensor`：(取得极小值的坐标, 极小值).
        """
        # small_iter_max: int = 100
        # e: float = 0.0
        # # 方向：ax->bx->cx
        # a: torch.Tensor = ax.clone()
        # b: torch.Tensor = cx.clone()
        #
        # x: torch.Tensor = bx.clone()
        # w: torch.Tensor = bx.clone()
        # v: torch.Tensor = bx.clone()
        #
        # fx: torch.Tensor = self.evaluate_serialize(ser_coe=x, save=False)  # single value
        # fw: torch.Tensor = fx.clone()
        # fv: torch.Tensor = fx.clone()
        #
        # delta0: float = 2e-4  # tol
        # for _ in range(small_iter_max):
        #     xm: torch.Tensor = a + 0.5 * (b - a)
        raise NotImplementedError()


if __name__ == '__main__':
    from typing import List, Tuple
    import json

    with open('../../data/examples10_100x100.json', 'r', encoding='utf-8') as f:
        example10: List[List[Tuple[float, float]]] = json.load(f)
        ex0 = example10[2]
        n0: np.ndarray = np.array(ex0)
        nm: np.ndarray = np.mean(n0, axis=0)
        ex0 = (n0 - nm).tolist()
        poly4: Poly4 = Poly4(torch.tensor(ex0))
        poly4.run()
        poly4.summarize()
