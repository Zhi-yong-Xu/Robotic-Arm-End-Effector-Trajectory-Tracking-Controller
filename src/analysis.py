"""数据记录与结果绘图。"""
import numpy as np
import matplotlib.pyplot as plt

from .utils import setup_chinese_plot


class Recorder:
    """按固定间隔采样跟踪与观测数据, 仿真结束后绘图并打印统计。"""

    def __init__(self, sample_every: int = 10):
        self.sample_every = sample_every
        self._t = []
        self.target_pos, self.actual_pos = [], []
        self.target_euler, self.actual_euler = [], []
        self.pos_err = []
        self.twist_est, self.twist_true = [], []

    def maybe_record(self, step, t, traj, p_cur, q_euler, err_p,
                     twist_est, twist_true):
        """满足采样间隔时记录一帧数据。"""
        if step % self.sample_every != 0:
            return
        self._t.append(t)
        self.target_pos.append(traj.pos)
        self.actual_pos.append(p_cur)
        self.target_euler.append(traj.euler if traj.euler is not None
                                 else np.full(3, np.nan))
        self.actual_euler.append(q_euler)
        self.pos_err.append(err_p)
        self.twist_est.append(twist_est)
        self.twist_true.append(twist_true if twist_true is not None
                               else np.full(6, np.nan))

    @property
    def empty(self):
        """是否未记录任何数据。"""
        return len(self._t) == 0

    def plot(self):
        """绘制位置/姿态跟踪、误差范数与 FTDSO 速度估计对比图。"""
        setup_chinese_plot()
        t = np.array(self._t)
        tp, ap = np.array(self.target_pos), np.array(self.actual_pos)
        te, ae = np.array(self.target_euler), np.array(self.actual_euler)
        pe_norm = np.linalg.norm(np.array(self.pos_err), axis=1)
        we, wt = np.array(self.twist_est), np.array(self.twist_true)

        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        fig.suptitle("末端位姿跟踪 + FTDSO 轨迹微分观测", fontsize=15, fontweight="bold")

        for i, (lb, c) in enumerate(zip("XYZ", "rgb")):
            axes[0, 0].plot(t, tp[:, i], c + "--", alpha=0.6, label=f"期望 {lb}")
            axes[0, 0].plot(t, ap[:, i], c + "-", label=f"实际 {lb}")
        axes[0, 0].set_title("位置跟踪")
        axes[0, 0].set_ylabel("位置 (m)")
        axes[0, 0].legend(fontsize="small"); axes[0, 0].grid(alpha=0.3)

        axes[0, 1].plot(t, pe_norm, "k-")
        axes[0, 1].axhline(0.01, color="r", ls="--", alpha=0.5, label="1cm")
        axes[0, 1].set_title("位置误差范数")
        axes[0, 1].set_ylabel("误差 (m)")
        axes[0, 1].legend(); axes[0, 1].grid(alpha=0.3)

        for i, (lb, c) in enumerate(zip(["Roll", "Pitch", "Yaw"], "mcy")):
            axes[1, 0].plot(t, te[:, i], c + "--", alpha=0.6, label=f"期望 {lb}")
            axes[1, 0].plot(t, ae[:, i], c + "-", label=f"实际 {lb}")
        axes[1, 0].set_title("姿态跟踪 (欧拉角)")
        axes[1, 0].set_xlabel("时间 (s)")
        axes[1, 0].legend(fontsize="small"); axes[1, 0].grid(alpha=0.3)

        for i, lb in enumerate(["vx", "vy", "vz"]):
            axes[1, 1].plot(t, wt[:, i], "-", alpha=0.6, label=f"真值 {lb}")
            axes[1, 1].plot(t, we[:, i], "--", label=f"FTDSO {lb}")
        axes[1, 1].set_title("FTDSO 线速度估计 vs 解析真值")
        axes[1, 1].set_xlabel("时间 (s)")
        axes[1, 1].set_ylabel("线速度 (m/s)")
        axes[1, 1].legend(fontsize="small"); axes[1, 1].grid(alpha=0.3)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        valid = ~np.isnan(wt[:, 0])
        if valid.any():
            rmse = np.sqrt(np.mean((we[valid, :3] - wt[valid, :3]) ** 2))
            print(f"\nFTDSO 线速度估计 RMSE: {rmse:.4f} m/s")
        print(f"位置误差  均值: {np.nanmean(pe_norm)*1000:.3f} mm, "
              f"最大: {np.nanmax(pe_norm)*1000:.3f} mm")
        plt.show()
