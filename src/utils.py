"""数学工具: 幂次函数、四元数/欧拉角转换、球面插值。"""
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation, Slerp


def sig(x, p: float) -> np.ndarray:
    """连续幂次函数 sig(x) = |x|^p * sign(x), 用于有限时间收敛律。"""
    x = np.asarray(x, dtype=float)
    return np.sign(x) * np.abs(x) ** p


def euler_to_quat(euler, seq: str = "xyz") -> np.ndarray:
    """xyz 欧拉角转四元数 (w,x,y,z)。"""
    e = np.asarray(euler, dtype=float).reshape(3, 1)
    q = np.zeros(4)
    mujoco.mju_euler2Quat(q, e, seq)
    return q


def quat_to_euler(quat_wxyz) -> np.ndarray:
    """四元数 (w,x,y,z) 转 xyz 欧拉角 (rad)。"""
    w, x, y, z = quat_wxyz
    return Rotation.from_quat([x, y, z, w]).as_euler("XYZ")

def quat_to_R(quat_wxyz: np.ndarray) -> np.ndarray:
    """四元数 -> 3x3 旋转矩阵 (体->世界)。"""
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.asarray(quat_wxyz, dtype=float))
    return R.reshape(3, 3)

def quat_error_local(quat_target, quat_cur) -> np.ndarray:
    """姿态误差, 轴角形式 3 维向量。"""
    err = np.zeros(3)
    mujoco.mju_subQuat(err, quat_target, quat_cur)
    return err


def slerp(quat0_wxyz, quat1_wxyz, u) -> np.ndarray:
    """四元数球面插值, u∈[0,1], 输入输出均为 (w,x,y,z)。"""
    rots = Rotation.from_quat([
        [quat0_wxyz[1], quat0_wxyz[2], quat0_wxyz[3], quat0_wxyz[0]],
        [quat1_wxyz[1], quat1_wxyz[2], quat1_wxyz[3], quat1_wxyz[0]],
    ])
    q = Slerp([0.0, 1.0], rots)([np.clip(u, 0.0, 1.0)]).as_quat()[0]
    return np.array([q[3], q[0], q[1], q[2]])


def quat_omega_fd(quat, quat_next, dt: float) -> np.ndarray:
    """由相邻两拍四元数差分数值计算角速度 (世界系)。"""
    err = np.zeros(3)
    mujoco.mju_subQuat(err, quat_next, quat)
    return quat_to_R(quat) @ err / dt


def setup_chinese_plot():
    """配置 Matplotlib 中文显示。"""
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
