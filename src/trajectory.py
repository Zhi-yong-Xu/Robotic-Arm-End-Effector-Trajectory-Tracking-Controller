"""参考轨迹: 输出各时刻位置/姿态; 速度交由 FTDSO 估计, 解析速度仅作真值对比。"""
import numpy as np

from .registry import trajectory_registry
from .utils import euler_to_quat, slerp, quat_omega_fd


class TrajState:
    """单时刻轨迹状态容器。"""

    def __init__(self, pos, quat, euler, vel_true, omega_true):
        self.pos = pos
        self.quat = quat
        self.euler = euler
        self.vel_true = vel_true
        self.omega_true = omega_true


class BaseTrajectory:
    """轨迹基类, 提供姿态角速度的数值微分辅助。"""

    def sample(self, t: float) -> TrajState:
        """采样 t 时刻的期望位姿与解析速度真值。"""
        raise NotImplementedError

    def sample_path(self, t_end: float, dt: float = 0.1):
        """按固定间隔采样一段位置路径, 用于可视化。"""
        n = int(t_end / dt) + 1
        return [self.sample(i * dt).pos for i in range(n)]

    def _angular_vel(self, t: float, quat: np.ndarray, eps: float = 1e-4) -> np.ndarray:
        """用 t 与 t+eps 两拍四元数差分计算角速度真值。"""
        _, q_next = self._pose_only(t + eps)
        return quat_omega_fd(quat, q_next, eps)

    def _pose_only(self, t: float):
        """仅返回 t 时刻位置与四元数, 供角速度差分内部使用。"""
        raise NotImplementedError


@trajectory_registry.register("circle")
class CircleTrajectory(BaseTrajectory):
    """水平圆周 + Z 向正弦振动, 姿态恒定, 机械臂最常见的标定轨迹。"""

    def __init__(self, cfg: dict):
        self.center = np.asarray(cfg["center"], dtype=float)
        self.radius = float(cfg["radius"])
        self.omega = float(cfg["omega"])
        self.z_amp = float(cfg.get("z_amplitude", 0.1))
        self.z_freq = float(cfg.get("z_freq", 5.0))
        self.euler = np.asarray(cfg.get("euler", [0.0, 0.0, 0.0]), dtype=float)
        self.quat = euler_to_quat(self.euler)

    def _pose_only(self, t):
        th = self.omega * t
        pos = np.array([
            self.center[0] + self.radius * np.cos(th),
            self.center[1] + self.radius * np.sin(th),
            self.center[2] + self.z_amp * np.sin(self.z_freq * th),
        ])
        return pos, self.quat.copy()

    def sample(self, t):
        pos, quat = self._pose_only(t)
        th = self.omega * t
        vel = np.array([
            -self.radius * self.omega * np.sin(th),
            self.radius * self.omega * np.cos(th),
            self.z_amp * self.z_freq * self.omega * np.cos(self.z_freq * th),
        ])
        return TrajState(pos, quat, self.euler.copy(), vel, np.zeros(3))


@trajectory_registry.register("line")
class LineTrajectory(BaseTrajectory):
    """两点间五次多项式直线轨迹 (零起停), 姿态沿四元数球面插值。"""

    def __init__(self, cfg: dict):
        self.p0 = np.asarray(cfg["start"], dtype=float)
        self.p1 = np.asarray(cfg["end"], dtype=float)
        self.T = float(cfg["duration"])
        self.e0 = np.asarray(cfg.get("euler_start", [0.0, 0.0, 0.0]), dtype=float)
        self.e1 = np.asarray(cfg.get("euler_end", self.e0), dtype=float)
        self.q0 = euler_to_quat(self.e0)
        self.q1 = euler_to_quat(self.e1)
        self.loop = cfg.get("loop", True)

    def _s(self, t):
        """归一化时间与五次多项式插值因子及导数。"""
        if self.loop:
            tau = (t % self.T) / self.T
        else:
            tau = min(max(t / self.T, 0.0), 1.0)
        s = 10 * tau ** 3 - 15 * tau ** 4 + 6 * tau ** 5
        ds = (30 * tau ** 2 - 60 * tau ** 3 + 30 * tau ** 4) / self.T
        return tau, s, ds

    def _pose_only(self, t):
        tau, s, _ = self._s(t)
        pos = self.p0 + s * (self.p1 - self.p0)
        quat = slerp(self.q0, self.q1, s)
        return pos, quat

    def sample(self, t):
        pos, quat = self._pose_only(t)
        _, _, ds = self._s(t)
        vel = ds * (self.p1 - self.p0)
        return TrajState(pos, quat, None, vel, self._angular_vel(t, quat))


@trajectory_registry.register("lissajous")
class LissajousTrajectory(BaseTrajectory):
    """利萨如曲线 (频率比 [1,2] 时为 8 字), 常用于考核多轴耦合跟踪。"""

    def __init__(self, cfg: dict):
        self.center = np.asarray(cfg.get("center", [0.3, 0.0, 0.5]), dtype=float)
        self.amp = np.asarray(cfg.get("amplitude", [0.3, 0.3, 0.1]), dtype=float)
        self.freq = np.asarray(cfg.get("freq", [1.0, 2.0, 3.0]), dtype=float)
        self.phase = np.asarray(cfg.get("phase", [0.0, 0.0, 0.0]), dtype=float)
        self.omega = float(cfg.get("omega", 1.0))
        self.euler = np.asarray(cfg.get("euler", [0.0, 0.0, 0.0]), dtype=float)
        self.quat = euler_to_quat(self.euler)

    def _pose_only(self, t):
        th = self.omega * t
        pos = self.center + self.amp * np.sin(self.freq * th + self.phase)
        return pos, self.quat.copy()

    def sample(self, t):
        pos, quat = self._pose_only(t)
        th = self.omega * t
        vel = self.amp * self.freq * self.omega * np.cos(self.freq * th + self.phase)
        return TrajState(pos, quat, self.euler.copy(), vel, np.zeros(3))


@trajectory_registry.register("helix")
class HelixTrajectory(BaseTrajectory):
    """等螺距螺旋线, 用于考核连续大范围位置变化下的跟踪。"""

    def __init__(self, cfg: dict):
        self.center = np.asarray(cfg.get("center", [0.0, 0.0, 0.3]), dtype=float)
        self.radius = float(cfg.get("radius", 0.3))
        self.omega = float(cfg.get("omega", 1.0))
        self.pitch = float(cfg.get("pitch", 0.4))
        self.euler = np.asarray(cfg.get("euler", [0.0, 0.0, 0.0]), dtype=float)
        self.quat = euler_to_quat(self.euler)

    def _pose_only(self, t):
        th = self.omega * t
        pos = np.array([
            self.center[0] + self.radius * np.cos(th),
            self.center[1] + self.radius * np.sin(th),
            self.center[2] + self.pitch * th / (2 * np.pi),
        ])
        return pos, self.quat.copy()

    def sample(self, t):
        pos, quat = self._pose_only(t)
        th = self.omega * t
        vel = np.array([
            -self.radius * self.omega * np.sin(th),
            self.radius * self.omega * np.cos(th),
            self.pitch * self.omega / (2 * np.pi),
        ])
        return TrajState(pos, quat, self.euler.copy(), vel, np.zeros(3))


@trajectory_registry.register("waypoints")
class WaypointTrajectory(BaseTrajectory):
    """多路标点轨迹, 段间五次多项式插值 + 四元数球面插值, 位置姿态同时变化。"""

    def __init__(self, cfg: dict):
        wps = cfg["waypoints"]
        if len(wps) < 2:
            raise ValueError("waypoints 至少需要 2 个路标点")
        self.points = [np.asarray(w[:3], dtype=float) for w in wps]
        self.quats = [euler_to_quat(w[3:6]) for w in wps]
        self.T = float(cfg["segment_duration"])
        self.loop = cfg.get("loop", True)

    def _seg(self, t):
        """返回当前段索引与段内插值因子。"""
        total = self.T * (len(self.points) - 1)
        if self.loop and total > 0:
            t = t % total
        idx = min(int(t / self.T), len(self.points) - 2)
        tau = min(max((t - idx * self.T) / self.T, 0.0), 1.0)
        s = 10 * tau ** 3 - 15 * tau ** 4 + 6 * tau ** 5
        ds = (30 * tau ** 2 - 60 * tau ** 3 + 30 * tau ** 4) / self.T
        return idx, s, ds

    def _pose_only(self, t):
        idx, s, _ = self._seg(t)
        pos = self.points[idx] + s * (self.points[idx + 1] - self.points[idx])
        quat = slerp(self.quats[idx], self.quats[idx + 1], s)
        return pos, quat

    def sample(self, t):
        pos, quat = self._pose_only(t)
        idx, _, ds = self._seg(t)
        vel = ds * (self.points[idx + 1] - self.points[idx])
        return TrajState(pos, quat, None, vel, self._angular_vel(t, quat))
