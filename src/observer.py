import numpy as np
import mujoco
from .registry import observer_registry
from .utils import sig, quat_to_euler ,quat_to_R

class BaseObserver:
    """update(pos, quat) 输入期望位姿 (pos∈R³ 世界系, quat=(w,x,y,z)),
    返回 6 维估计变化率/参考扭 [v_est; w_est] (世界系)。"""
    # [改动2] 契约统一: 所有观测器入口签名一致, 位姿以结构化 (pos, quat) 传入

    def update(self, pos: np.ndarray, quat: np.ndarray,twist_true: np.ndarray = None) -> np.ndarray:
        raise NotImplementedError

    @property
    def estimated_state(self) -> np.ndarray:
        raise NotImplementedError

@observer_registry.register("ftdso")
class FTDSO(BaseObserver):
    """有限时间干扰/微分观测器 (FTDSO) - 6D 版本。

    状态维度: 6, 对应 [pos(3), 欧拉角(3)] 的跟踪状态
    输入 (pos, quat) —— 内部将 quat 转为欧拉角
        update([pos; euler])
    """

    def __init__(self, cfg: dict, dt: float):
        self.K_o1 = cfg["K_o1"]
        self.K_o2 = cfg["K_o2"]
        self.p_o1 = cfg["p_o1"]
        self.p_o2 = cfg["p_o2"]
        self.dim = cfg.get("dim", 6)

        init = np.array(cfg["init"], dtype=float)
        assert init.shape == (self.dim,), \
            f"init 维度应为 {self.dim}, 实际为 {init.shape}"

        self._init = init.copy()
        self.state_hat = init.copy()
        self.dt = dt

    @staticmethod
    def _euler_dot_to_omega(euler: np.ndarray, euler_dot: np.ndarray) -> np.ndarray:
        """xyz 欧拉角导数 -> 世界系角速度 (R = Rx·Ry·Rz 约定)。"""
        a, b = euler[0], euler[1]
        ca, sa, cb, sb = np.cos(a), np.sin(a), np.cos(b), np.sin(b)
        if abs(cb) < 1e-3:                      # 万向锁保护: cb→0 时 T 病态
            cb = np.sign(cb) * 1e-3 if cb != 0 else 1e-3
        T = np.array([
            [1.0, 0.0, sb],
            [0.0, ca,   -cb * sa],
            [0.0, sa,   cb * ca],
        ])
        return T @ euler_dot

    def update(self, pos: np.ndarray, quat: np.ndarray, twist_true: np.ndarray = None) -> np.ndarray:
        qw = np.concatenate([np.asarray(pos, dtype=float),
                             quat_to_euler(quat)])
        e_o = qw[: self.dim] - self.state_hat
        e_o[3:] = (e_o[3:] + np.pi) % (2 * np.pi) - np.pi

        # 有限时间收敛律 (双幂次项)
        d_est = (self.K_o1 * sig(e_o, self.p_o1)
                 + self.K_o2 * sig(e_o, self.p_o2))

        self.state_hat += d_est * self.dt
        d_est = np.concatenate([
            d_est[:3],
            self._euler_dot_to_omega(self.state_hat[3:], d_est[3:]),
        ])
        return d_est

    @property
    def estimated_state(self) -> np.ndarray:
        return self.state_hat.copy()

    @property
    def estimated_vel(self) -> np.ndarray:
        return self.state_hat[:3].copy()

    @property
    def estimated_omega(self) -> np.ndarray:
        return self.state_hat[3:].copy()

    def reset(self):
        self.state_hat = self._init.copy()

@observer_registry.register("none")
class NullObserver(BaseObserver):
    """空观测器: 输出恒为零向量, 用于关闭观测器的对照实验。"""

    def __init__(self, cfg: dict, dt: float):
        self.dim = cfg.get("dim", 6)
        self._zero = np.zeros(self.dim)

    # [改动4] 签名统一为 update(pos, quat)
    def update(self, pos: np.ndarray, quat: np.ndarray, twist_true: np.ndarray = None) -> np.ndarray:
        return self._zero

    @property
    def estimated_state(self) -> np.ndarray:
        return self._zero

    def reset(self):
        pass

@observer_registry.register("se3")
class SE3FixedTimeObserver(BaseObserver):
    """SE(3) 固定时间扭转观测器 (体坐标右乘传播)。

        T̂̇ = T̂[(ξ̂)∧ + (k_x1·sig^{r_x1}(s_x) + k_x2·sig^{r_x2}(s_x))∧]
        ξ̂̇ = k_v1·sig^{r_v1}(s_x) + k_v2·sig^{r_v2}(s_x)
    误差 s_x = log(T̂⁻¹T) 全程 O 系(体坐标); 输出前经伴随变换 Ad_T̂ 转 W 系,
    与速度级逆解的世界系几何雅可比口径一致。

    与 FTDSO 的区别:
        - 位姿不经欧拉角, 无约定/奇异问题;
        - 输出 [v_w; w_w] 是物理定义的扭估计, 而非量测向量逐元素导数。
    """

    def __init__(self, cfg: dict, dt: float):
        self.dt = dt
        # 位姿通道注入增益 (决定 T̂ 追上 T 的速度)
        self.kx1, self.rx1 = cfg["k_x1"], cfg["r_x1"]
        self.kx2, self.rx2 = cfg["k_x2"], cfg["r_x2"]
        # 扭估计增益 (决定 ξ̂ 追上真实 ξ 的速度, 需支配 ‖ξ̇‖∞)
        self.kv1, self.rv1 = cfg["k_v1"], cfg["r_v1"]
        self.kv2, self.rv2 = cfg["k_v2"], cfg["r_v2"]
        self.R_hat = None            # 3x3 位姿估计旋转
        self.p_hat = None            # 3   位姿估计平移
        self.xi_hat = np.zeros(6)    # [v_o; w_o] 体坐标扭估计
        self._first = True

    def reset(self):
        self.R_hat = None
        self.p_hat = None
        self.xi_hat = np.zeros(6)
        self._first = True

    def _error(self, R: np.ndarray, p: np.ndarray) -> np.ndarray:
        """s_x = log(T̂⁻¹T), O 系, 6 维 [平移; 旋转]。"""
        R_rel = self.R_hat.T @ R
        e_p = self.R_hat.T @ (p - self.p_hat)
        e_R = np.zeros(3)
        q_rel = np.zeros(4)
        mujoco.mju_mat2Quat(q_rel, R_rel.flatten())
        mujoco.mju_quat2Vel(e_R, q_rel, 1)
        return np.concatenate([e_p, e_R])

    def _propagate_pose(self, xi: np.ndarray):
        """T̂̇ = T̂[ξ]∧ 离散一步: R̂←R̂·exp(ŵΔt) (右乘!), p̂←p̂+R̂vΔt。"""
        w_o, v_o = xi[3:], xi[:3]
        nw = np.linalg.norm(w_o)
        dq = np.zeros(4)
        if nw > 1e-12:
            mujoco.mju_axisAngle2Quat(dq, w_o / nw, nw * self.dt)
        else:
            dq[:] = [1.0, 0.0, 0.0, 0.0]
        q_hat = np.zeros(4)
        mujoco.mju_mat2Quat(q_hat, self.R_hat.flatten())
        q_new = np.zeros(4)
        mujoco.mju_mulQuat(q_new, q_hat, dq)      # 右乘: q̂ ⊗ dq, 顺序不可反!
        q_new /= np.linalg.norm(q_new)            # 防漂移归一化
        R_new = np.zeros(9)
        mujoco.mju_quat2Mat(R_new, q_new)
        self.R_hat = R_new.reshape(3, 3)
        self.p_hat = self.p_hat + self.R_hat @ (v_o * self.dt)

    def _to_world_twist(self) -> np.ndarray:
        """伴随变换 Ad_T̂: O 系扭 -> W 系扭。"""
        w_w = self.R_hat @ self.xi_hat[3:]
        v_w = self.R_hat @ self.xi_hat[:3] # + np.cross(self.p_hat, w_w)
        return np.concatenate([v_w, w_w])

    def update(self, pos: np.ndarray, quat: np.ndarray,twist_true: np.ndarray = None) -> np.ndarray:
        """输入期望位姿, 返回世界系参考 twist [v_w; w_w]。"""
        R = quat_to_R(quat)
        p = np.asarray(pos, dtype=float)

        if self._first:                # 首拍初始化 T̂(0)=T(0), 消除初始瞬态
            self.R_hat = R.copy()
            self.p_hat = p.copy()
            self._first = False
            return np.zeros(6)

        s_x = self._error(R, p)
        u = (self.kx1 * sig(s_x, self.rx1) + self.kx2 * sig(s_x, self.rx2))
        d_xi = (self.kv1 * sig(s_x, self.rv1) + self.kv2 * sig(s_x, self.rv2))

        self._propagate_pose(self.xi_hat + u)       # T̂ 用 ξ̂+u 传播
        self.xi_hat = self.xi_hat + d_xi * self.dt  # ξ̂ 用 kv 项更新

        return self._to_world_twist()

    @property
    def estimated_state(self) -> np.ndarray:
        """返回 6 维世界系扭估计 [v_w; w_w] (由估计位姿作伴随变换)。"""
        return self._to_world_twist()

@observer_registry.register("passthrough")
class PassthroughObserver(BaseObserver):
    """期望速度直通观测器: 不做任何估计, 直接转发轨迹解析真值 [v_true; w_true] (世界系)。
    """

    def __init__(self, cfg: dict, dt: float):
        self._out = np.zeros(6)          # [v_w; w_w], 与基类契约同序

    def update(self, pos: np.ndarray, quat: np.ndarray,
               twist_true: np.ndarray = None) -> np.ndarray:
        if twist_true is not None:
            t = np.asarray(twist_true, dtype=float)
            if t.shape == (6,):
                self._out = t.copy()
        return self._out.copy()          # 返回副本, 防外部误改内部状态

    @property
    def estimated_state(self) -> np.ndarray:
        return self._out.copy()

    def reset(self):
        pass                             # 无内部状态, 无需重置