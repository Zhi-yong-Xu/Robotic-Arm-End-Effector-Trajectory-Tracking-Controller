"""运动学控制器: 期望位姿 + 参考twist -> 期望关节速度 (速度级逆解)。
约定: 所有误差与 twist 均在世界系表达, 与几何雅可比的行序 [v; w] 一致。"""
import mujoco
import numpy as np

from .registry import kinematic_registry


class BaseKinematicController:
    """运动学控制器基类。"""

    def __init__(self, robot, cfg: dict):
        self.robot = robot
        self.cfg = cfg

    def compute(self, traj, twist_ref) -> dict:
        raise NotImplementedError


def orientation_error_world(quat_des: np.ndarray,
                            quat_cur: np.ndarray) -> np.ndarray:
    """世界系姿态误差: 把当前姿态转到期望姿态所需的世界系角速度向量。

    e = log(q_des ⊗ q_cur⁻¹),  施加 ω = k·e 后姿态按 ė ≈ −k·e 收敛。
    注意: mju_subQuat(qa, qb) 返回 qb 局部系下的误差, 不能直接与世界系
    雅可比相乘; 此处显式在世界系构造, 姿态非零时方向才正确。
    """
    q_cur_inv = np.zeros(4)
    q_err = np.zeros(4)
    mujoco.mju_negQuat(q_cur_inv, quat_cur)
    mujoco.mju_mulQuat(q_err, quat_des, q_cur_inv)   # q_err = q_des ⊗ q_cur⁻¹
    e_world = np.zeros(3)
    mujoco.mju_quat2Vel(e_world, q_err, 1)
    return e_world


@kinematic_registry.register("velocity_pd")
class VelocityPDController(BaseKinematicController):
    """速度级逆解 + 误差反馈: J·dq = twist_ref + Kp·[位置误差; 姿态误差]。

    位置/姿态误差均在世界系; 关节限速采用等比缩放以保持末端方向。
    """

    def __init__(self, robot, cfg: dict):
        super().__init__(robot, cfg)
        self.kp = cfg["kp"]
        self.ee_vel_limit = cfg.get("ee_vel_limit")  
        self.twist_cmd_raw = np.zeros(6)   # 限幅前的指令 (看超限多少)
        self.twist_cmd = np.zeros(6)       # 实际送入逆解的指令
        self.clipped = False               # 本步是否触发限幅

        self.dls_lambda = float(cfg.get("dls_lambda", 0.0))
        self.dls_eps = float(cfg.get("dls_eps", 1e-2))
        self.last_scale = 1.0

        # --- 新增: 零空间奇异规避参数 ---
        self.k_manip = float(cfg.get("k_manip", 0.3))   # 梯度步长, 0 即关闭
        self.h = float(cfg.get("fd_step", 1e-5))        # 差分步长
        self.last_scale = 1.0
        self.manipulability = 0.0                        # 诊断量

    # ---------- 新增: 可操作度及其关节梯度 ----------
    def _manipulability(self) -> float:
        J = self.robot.ee_jacobian()
        return float(np.sqrt(max(np.linalg.det(J @ J.T), 0.0)))

    def _manipulability_grad(self) -> np.ndarray:
        """∂w/∂q, 中心差分, 梯度归一化 (方向驱动, 步长由 k_manip 定)。"""
        r = self.robot
        adr, n, h = r.qpos_adr, r.n_arm, self.h
        q0 = r.data.qpos[adr].copy()
        g = np.zeros(n)
        try:
            for i in range(n):
                for s in (+1, -1):
                    r.data.qpos[adr] = q0
                    r.data.qpos[adr[i]] += s * h
                    mujoco.mj_kinematics(r.model, r.data)
                    mujoco.mj_comPos(r.model, r.data)
                    g[i] += s * self._manipulability()
            g /= (2.0 * h)
        finally:
            r.data.qpos[adr] = q0
            mujoco.mj_kinematics(r.model, r.data)
            mujoco.mj_comPos(r.model, r.data)
        norm = np.linalg.norm(g)
        return g / norm if norm > 1e-12 else g
    
    def compute(self, traj, twist_ref) -> dict:
        """解算期望关节速度, 返回 dq 与误差诊断量。"""
        r = self.robot
        p_cur, q_cur = r.ee_pose()

        err_p = traj.pos - p_cur
        err_R = orientation_error_world(traj.quat, q_cur)
        err6 = np.concatenate([err_p, err_R])

        twist = np.asarray(twist_ref, dtype=float) + self.kp * err6

        # ---------- 末端速度指令限幅: 直接限制指令, 之后正常求解 ----------
        self.clipped = False
        self.twist_cmd_raw = twist.copy()
        if self.ee_vel_limit is not None:
            v_max, w_max = float(self.ee_vel_limit[0]), float(self.ee_vel_limit[1])
            nv = np.linalg.norm(twist[:3])
            nw = np.linalg.norm(twist[3:])
            if nv > v_max:
                twist[:3] *= v_max / nv
                self.clipped = True
            if nw > w_max:
                twist[3:] *= w_max / nw
                self.clipped = True
        self.twist_cmd = twist


        J = r.ee_jacobian()

        # ---------- DLS 解算: 一次求 J_pinv, 主任务与零空间共用 ----------
        if self.dls_lambda > 0.0:
            # 自适应阻尼: 只在接近奇异时启用
            s = np.linalg.svd(J, compute_uv=False)      # 降序奇异值
            sigma_min = float(s[-1])
            if sigma_min < self.dls_eps:
                ratio = sigma_min / self.dls_eps
                lam2 = self.dls_lambda ** 2 * (1.0 - ratio * ratio)
            else:
                lam2 = 0.0
            # 右端用 6x6 单位阵: 一次 LU 分解同时得到伪逆的全部列
            J_pinv = J.T @ np.linalg.solve(J @ J.T + lam2 * np.eye(6), np.eye(6))
        else:
            lam2 = 0.0
            J_pinv = np.linalg.pinv(J)

        # 主任务: 与 solve(向量) 版本数值等价
        dq = J_pinv @ twist

        # ---------- 零空间奇异规避 ----------
        if self.k_manip > 0.0:
            g = self._manipulability_grad()
            N = np.eye(r.n_arm) - J_pinv @ J          # DLS 零空间投影
            dq = dq + self.k_manip * (N @ g)          # 投影后注入, 不扰动末端
        self.manipulability = self._manipulability()



        return {"dq": dq, "err_p": err_p, "err_R": err_R}
