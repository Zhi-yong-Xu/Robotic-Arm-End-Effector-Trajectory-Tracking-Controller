"""动力学控制器: 关节速度/加速度 -> 关节力矩, 提供 PD / DSC / 纯补偿三种实现。"""
import numpy as np

from .registry import dynamic_registry


class BaseDynamicController:
    """动力学控制器基类, compute 输入期望关节速度, 输出关节力矩。"""

    def __init__(self, robot, cfg: dict):
        self.robot = robot
        self.cfg = cfg

    def compute(self, dq_des) -> np.ndarray:
        raise NotImplementedError


@dynamic_registry.register("pd")
class JointPDController(BaseDynamicController):
    """关节速度 PD 伺服: tau = Kp(dq_des - dq) + bias - passive, 无加速度前馈。"""

    def __init__(self, robot, cfg: dict):
        super().__init__(robot, cfg)
        self.kp = cfg["kp"]
        self.tau_limit = cfg.get("tau_limit")

    def compute(self, dq_des) -> np.ndarray:
        """PD 速度误差项 + 重力/阻尼补偿, 可选力矩限幅。"""
        r = self.robot
        tau = self.kp * (dq_des - r.qvel_arm) + r.bias_torque - r.passive_torque
        if self.tau_limit:
            tau = np.clip(tau, -self.tau_limit, self.tau_limit)
        return tau


@dynamic_registry.register("dsc")
class DSCController(BaseDynamicController):
    """动态面控制: tau = bias - passive + M(α̇ + Kd(dq_des - dq))。

    dq_des 经一阶滤波得面信号 α (α̇ = k_f(dq_des - α)),
    以滤波导数 α̇ 代替解析 dq̈_d, 避免雅可比导数的项数爆炸;
    级联误差系统 (s, r) 指数稳定, 跟踪误差界 O(||dq̈_d||/k_f)。
    """

    def __init__(self, robot, cfg: dict):
        super().__init__(robot, cfg)
        self.dt = robot.dt
        self.filter_gain = cfg["filter_gain"]
        self.kd = cfg["kd"]
        self.tau_limit = cfg.get("tau_limit")
        self.alpha = np.zeros(robot.n_arm)

    def compute(self, dq_des) -> np.ndarray:
        """滤波面更新 + 惯性前馈 + 速度阻尼, 输出限幅后的关节力矩。"""
        r = self.robot
        tau = r.bias_torque - r.passive_torque

        d_alpha = self.filter_gain * (np.asarray(dq_des) - self.alpha)
        self.alpha += d_alpha * self.dt

        tau += r.arm_mass_matrix() @ (d_alpha + self.kd * (dq_des - r.qvel_arm))

        if self.tau_limit:
            tau = np.clip(tau, -self.tau_limit, self.tau_limit)
        return tau


