"""仿真调度器: 组装组件并运行主循环。

每步流程:
  1. 轨迹采样 -> pos_des, quat_des (不含速度)
  2. FTDSO: 量测 [pos_des; euler_des] -> twist_ref = [v_ref; w_ref]
  3. 运动学控制器: twist_ref + Kp·误差 -> dq_des (含关节限速)
  4. 动力学控制器: dq_des -> tau
  5. 步进
"""
import copy

import mujoco
import mujoco.viewer
import numpy as np

from .registry import (
    robot_registry, trajectory_registry, observer_registry,
    kinematic_registry, dynamic_registry,
)
from .utils import quat_to_euler
from .visualization import draw_scene
from .analysis import Recorder


class Simulator:
    """按配置从注册表创建各组件并驱动仿真主循环。"""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.duration = cfg["simulation"].get("duration")

        robot_cfg = copy.deepcopy(cfg["robot"])
        robot_cfg["model_path"] = cfg["simulation"]["model_path"]
        self.robot = robot_registry.create(robot_cfg.pop("type"), cfg=robot_cfg)

        traj_cfg = copy.deepcopy(cfg["trajectory"])
        self.traj = trajectory_registry.create(traj_cfg.pop("type"), cfg=traj_cfg)

        obs_cfg = copy.deepcopy(cfg["observer"])
        self.observer = observer_registry.create(
            obs_cfg.pop("type"), cfg=obs_cfg, dt=self.robot.dt)

        kin_cfg = copy.deepcopy(cfg["kinematic_controller"])
        self.kin_ctrl = kinematic_registry.create(
            kin_cfg.pop("type"), cfg=kin_cfg, robot=self.robot)

        dyn_cfg = copy.deepcopy(cfg["dynamic_controller"])
        self.dyn_ctrl = dynamic_registry.create(
            dyn_cfg.pop("type"), cfg=dyn_cfg, robot=self.robot)

        log_cfg = cfg.get("logging", {})
        self.recorder = Recorder(log_cfg.get("sample_every", 10))
        self.plot_enabled = log_cfg.get("plot", True)

    def run(self):
        """启动 viewer 并执行 观测-控制-步进 主循环, 结束后绘图。"""
        robot = self.robot
        vis_points = self.traj.sample_path(2 * np.pi, dt=0.1)

        with mujoco.viewer.launch_passive(robot.model, robot.data) as viewer:
            t, step = 0.0, 0
            dt = robot.dt

            while viewer.is_running():
                if self.duration is not None and t >= self.duration:
                    break

                robot.forward()

                traj = self.traj.sample(t)
                twist_true = np.concatenate([traj.vel_true, traj.omega_true])
                twist_ref = self.observer.update(traj.pos, traj.quat, twist_true)

                kin_out = self.kin_ctrl.compute(traj, twist_ref)
                tau = self.dyn_ctrl.compute(kin_out["dq"])

                p_cur, q_cur = robot.ee_pose()
                true_twist = None
                if traj.vel_true is not None:
                    true_twist = np.concatenate(
                        [traj.vel_true, traj.omega_true])
                self.recorder.maybe_record(
                    step, t, traj, p_cur, quat_to_euler(q_cur),
                    kin_out["err_p"], twist_ref, true_twist)

                draw_scene(viewer, vis_points, traj.pos)
                robot.step(tau)
                viewer.sync()
                t += dt
                step += 1

        print("仿真已停止。")
        if self.plot_enabled and not self.recorder.empty:
            print("正在生成分析图表...")
            self.recorder.plot()
