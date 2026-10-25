"""机器人模型: 封装 MuJoCo 模型与运动学/动力学查询, 约定 twist = [v(3); w(3)]。
末端以 site 点为跟踪目标, 位姿/雅可比均基于 site 计算。"""
import mujoco
import numpy as np

from .registry import robot_registry


class BaseRobot:
    """机械臂基类, 提供关节寻址、末端 site 运动学、关节空间动力学接口。"""

    def __init__(self, model_path: str, joint_names: list, ee_site: str):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.joint_names = list(joint_names)
        self.n_arm = len(self.joint_names)

        self.qpos_adr, self.dof_adr = [], []
        for name in self.joint_names:
            jid = self.model.joint(name).id
            self.qpos_adr.append(int(self.model.jnt_qposadr[jid]))
            self.dof_adr.append(int(self.model.jnt_dofadr[jid]))

        self.ee_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, ee_site)
        if self.ee_site_id < 0:
            raise ValueError(f"找不到末端 site: '{ee_site}'")
        # site 姿态变化随其所属 body 运动, 角速度雅可比按该 body 计算
        self.ee_body_id = int(self.model.site_bodyid[self.ee_site_id])

        self._M_full = np.zeros((self.model.nv, self.model.nv))

    @property
    def dt(self):
        """仿真步长。"""
        return self.model.opt.timestep

    @property
    def qvel_arm(self) -> np.ndarray:
        """当前机械臂关节速度。"""
        return self.data.qvel[self.dof_adr].copy()

    @property
    def bias_torque(self) -> np.ndarray:
        """重力 + 科氏力 + 离心力。"""
        return self.data.qfrc_bias[self.dof_adr].copy()

    @property
    def passive_torque(self) -> np.ndarray:
        """关节阻尼等被动力。"""
        return self.data.qfrc_passive[self.dof_adr].copy()

    def forward(self):
        """执行前向计算, 更新 bias/passive/雅可比等缓存量。"""
        mujoco.mj_forward(self.model, self.data)

    def step(self, tau: np.ndarray):
        """写入控制力矩并推进仿真一步。"""
        self.data.ctrl[: self.n_arm] = tau
        mujoco.mj_step(self.model, self.data)

    def ee_pose(self):
        """返回末端 site 位置 (3,) 与四元数 (4,) wxyz。"""
        pos = self.data.site_xpos[self.ee_site_id].copy()
        xmat = self.data.site_xmat[self.ee_site_id]  # 3x3 行主序旋转矩阵
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, np.asarray(xmat).flatten())
        return pos, quat

    def ee_jacobian(self) -> np.ndarray:
        """末端 site 雅可比 (6 x n_arm), 行序 [v; w]。"""
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        point = self.data.site_xpos[self.ee_site_id]
        mujoco.mj_jac(self.model, self.data, jacp, jacr,
                      point, self.ee_body_id)
        return np.vstack([jacp[:, self.dof_adr], jacr[:, self.dof_adr]])

    def ee_twist(self) -> np.ndarray:
        """真实末端 site 速度 [v; w] = J·dq, 仅用于绘图验证。"""
        return self.ee_jacobian() @ self.qvel_arm

    def arm_mass_matrix(self) -> np.ndarray:
        """机械臂 n_arm x n_arm 关节空间惯性矩阵。"""
        mujoco.mj_fullM(self.model, self.data, self._M_full)
        return self._M_full[: self.n_arm, : self.n_arm].copy()


@robot_registry.register("panda")
@robot_registry.register("ur3")
@robot_registry.register("iiwa14")
class PandaRobot(BaseRobot):
    """joint_names 与 ee_site 为必填配置。"""

    def __init__(self, cfg: dict):
        for key in ("joint_names", "ee_site"):
            if key not in cfg:
                raise KeyError(f"robot 配置缺少必填项: '{key}'")
        super().__init__(cfg["model_path"], cfg["joint_names"], cfg["ee_site"])
