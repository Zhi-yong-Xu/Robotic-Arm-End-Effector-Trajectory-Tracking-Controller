import mujoco
import mujoco.viewer
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation


def quat2euler(target_quat):
    # scipy输入四元数顺序 [x,y,z,w]，需要调换
    w, x, y, z = target_quat
    r = Rotation.from_quat([x, y, z, w])
    # 输出xyz欧拉角，弧度
    euler = r.as_euler("xyz", degrees=False)
    return euler


# ============================================
# 修复 Matplotlib 中文显示问题
# ============================================
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

class PandaPDController:
    def __init__(self, model, joint_names, kp=200, kd=0.1):
        self.model = model
        self.joint_names = joint_names
        self.num_joints = len(joint_names)
        self.qpos_adr = []
        self.dof_adr = []
        for name in joint_names:
            jid = model.joint(name).id
            self.qpos_adr.append(model.jnt_qposadr[jid])
            self.dof_adr.append(model.jnt_dofadr[jid])
        self.Kp = np.eye(self.num_joints) * kp
        self.Kd = np.eye(self.num_joints) * kd
        self.q_des = np.zeros(self.num_joints)
        self.dq_des = np.zeros(self.num_joints)
        self.ddq_des = np.zeros(self.num_joints)

    def set_target(self, dq_des, ddq_des):
        if len(dq_des) == self.num_joints:
            self.dq_des = np.array(dq_des)
        if len(ddq_des) == self.num_joints:
            self.ddq_des = np.array(ddq_des)

    def compute_torque(self, data):
        tau_bias = np.array([data.qfrc_bias[dof] for dof in self.dof_adr])
        tau_passive = np.array([data.qfrc_passive[dof] for dof in self.dof_adr])
        tau = tau_bias - tau_passive
        return tau

# ============================================
# 轨迹生成器参数
# ============================================
CENTER = np.array([0.0, 0.0, 0.5])
RADIUS = 0.5
OMEGA = 1.0
TARGET_QUAT = np.array([1, 0, 0, 0])
N_ARM = 7

def generate_trajectory(t):
    """生成 t 时刻的参考轨迹"""
    theta = OMEGA * t
    
    # --- 位置 ---
    target_pos = np.array([
        CENTER[0] + RADIUS * np.cos(theta),
        CENTER[1] + RADIUS * np.sin(theta),
        CENTER[2] + np.sin(5*theta) * 0.1
    ])
    
    # --- 姿态 (恒定) ---
    euler_1d = np.array([0.6, 0.5, 0.3])
    euler_angles = euler_1d.reshape(3, 1)
    quat = np.zeros((4, 1))
    mujoco.mju_euler2Quat(quat, euler_angles, "xyz")
    target_quat = quat.flatten() # 确保 shape 为 (4,)
    
    # --- 速度 ---
    target_vel = np.array([
        -RADIUS * OMEGA * np.sin(theta),
        RADIUS * OMEGA * np.cos(theta),
        np.cos(5*theta) * 0.5
    ])
    target_omega = np.zeros(3)
    
    return target_pos, target_quat, target_vel, target_omega

def add_line(scn, p1, p2, radius, rgba):
    if scn.ngeom >= scn.maxgeom: return
    mid = (p1 + p2) / 2
    diff = p2 - p1
    length = np.linalg.norm(diff)
    if length < 1e-10: return
    direction = diff / length
    z_axis = np.array([0, 0, 1])
    if np.abs(np.dot(direction, z_axis)) > 0.999:
        mat = np.eye(3) if np.dot(direction, z_axis) > 0 else np.diag([1, -1, -1]).astype(float)
    else:
        v = np.cross(z_axis, direction)
        v = v / np.linalg.norm(v)
        c = np.dot(z_axis, direction)
        s = np.sqrt(1 - c*c)
        mat = np.array([
            [c + v[0]*v[0]*(1-c), v[0]*v[1]*(1-c) - v[2]*s, v[0]*v[2]*(1-c) + v[1]*s],
            [v[1]*v[0]*(1-c) + v[2]*s, c + v[1]*v[1]*(1-c), v[1]*v[2]*(1-c) - v[0]*s],
            [v[2]*v[0]*(1-c) - v[1]*s, v[2]*v[1]*(1-c) + v[0]*s, c + v[2]*v[2]*(1-c)]
        ])
    mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_CYLINDER,
                        np.array([radius, length/2, 0]), mid, mat.flatten(), rgba)
    scn.ngeom += 1

def velocity_pd_controller(model, data, target_pos, target_quat, target_vel, target_omega, kp=5.0):
    """带速度输入的PD控制器，并返回当前实际位姿"""
    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    
    # 获取当前末端状态
    p_cur = data.body(ee_id).xpos.copy()
    q_cur = data.body(ee_id).xquat.copy().flatten() # 确保 shape 为 (4,)
    
    # 误差计算 (控制律中仍使用轴角误差，避免欧拉角奇异点问题)
    err_p = target_pos - p_cur
    err_r_body = np.zeros(3)
    mujoco.mju_subQuat(err_r_body, target_quat, q_cur)
    R_cur = np.zeros(9)
    mujoco.mju_quat2Mat(R_cur, q_cur)
    err_r = R_cur.reshape(3, 3) @ err_r_body
    
    pose_error = np.concatenate([err_r, err_p])
    
    # 雅可比
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jac(model, data, jacp, jacr, data.body(ee_id).xpos, ee_id)
    jac = np.vstack([jacr[:, :7], jacp[:, :7]])
    
    desired_motion = np.concatenate([target_omega, target_vel]) + kp * pose_error
    damp = 1e-4
    jjt = jac @ jac.T + damp * np.eye(6)
    dq = jac.T @ np.linalg.solve(jjt, desired_motion)
    
    return dq, err_p, err_r, p_cur, q_cur

# ============================================
# 修改后的绘图函数：使用欧拉角
# ============================================
def plot_results(time_list, 
                 target_pos_list, actual_pos_list,
                 target_euler_list, actual_euler_list,
                 pos_err_list, euler_err_list):
    
    # 转换为 numpy 数组
    t = np.array(time_list)
    target_pos = np.array(target_pos_list)
    actual_pos = np.array(actual_pos_list)
    target_euler = np.array(target_euler_list)
    actual_euler = np.array(actual_euler_list)
    pos_err = np.array(pos_err_list)
    euler_err = np.array(euler_err_list)
    
    # 计算位置误差范数
    pos_err_norm = np.linalg.norm(pos_err, axis=1)

    # 创建画布：3行2列
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle('机械臂末端执行器位姿跟踪分析 (欧拉角版)', fontsize=18, fontweight='bold')

    # ============================
    # 第一行：位置跟踪
    # ============================
    axes[0, 0].plot(t, target_pos[:, 0], 'r--', label='期望 X', alpha=0.6)
    axes[0, 0].plot(t, actual_pos[:, 0], 'r-', label='实际 X', linewidth=1.5)
    axes[0, 0].set_ylabel('位置 X (m)')
    axes[0, 0].legend(loc='upper right')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_title('位置 X 轴跟踪', fontsize=12)

    axes[0, 1].plot(t, target_pos[:, 1], 'g--', label='期望 Y', alpha=0.6)
    axes[0, 1].plot(t, actual_pos[:, 1], 'g-', label='实际 Y', linewidth=1.5)
    axes[0, 1].set_ylabel('位置 Y (m)')
    axes[0, 1].legend(loc='upper right')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_title('位置 Y 轴跟踪', fontsize=12)

    # ============================
    # 第二行：位置Z 和 欧拉角跟踪
    # ============================
    axes[1, 0].plot(t, target_pos[:, 2], 'b--', label='期望 Z', alpha=0.6)
    axes[1, 0].plot(t, actual_pos[:, 2], 'b-', label='实际 Z', linewidth=1.5)
    axes[1, 0].set_ylabel('位置 Z (m)')
    axes[1, 0].set_xlabel('时间')
    axes[1, 0].legend(loc='upper right')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_title('位置 Z 轴跟踪', fontsize=12)

    # 欧拉角对比 (Roll, Pitch, Yaw)
    axes[1, 1].plot(t, target_euler[:, 0], 'm--', label='期望 Roll', alpha=0.6)
    axes[1, 1].plot(t, actual_euler[:, 0], 'm-', label='实际 Roll', linewidth=1.5)
    axes[1, 1].plot(t, target_euler[:, 1], 'c--', label='期望 Pitch', alpha=0.6)
    axes[1, 1].plot(t, actual_euler[:, 1], 'c-', label='实际 Pitch', linewidth=1.5)
    axes[1, 1].plot(t, target_euler[:, 2], 'y--', label='期望 Yaw', alpha=0.6)
    axes[1, 1].plot(t, actual_euler[:, 2], 'y-', label='实际 Yaw', linewidth=1.5)
    axes[1, 1].set_ylabel('欧拉角')
    axes[1, 1].set_xlabel('时间')
    axes[1, 1].legend(loc='upper right', fontsize='small')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_title('姿态跟踪 (欧拉角 R/P/Y)', fontsize=12)

    # ============================
    # 第三行：误差分析
    # ============================
    # 位置误差范数
    axes[2, 0].plot(t, pos_err_norm, 'k-', linewidth=2)
    axes[2, 0].set_ylabel('误差 (m)')
    axes[2, 0].set_xlabel('时间')
    axes[2, 0].set_title('位置误差范数', fontsize=12)
    axes[2, 0].grid(True, alpha=0.3)
    axes[2, 0].axhline(y=0.01, color='r', linestyle='--', alpha=0.5, label='1cm 阈值')
    axes[2, 0].legend()

    # 欧拉角误差 (各分量)
    axes[2, 1].plot(t, euler_err[:, 0], 'm-', label='Roll 误差', linewidth=1.5)
    axes[2, 1].plot(t, euler_err[:, 1], 'c-', label='Pitch 误差', linewidth=1.5)
    axes[2, 1].plot(t, euler_err[:, 2], 'y-', label='Yaw 误差', linewidth=1.5)
    axes[2, 1].set_ylabel('误差')
    axes[2, 1].set_xlabel('时间')
    axes[2, 1].set_title('姿态误差 (欧拉角分量)', fontsize=12)
    axes[2, 1].legend(loc='upper right', fontsize='small')
    axes[2, 1].grid(True, alpha=0.3)
    axes[2, 1].axhline(y=0.017, color='r', linestyle='--', alpha=0.5, label='1度阈值')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
    
    # 打印统计信息
    print("\n" + "="*60)
    print("误差统计信息")
    print("="*60)
    print(f"位置误差 - 均值: {np.mean(pos_err_norm)*1000:.3f} mm, 最大值: {np.max(pos_err_norm)*1000:.3f} mm")
    
    # 计算欧拉角误差的范数
    euler_err_norm = np.linalg.norm(euler_err, axis=1)
    print(f"姿态误差(Euler) - 均值: {np.mean(euler_err_norm):.4f} rad ({np.degrees(np.mean(euler_err_norm)):.2f}°)")
    print(f"姿态误差(Euler) - 最大值: {np.max(euler_err_norm):.4f} rad ({np.degrees(np.max(euler_err_norm)):.2f}°)")
    print("="*60)
    
    plt.show()

def main():
    model = mujoco.MjModel.from_xml_path("model/panda/panda.xml")
    data = mujoco.MjData(model)
    
    joint_names = [f"joint{i}" for i in range(1, 8)]
    controller = PandaPDController(model, joint_names)
    
    traj_vis_points = [generate_trajectory(i*0.1)[0] for i in range(int(2*np.pi/0.1) + 1)]
    
    step_count = 0
    
    # ============================================
    # 数据存储列表
    # ============================================
    time_list = []
    target_pos_list, actual_pos_list = [], []
    target_euler_list, actual_euler_list = [], [] # 新增：存储欧拉角
    pos_err_list, euler_err_list = [], []         # 修改：存储欧拉角误差
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        t = 0.0
        dt = model.opt.timestep
        alpha = np.zeros(7)
        dq_prev = data.qvel[:7].copy()
        
        nv_total = model.nv
        n_arm = 7
        M_full = np.zeros((nv_total, nv_total))
        
        while viewer.is_running():
            mujoco.mj_forward(model, data)
            mujoco.mj_fullM(model, data, M_full)
            M_arm = M_full[:n_arm, :n_arm]
            
            target_pos, target_quat, target_vel, target_omega = generate_trajectory(t)
            
            # 获取控制量、误差和实际位姿
            dq, err_p, err_r, p_cur, q_cur = velocity_pd_controller(
                model, data, target_pos, target_quat, target_vel, target_omega
            )
            
            # ============================================
            # 转换并记录欧拉角数据
            # ============================================
            # 将目标四元数转为欧拉角
            tar_euler = np.zeros(3)
            tar_euler = quat2euler(target_quat)
            
            # 将实际四元数转为欧拉角
            act_euler = np.zeros(3)
            act_euler = quat2euler(q_cur)
            
            # 计算欧拉角误差 (直接相减，假设误差较小无跳变)
            # 注意：如果存在奇异性或角度跳变，这里可能需要解包处理，
            # 但对于此圆周跟踪任务通常没问题。
            err_euler = tar_euler - act_euler
            
            if step_count % 10 == 0:
                time_list.append(t)
                target_pos_list.append(target_pos)
                actual_pos_list.append(p_cur)
                
                target_euler_list.append(tar_euler)
                actual_euler_list.append(act_euler)
                
                pos_err_list.append(err_p)
                euler_err_list.append(err_euler)
            
            d_alpha = - 20 * (alpha - dq)
            ddq = (dq - dq_prev) / dt
            controller.set_target(dq, ddq)
            tau = controller.compute_torque(data)
            dq_prev = dq
            alpha += d_alpha * dt
            tau = tau + M_arm @ (d_alpha + 5 * (dq - data.qvel[:7].copy()))
            
            # 可视化部分
            viewer.user_scn.ngeom = 0
            for i in range(len(traj_vis_points) - 1):
                add_line(viewer.user_scn, traj_vis_points[i], traj_vis_points[i+1], 0.005, np.array([1, 0, 0, 0.5]))
            if viewer.user_scn.ngeom < viewer.user_scn.maxgeom:
                mujoco.mjv_initGeom(viewer.user_scn.geoms[viewer.user_scn.ngeom],
                                    mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.02, 0, 0]),
                                    target_pos, np.eye(3).flatten(), np.array([1, 1, 0, 1]))
                viewer.user_scn.ngeom += 1
            
            data.ctrl[:7] = tau
            mujoco.mj_step(model, data)
            step_count += 1
            viewer.sync()
            t += dt
    
    print("仿真已停止。")
    
    if len(time_list) > 0:
        print("\n正在生成分析图表...")
        plot_results(time_list, 
                     target_pos_list, actual_pos_list,
                     target_euler_list, actual_euler_list,
                     pos_err_list, euler_err_list)

if __name__ == "__main__":
    main()
