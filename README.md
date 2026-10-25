# Robotic-Arm-End-Effector-Trajectory-Tracking-Controller
A MuJoCo-based end-effector trajectory tracking control framework for robotic arms. Features a plug-and-play registration architecture for swapping robot models (Panda, iiwa14), differential IK controllers, and dynamic controllers via a single YAML config file.

# 项目架构
```
                config/config.yaml
                       │
                       ▼
              ┌──────────────────┐
              │   Simulator      │   src/simulation.py
              │  (组装 + 主循环) │
              └────┬──┬──┬──┬──┬─┘
                   │  │  │  │  │
        ┌──────────┘  │  │  │  └──────────┐
        ▼             ▼  ▼  ▼             ▼
  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
  │  Robot   │ │Trajectory│ │ Observer │ │ Kinematic│ │ Dynamic  │
  │ registry │ │ registry │ │ registry │ │ registry │ │ registry │
  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘
       │            │            │            │            │
       ▼            ▼            ▼            ▼            ▼
   panda/       circle/      ftdso/      velocity_pd/     pd/
   iiwa14/      line/        se3/                        dsc/
                lissajous/   none                        gravity_comp
                helix/
                waypoints/

每步流程：
  1. 轨迹采样          → pos_des, quat_des
  2. 观测器            → twist_ref = [v_ref; w_ref]
  3. 运动学控制器      → dq_des（含奇异规避、限幅）
  4. 动力学控制器      → tau（关节力矩）
  5. mj_step + viewer  → 可视化 & 记录
```

# 快速开始
```
 1. 克隆仓库
git clone https://github.com/Zhi-yong-Xu/Robotic-Arm-End-Effector-Trajectory-Tracking-Controller.git
cd Robotic-Arm-End-Effector-Trajectory-Tracking-Controller

# 2. (推荐) 创建虚拟环境
python -m venv venv
# Linux / macOS
source venv/bin/activate
# Windows
venv\Scripts\activate

# 3. 安装依赖
pip install mujoco mujoco-viewer numpy scipy matplotlib pyyaml

python main.py

```

## 展示
### panda机械臂
<img width="522" height="404" alt="image" src="https://github.com/user-attachments/assets/c49f032a-6a83-4e26-b534-5e36f6c00ddf" />
<img width="1046" height="719" alt="image" src="https://github.com/user-attachments/assets/a59bdacd-1985-49ee-ae16-78b89fef3a42" />


### iiwa14机械臂
<img width="707" height="541" alt="image" src="https://github.com/user-attachments/assets/6f70afce-fba1-480b-a3f8-52042be04ac4" />
<img width="1034" height="665" alt="image" src="https://github.com/user-attachments/assets/a2e30e7b-1556-4f06-a5d3-f57a98a17551" />

