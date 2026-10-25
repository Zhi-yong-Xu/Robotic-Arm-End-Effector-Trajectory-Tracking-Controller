"""MuJoCo viewer 自定义绘制辅助。"""
import numpy as np
import mujoco


def add_line(scn, p1, p2, radius, rgba):
    """用圆柱体在 user_scn 中连接两点。"""
    if scn.ngeom >= scn.maxgeom:
        return
    mid = (p1 + p2) / 2
    diff = p2 - p1
    length = np.linalg.norm(diff)
    if length < 1e-10:
        return
    direction = diff / length
    z_axis = np.array([0, 0, 1])

    if np.abs(np.dot(direction, z_axis)) > 0.999:
        mat = (np.eye(3) if np.dot(direction, z_axis) > 0
               else np.diag([1, -1, -1]).astype(float))
    else:
        v = np.cross(z_axis, direction)
        v = v / np.linalg.norm(v)
        c = np.dot(z_axis, direction)
        s = np.sqrt(1 - c * c)
        mat = np.array([
            [c + v[0]*v[0]*(1-c), v[0]*v[1]*(1-c) - v[2]*s, v[0]*v[2]*(1-c) + v[1]*s],
            [v[1]*v[0]*(1-c) + v[2]*s, c + v[1]*v[1]*(1-c), v[1]*v[2]*(1-c) - v[0]*s],
            [v[2]*v[0]*(1-c) - v[1]*s, v[2]*v[1]*(1-c) + v[0]*s, c + v[2]*v[2]*(1-c)],
        ])

    mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_CYLINDER,
                        np.array([radius, length / 2, 0]),
                        mid, mat.flatten(), rgba)
    scn.ngeom += 1


def draw_scene(viewer, vis_points, target_pos):
    """绘制参考路径圆环与当前目标点。"""
    viewer.user_scn.ngeom = 0
    rgba = np.array([1, 0, 0, 0.5])
    for i in range(len(vis_points) - 1):
        add_line(viewer.user_scn, vis_points[i], vis_points[i + 1], 0.005, rgba)
    if viewer.user_scn.ngeom < viewer.user_scn.maxgeom:
        mujoco.mjv_initGeom(viewer.user_scn.geoms[viewer.user_scn.ngeom],
                            mujoco.mjtGeom.mjGEOM_SPHERE,
                            np.array([0.02, 0, 0]),
                            target_pos, np.eye(3).flatten(),
                            np.array([1, 1, 0, 1]))
        viewer.user_scn.ngeom += 1
