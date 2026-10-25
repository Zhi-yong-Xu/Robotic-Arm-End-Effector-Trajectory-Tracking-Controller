"""MuJoCo 模型信息查询工具: 自动识别手臂主链, 提取 config.yaml 所需配置。

用法:
    python inspect_model.py assets/scene.xml
    python inspect_model.py assets/scene.xml --all
    python inspect_model.py assets/scene.xml --exclude "finger|gripper"
"""
import argparse
import re

import mujoco


ARM_JOINT_TYPES = {mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE}


def build_children_map(model) -> dict:
    """构建 parent_body_id -> [child_body_id] 的运动树映射。"""
    children = {i: [] for i in range(model.nbody)}
    for i in range(1, model.nbody):
        children[int(model.body_parentid[i])].append(i)
    return children


def count_chain_joints(model, children, root: int) -> int:
    """统计从 root 沿单链下探可收集到的关节数, 用于挑选机器人根 body。"""
    n = 0
    bid = root
    while True:
        n += sum(1 for j in range(model.njnt)
                 if int(model.jnt_bodyid[j]) == bid
                 and model.jnt_type[j] in ARM_JOINT_TYPES)
        if len(children[bid]) != 1:
            break
        bid = children[bid][0]
    return n


def find_arm_chain(model, children):
    """沿运动树单链下探到分支点。

    手臂是单链结构 (link0->...->hand), 分支点即夹爪/末端所在的 body;
    分支点及其后代的关节 (如手指 slide 关节) 不属于手臂主链。
    返回 (链上关节名列表, 末端 body 名)。
    """
    def body_joints(bid):
        return [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
                for j in range(model.njnt)
                if int(model.jnt_bodyid[j]) == bid
                and model.jnt_type[j] in ARM_JOINT_TYPES]

    roots = children[0]
    if not roots:
        return [], None
    root = max(roots, key=lambda b: count_chain_joints(model, children, b))

    chain_joints, bid = [], root
    while True:
        chain_joints += body_joints(bid)
        if len(children[bid]) != 1:
            break
        bid = children[bid][0]

    ee_body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
    return chain_joints, ee_body


def body_sites(model, data, body_id: int):
    """返回指定 body 上的 site 名与位置信息列表。"""
    result = []
    for sid in range(model.nsite):
        if int(model.site_bodyid[sid]) == body_id:
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, sid)
            pos = data.site_xpos[sid]
            result.append((name, pos))
    return result


def inspect(path: str, show_all: bool, exclude: str | None):
    """加载模型, 识别手臂链并打印 config.yaml 填写模板。"""
    model = mujoco.MjModel.from_xml_path(path)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    print("=" * 60)
    print(f"模型: {path}")
    print(f"nbody={model.nbody}  nq={model.nq}  nv={model.nv}  "
          f"nu={model.nu}  timestep={model.opt.timestep}")
    print("=" * 60)

    children = build_children_map(model)
    chain_joints, ee_body = find_arm_chain(model, children)

    excluded = []
    if exclude:
        pattern = re.compile(exclude)
        excluded = [j for j in chain_joints if pattern.search(j)]
        chain_joints = [j for j in chain_joints if not pattern.search(j)]
        if excluded:
            print(f"--exclude 过滤掉的关节: {excluded}")

    ee_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ee_body)
    sites = body_sites(model, data, ee_body_id)

    if show_all:
        print("\n[手臂链上的 body 与关节]")
        bid = children[0][0]
        while True:
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
            print(f"  {name:<20} 关节: {body_joints_of(model, bid) or '无'}")
            if len(children[bid]) != 1:
                print(f"  └─ 分支点: {children[bid]} 个子body -> "
                      f"{[mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, c) for c in children[bid]]}")
                break
            bid = children[bid][0]
        print("\n[全部关节]")
        for jid in range(model.njnt):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
            bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                                      int(model.jnt_bodyid[jid]))
            print(f"  {name:<20} type={mujoco.mjtJoint(model.jnt_type[jid]).name}  "
                  f"body={bname}")
        print("\n[全部 site]")
        for sid in range(model.nsite):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, sid)
            bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                                      int(model.site_bodyid[sid]))
            pos = data.site_xpos[sid]
            print(f"  {name:<20} body={bname:<15} "
                  f"pos=[{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")

    print("=" * 60)
    print("config.yaml 填写模板:")
    print("robot:")
    print('  type: "panda"')
    print(f"  joint_names: {chain_joints}")
    ee_body_str = f'"{ee_body}"' if ee_body else '"???"'
    print(f"  # 末端 body: {ee_body_str}")
    if sites:
        name, pos = sites[0]
        print(f'  ee_site: "{name}"   # pos=[{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]')
    else:
        print('  ee_site: "???"   # 末端 body 上无 site, 可在 XML 中添加:')
        print(f'         <body name="{ee_body}"> 内加入 <site name="tool0" pos="0 0 0.1"/>')
    print("=" * 60)


def body_joints_of(model, bid: int):
    """返回指定 body 上的手臂关节名列表。"""
    return [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
            for j in range(model.njnt)
            if int(model.jnt_bodyid[j]) == bid
            and model.jnt_type[j] in ARM_JOINT_TYPES]


def main():
    parser = argparse.ArgumentParser(description="MuJoCo 模型手臂链/关节/Site 查询")
    parser.add_argument("model_path", help="模型 XML 文件路径")
    parser.add_argument("--all", action="store_true", help="显示运动树与全部关节/site")
    parser.add_argument("--exclude", default=None,
                        help="手动排除关节名的正则, 如 --exclude \"finger|gripper\"")
    args = parser.parse_args()
    inspect(args.model_path, args.all, args.exclude)


if __name__ == "__main__":
    main()
