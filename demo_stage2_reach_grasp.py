"""Stage 2 demo: the scripted (privileged-state) reach + grasp + lift baseline, rendered from
the SAME fixed `workspace_cam` the eventual pixels-only policy will observe from -- this is the
view the vision task actually has to solve from, not a cosmetic orbiting demo camera.

Same phases as eval_stage2_reach_grasp.py (pregrasp / descend / close+hold / lift). seed=0 is
one of that eval sweep's 20/20 successes.

Usage:
    python3 demo_stage2_reach_grasp.py --record media/stage2_reach_grasp.gif
"""
import argparse
from pathlib import Path

import mujoco as mj
import numpy as np
from PIL import Image

from control import pd_gravity_comp_step
from eval_stage2_reach_grasp import (SCENE, SITE, STANDOFF, LIFT_HEIGHT, GRIPPER_OPEN,
                                      GRIPPER_CLOSE, PREGRASP_SECONDS, DESCEND_SECONDS,
                                      CLOSE_SECONDS, HOLD_SECONDS, LIFT_SECONDS,
                                      POST_LIFT_HOLD_SECONDS, random_table_xy, best_oriented_ik,
                                      object_yaw_for_gripper, _yaw_quat)
from ik import solve_ik, N_ARM

SEED = 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", type=str, default=None)
    args = ap.parse_args()

    model = mj.MjModel.from_xml_path(SCENE)
    data = mj.MjData(model)
    site_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, SITE)
    obj_body = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "object")
    obj_joint = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, "object_free")
    obj_qadr = model.jnt_qposadr[obj_joint]
    obj_dof = model.jnt_dofadr[obj_joint]
    obj_geom = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, "object_geom")
    table_geom = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, "table_top")
    obj_half_height = model.geom_size[obj_geom][2]

    rng = np.random.default_rng(SEED)
    obj_xy = random_table_xy(rng)
    data.qpos[:N_ARM] = 0.0
    data.qvel[:N_ARM] = 0.0
    data.qpos[obj_qadr:obj_qadr + 3] = [2.0, 2.0, -1.0]
    data.qpos[obj_qadr + 3:obj_qadr + 7] = [1, 0, 0, 0]
    mj.mj_forward(model, data)
    table_top_z = data.geom_xpos[table_geom][2] + model.geom_size[table_geom][2]
    obj_pos0 = np.array([obj_xy[0], obj_xy[1], table_top_z + obj_half_height])

    dt = model.opt.timestep
    restart_rng = np.random.default_rng(40_000 + SEED)

    renderer = mj.Renderer(model, height=480, width=640)
    frames = []
    step_count = [0]

    def render():
        renderer.update_scene(data, camera="workspace_cam")
        frames.append(renderer.render().copy())

    def run(q_target, gripper_target, n):
        for i in range(n):
            frac = (i + 1) / n
            pd_gravity_comp_step(model, data, q_target[0] + (q_target[1] - q_target[0]) * frac,
                                  gripper_target)
            step_count[0] += 1
            if step_count[0] % 20 == 0:
                render()

    above = obj_pos0 + np.array([0, 0, STANDOFF])
    q_pregrasp, orient_score = best_oriented_ik(model, above, restart_rng,
                                                 descend_check_pos=obj_pos0)
    n = int(PREGRASP_SECONDS / dt)
    for i in range(n):
        pd_gravity_comp_step(model, data, q_pregrasp, GRIPPER_OPEN)
        step_count[0] += 1
        if step_count[0] % 20 == 0:
            render()

    yaw = object_yaw_for_gripper(model, q_pregrasp, site_id)
    data.qpos[obj_qadr:obj_qadr + 3] = obj_pos0
    data.qpos[obj_qadr + 3:obj_qadr + 7] = _yaw_quat(yaw)
    data.qvel[obj_dof:obj_dof + 6] = 0.0
    mj.mj_forward(model, data)

    q_grasp, _, _ = solve_ik(model, obj_pos0, q_pregrasp, max_iters=150, tol=0.01)
    run((q_pregrasp, q_grasp), GRIPPER_OPEN, int(DESCEND_SECONDS / dt))

    run((q_grasp, q_grasp), GRIPPER_CLOSE, int((CLOSE_SECONDS + HOLD_SECONDS) / dt))

    lifted = obj_pos0 + np.array([0, 0, LIFT_HEIGHT])
    q_lifted, _, _ = solve_ik(model, lifted, q_grasp, max_iters=150, tol=0.01)
    run((q_grasp, q_lifted), GRIPPER_CLOSE, int(LIFT_SECONDS / dt))

    run((q_lifted, q_lifted), GRIPPER_CLOSE, int(POST_LIFT_HOLD_SECONDS / dt))

    obj_final = data.xpos[obj_body].copy()
    rise = obj_final[2] - obj_pos0[2]
    drift = float(np.linalg.norm(obj_final[:2] - obj_pos0[:2]))
    print(f"seed={SEED} orient_score={orient_score:.3f} obj_xy={np.round(obj_xy, 3)} "
          f"rise={rise:.4f}m drift={drift:.4f}m")

    if args.record:
        Path(args.record).parent.mkdir(parents=True, exist_ok=True)
        # GitHub's README renderer won't animate (or sometimes won't even display) GIFs above
        # ~10MB -- half-resolution + an optimized palette keeps every rendered GIF well under
        # that (see demo_stage5_rl_failure.py).
        half_res = [f[::2, ::2] for f in frames]
        imgs = [Image.fromarray(f) for f in half_res]
        imgs[0].save(args.record, save_all=True, append_images=imgs[1:], duration=33, loop=0,
                     optimize=True)
        print(f"Saved {len(frames)} frames to {args.record}")

    renderer.close()


if __name__ == "__main__":
    main()
