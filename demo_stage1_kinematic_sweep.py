"""Stage 1 verification: does the vendored Franka Emika Panda MJCF (assets/scene.xml, from
mujoco_menagerie) move the way a real Panda should under its own (already-tuned) position
actuators?

Unlike barrett_wam's Stage 1 (pure kinematics -- that URDF import had no actuators yet), this
model ships with position-controlled actuators already tuned by mujoco_menagerie, so this
sweep drives REAL closed-loop physics (mj_step under gravity, not just mj_forward on a set
qpos) -- confirming the ready-made model actually holds/tracks a commanded pose, not just that
it loads.

Usage:
    python3 demo_stage1_kinematic_sweep.py --record media/stage1_kinematic_sweep.gif
"""
import argparse
from pathlib import Path

import mujoco as mj
import numpy as np
from PIL import Image

ARM_JOINTS = [f"joint{i}" for i in range(1, 8)]
N_ARM = 7
GRIPPER_ACTUATOR = "actuator8"  # ctrl=0 -> closed (finger joints at 0), ctrl=255 -> open (0.04)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=str, default="assets/scene.xml")
    ap.add_argument("--record", type=str, default=None)
    ap.add_argument("--seconds-per-target", type=float, default=0.6)
    args = ap.parse_args()

    model = mj.MjModel.from_xml_path(args.scene)
    data = mj.MjData(model)
    dt = model.opt.timestep
    jid = {name: mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, name) for name in ARM_JOINTS}
    qadr = {name: model.jnt_qposadr[jid[name]] for name in ARM_JOINTS}
    gripper_act = mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, GRIPPER_ACTUATOR)
    n = int(args.seconds_per_target / dt)

    renderer = mj.Renderer(model, height=480, width=640)
    cam = mj.MjvCamera()
    cam.type = mj.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0.3, 0, 0.4]
    cam.distance = 1.6
    cam.elevation = -20

    frames = []
    step_count = [0]
    azimuth = [50.0]

    def render():
        azimuth[0] += 0.3
        cam.azimuth = azimuth[0]
        renderer.update_scene(data, camera=cam)
        frames.append(renderer.render().copy())

    data.ctrl[:N_ARM] = 0.0
    data.ctrl[gripper_act] = 0.0

    def hold_target(arm_target, gripper_target):
        data.ctrl[:N_ARM] = arm_target
        data.ctrl[gripper_act] = gripper_target
        for _ in range(n):
            mj.mj_step(model, data)
            step_count[0] += 1
            if step_count[0] % 20 == 0:
                render()

    arm_target = np.zeros(N_ARM)
    # Sweep each arm joint, one at a time, to its own [low, high] range and back to 0, holding
    # every other joint at its last commanded (not necessarily yet-achieved) target meanwhile.
    for i, name in enumerate(ARM_JOINTS):
        lo, hi = model.jnt_range[jid[name]]
        for val in (hi, lo, 0.0):
            arm_target[i] = val
            hold_target(arm_target, 0.0)
        arm_target[i] = 0.0

    # Close then open the gripper.
    for gripper_target in (0.0, 255.0):
        hold_target(arm_target, gripper_target)

    achieved = np.array([data.qpos[qadr[name]] for name in ARM_JOINTS])
    print(f"Rendered {len(frames)} frames. Commanded arm target: {np.round(arm_target, 3)}, "
          f"achieved qpos: {np.round(achieved, 3)} (should match closely -- confirms real "
          f"closed-loop tracking, not just forward kinematics).")

    if args.record:
        Path(args.record).parent.mkdir(parents=True, exist_ok=True)
        imgs = [Image.fromarray(f) for f in frames]
        imgs[0].save(args.record, save_all=True, append_images=imgs[1:], duration=33, loop=0)
        print(f"Saved {len(frames)} frames to {args.record}")

    renderer.close()


if __name__ == "__main__":
    main()
