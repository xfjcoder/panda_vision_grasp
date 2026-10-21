# Panda Vision Grasp

A vision-based grasping project for the Franka Emika Panda arm, simulated in MuJoCo: a policy
learns to grasp an object from **camera pixels alone**, with no privileged object pose in its
observation. A separate project (and repo) from `go1_rl_walk`, `slam_sim`, `wam`, and
`barrett_wam` (sibling directories), following the same convention: each new kind of problem
gets its own repo, not a branch of an existing one.

<p align="center">
  <img src="media/stage1_kinematic_sweep.gif" width="480" alt="Franka Panda arm, each joint swept through its own range one at a time under real closed-loop position control, then the gripper closing and opening">
</p>

*Stage 1's verification sweep: each of the 7 arm joints driven to its own range limits and back
under the model's own (already-tuned) position actuators, then the gripper closing and opening.*

## Scoping decisions

- **Platform: the Franka Emika Panda** (`mujoco_menagerie`'s `franka_emika_panda` package),
  not the Barrett WAM again or a different arm -- chosen specifically *because* a ready-made,
  already-tuned MuJoCo model exists for it (vendored wholesale into `assets/`, see
  `assets/NOTICE.md`), unlike `barrett_wam`'s arm which had to be built from a raw URDF. That
  frees this project to spend its effort on the actual new problem (vision) instead of
  re-doing `barrett_wam`'s model-building work on different hardware.
- **Task: vision-based grasping from pixels**, not reach/track/grasp staged from scratch again
  -- `barrett_wam` already covered that staged arc on different hardware; the genuinely new
  capability here is replacing privileged ground-truth object pose with a camera observation,
  which every prior project in this environment has either used ground truth for (`barrett_wam`)
  or deferred entirely (`wam`'s own pixel-based-world-model track got as far as one-step/rollout
  prediction, never a working grasp/control policy from pixels).
- **Approach: end-to-end RL, pixels directly to actions**, not a classical pose-estimation step
  feeding the existing scripted/IK control stack. A bigger, more open-ended learning problem than
  a perception-only swap -- deliberately chosen over the smaller alternative (estimate pose from
  the image, reuse scripted control) to make the vision question itself the hard part being
  tested, not an isolated side-channel.
- **Simulator: MuJoCo**, reusing the same physics engine every sibling project already uses, via
  `go1_rl_walk/mpc`'s venv (same `mujoco>=3.1.0` dependency, no new venv needed).

## Staged plan

1. **Vendor the Panda model, verify it loads and actually moves under its own actuators --
   DONE.** `mujoco_menagerie`'s `franka_emika_panda` package is already a curated, ready-to-use
   MJCF model (position-controlled arm actuators, a tendon-coupled two-finger gripper actuator
   remapped to a 0-255 command range) -- copied wholesale into `assets/` rather than re-derived,
   since (unlike `barrett_wam`'s arm) there was no import problem to fix here.

   **Verified with real closed-loop physics stepping (`mj_step` under gravity), not just
   forward kinematics on a set `qpos`** -- each of the 7 arm joints commanded to its own
   `[low, high]` range and back to 0 one at a time (holding the others at their last commanded
   target), then the gripper closed (`ctrl=0`) and opened (`ctrl=255`, confirmed empirically:
   `ctrl=0` converges `finger_joint1` to 0/closed, `ctrl=255` converges it to 0.04/open).

   **Found a real, repeatable steady-state gravity sag, not hidden**: holding all 7 joints at a
   commanded target of 0 for 3s converges to `[0, 0.001, 0, -0.07, 0, 0.193, 0.001]`, not exactly
   0 -- joint6 in particular sags ~11 degrees under gravity at this pose. Not fixed here (no
   gravity-compensation feedforward added, unlike `barrett_wam`'s Stage 2): the end-to-end RL
   policy planned for later stages observes pixels and outputs actions through the same
   imperfect position-control loop, so it has every opportunity to learn around this bias as
   part of the task itself, the same way a real robot's controller imperfections are just part
   of what a learned policy has to work with -- revisit only if a later stage's results suggest
   this sag is actually a limiting factor, not before.

## Repository layout

```
panda_vision_grasp/
├── assets/
│   ├── panda.xml, hand.xml, scene.xml   # vendored from mujoco_menagerie, see NOTICE.md
│   ├── assets/                          # meshes the above MJCF references
│   ├── LICENSE.franka_emika_panda
│   ├── CHANGELOG.franka_emika_panda.md
│   └── NOTICE.md                        # attribution + what's vendored vs. original here
├── demo_stage1_kinematic_sweep.py   # Stage 1: verification sweep, see media/ GIF above
├── media/
│   └── stage1_kinematic_sweep.gif
├── requirements.txt
└── README.md
```
