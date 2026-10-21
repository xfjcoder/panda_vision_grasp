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

2. **Grasp scene + a scripted (privileged-state) reach/grasp/lift sanity check -- DONE.**
   Before spending any effort on the actual pixels-only learning problem, confirmed the scene
   itself (table height, object size, gripper geometry, joint reach) is physically solvable at
   all -- same discipline as `barrett_wam`'s own staged approach. This scripted baseline is
   **not** the project's deliverable (it uses privileged ground-truth object position), only a
   feasibility check and a reference number to compare the eventual vision policy against.

   `assets/grasp_scene.xml` adds a static table, a free-floating graspable sphere (radius
   2.5cm, same choice as `barrett_wam`'s own grasp object and for the same reason --
   orientation-invariant, so this first vision task doesn't also have to solve orientation
   estimation from pixels), and a fixed external `workspace_cam` (computed via a look-at from
   `(1.3, 0, 1.05)` toward the table center) -- the same view the eventual pixels-only policy
   will actually observe from. One small, documented modification to the vendored `panda.xml`
   (see `assets/NOTICE.md`): added a `gripper` site at the same local offset the upstream
   package's own MJX variant already uses, needed as an IK/grasp-point reference.

   **Two real bugs found and fixed, not glossed over**:
   - **Object spawned inside the table.** `table_top_z` was computed from `model.geom_pos`
     (the table-top geom's *local* offset inside its parent body, which is `(0,0,0)` here,
     since this scene nests the geom inside a positioned body rather than placing it directly
     in `worldbody` the way `barrett_wam`'s layout did) instead of `data.geom_xpos` (the
     resolved *world*-frame position) -- silently computing the body's own origin height
     instead of the table's actual top surface. The sphere spawned embedded 21cm deep inside
     the table block, and the contact solver's violent resolution of that overlap sent it
     sliding clear across the table at explosive speed (drift up to 20m, confirmed). Fixed by
     reading `data.geom_xpos` (post `mj_forward`) instead.
   - **Position-only IK's redundant-DOF drift, a new failure mode this project hadn't hit
     before.** Even after fixing the spawn bug, re-solving IK on a slowly-moving Cartesian
     target every single control step (the same pattern `barrett_wam`'s `run_phase` used
     successfully) let the arm's 4 redundant DOF drift through null space over hundreds of
     tiny re-solves, with nothing pulling them back -- confirmed directly: the forearm
     (`link3`) ended up pinned against the table, `qfrc_actuator` saturated at its torque limit
     fighting a real contact, even though every single per-step IK solve reported success
     (its convergence check is purely kinematic and has no notion of what the real,
     physically-driven arm can or can't do). A pure null-space bias term (pulling the
     redundant DOF toward the original posture) was tried first and made things worse, not
     better -- discrete joint-limit clipping breaks the clean orthogonality the math assumes,
     so a non-trivial bias weight just destabilized the primary-task convergence instead.
     **Fixed with a structurally different approach instead of a tuning fix**: solve IK
     *once* for the final descend target (warm-started from the already-validated pregrasp
     posture, not from scratch), then interpolate the commanded pose in *joint* space between
     the two solved endpoints -- no per-step re-solving, so there's nothing left to drift.
     Also found that orientation-score selection alone (picking whichever `best_oriented_ik`
     candidate faces most nearly downward, `barrett_wam`'s own criterion) wasn't sufficient by
     itself: the single best-scoring candidate for one specific table position still pinned a
     joint at its hard limit and, when warm-started down to the grasp height, drove the
     forearm into the table anyway. Fixed by folding that check directly into candidate
     selection: a candidate is rejected outright if warm-starting it down to the actual grasp
     height produces any arm/table contact, before orientation is used to rank the survivors.

   **Evaluated over 20 random table positions** (`eval_stage2_reach_grasp.py`): **20/20
   grasped** (object consistently rises ~0.133-0.137m of a commanded 0.15m lift, settles
   within 7mm of its starting position) -- a clean, reliable baseline, unlike `barrett_wam`'s
   own first-grasp-attempt numbers (11/20), mostly because a rigid 2-finger parallel gripper
   squeezing a sphere has no analog of the BHand's coupled-linkage "full closure retracts past
   the object" failure mode, and this sanity check doesn't yet have to contend with a
   learned-from-pixels perception error on top of the control problem.

   <p align="center">
     <img src="media/stage2_reach_grasp.gif" width="480" alt="Franka Panda arm reaching down, closing its gripper around a red sphere, and lifting it off the table, viewed from the fixed workspace camera">
   </p>

   *Stage 2's demo (`demo_stage2_reach_grasp.py`, seed 0): rendered from the exact same fixed
   `workspace_cam` the eventual pixels-only policy will observe from, not a cosmetic orbiting
   camera -- this is the view the vision task actually has to solve from.*

## Repository layout

```
panda_vision_grasp/
├── assets/
│   ├── panda.xml, hand.xml, scene.xml   # vendored from mujoco_menagerie, see NOTICE.md
│   ├── assets/                          # meshes the above MJCF references
│   ├── LICENSE.franka_emika_panda
│   ├── CHANGELOG.franka_emika_panda.md
│   └── NOTICE.md                        # attribution + what's vendored vs. original here
├── ik.py                             # Stage 2: damped-least-squares Jacobian IK
├── control.py                        # Stage 2: gravity-compensation feedforward
├── eval_stage2_reach_grasp.py        # Stage 2: 20-trial scripted reach/grasp/lift sanity check
├── demo_stage1_kinematic_sweep.py    # Stage 1: verification sweep, see media/ GIF above
├── demo_stage2_reach_grasp.py        # Stage 2: scripted grasp demo, see media/ GIF above
├── media/
│   ├── stage1_kinematic_sweep.gif
│   └── stage2_reach_grasp.gif
├── requirements.txt
└── README.md
```
