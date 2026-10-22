"""Gymnasium env for the actual project goal: grasp + lift the table-top sphere using ONLY
camera images + joint proprioception as observation -- no object position, no contact force, no
privileged state of any kind reaches the policy. (Reward computation, during training only,
freely uses ground-truth object state -- that's standard practice and doesn't leak into what
the policy itself ever sees.)

**Two cameras as of this stage**: the fixed external `workspace_cam` (the only one used in every
prior Stage 5 attempt, `bc01`/`bc02`/`grasp01`-`08`) plus the eye-in-hand `wrist_cam` added to
`panda.xml` for visualization in demo GIFs -- that camera is now ALSO part of the policy's actual
observation, not just a visualization add-on. Motivated by a concrete, measured finding: `bc02`'s
alignment offset at the gripper-close phase (0.03-0.12m across several seeds) far exceeds the
4cm-wide object's 2cm half-width, and judging that offset from a single small, distant blob in a
64x64 external frame is a much harder estimation problem than from a close, high-resolution
eye-in-hand view during the precision-critical final approach. This is a clean break from every
earlier Stage 5 checkpoint's observation space -- `bc01`/`bc02`/`grasp01`-`08` are NOT
re-evaluable against this env version (their saved networks have no `wrist_image` branch), but
their results stay documented as-is; this project deliberately doesn't keep a dual code path for
re-running already-concluded experiments.

**Scope decision, made here rather than left implicit**: the episode starts from a
`best_oriented_ik`-computed pregrasp posture above the (randomized) object position, not from
the arm's arbitrary home pose. Same reasoning as barrett_wam's own first RL stage (which only
tackled its close phase, not the whole pick): cold-start RL on a long-horizon, multi-phase
manipulation task is a well-documented hard problem (barrett_wam hit this directly -- repeated
from-scratch failures even WITH privileged state), so the first attempt here narrows the learned
portion to descend + grasp + lift, deferring "vision-guided reach from an arbitrary starting
pose" to a later stage once this narrower problem is shown to work at all.

Action (8-dim, all in [-1, 1]): a per-step DELTA on the 7 arm joint targets (scaled by
MAX_JOINT_RATE) plus a direct (not delta) gripper ctrl command. Chosen over absolute joint
targets for the same reason barrett_wam's residual design did -- bounding how much a single
step can move keeps early random-policy exploration from immediately slamming the arm into the
table or a joint limit.
"""
import gymnasium as gym
import mujoco as mj
import numpy as np
from gymnasium import spaces

from control import pd_gravity_comp_step
from eval_stage2_reach_grasp import (SCENE, SITE, STANDOFF, LIFT_HEIGHT, GRIPPER_OPEN,
                                      GRIPPER_CLOSE, random_table_xy_scaled, best_oriented_ik,
                                      object_yaw_for_gripper, _yaw_quat)
from ik import N_ARM

IMG_SIZE = 64
CONTROL_DT = 0.05
EPISODE_SECONDS = 6.0
EPISODE_STEPS = int(EPISODE_SECONDS / CONTROL_DT)
MAX_JOINT_RATE = 0.08  # rad per control step
SUCCESS_RISE_FRAC = 0.6  # matches eval_stage2_reach_grasp's own success criterion
SUCCESS_DRIFT_MAX = 0.04

# Reward weights -- dense terms kept deliberately small relative to the terminal bonus, after
# barrett_wam's own "reward hacking" finding (a dense term that can sum to more than the
# terminal bonus over an episode gets exploited instead of the real objective).
W_DIST = 0.5
W_HEIGHT = 0.5
W_ACTION = 0.001
TERMINAL_BONUS = 20.0

# grasp05 (PPO fine-tuning on a behavior-cloned checkpoint) measurably pushed the policy's
# per-step action magnitude 13x higher than the demonstrator's own pace (0.0088 -> 0.116 rad),
# reproducing the same "rush in and knock the object sideways" failure grasp04's GIF shows --
# because nothing in the reward above penalizes HOW the arm approaches, only distance and
# height. W_ACTION alone can't fix this: it's uniform over the whole episode, so raising it
# enough to matter near the object would also penalize the (necessary, and fine as-is) fast
# reach phase. This term only activates within PROXIMITY_RADIUS of the object, scaling up as
# the gripper gets closer, so the reach phase is untouched but a careless final approach isn't.
PROXIMITY_RADIUS = STANDOFF  # 0.15m -- same scale Stage 2's own pregrasp standoff uses
# grasp06's single-episode breakdown (dist -20.0 vs approach -0.33) suggested a ~60x weight
# bump, but that ratio isn't stable across episodes -- it depends heavily on how long the
# rollout actually spends inside PROXIMITY_RADIUS, which differs a lot between a successful
# (long dwell) and a failed (barely gets close) trajectory. A weight sweep against BOTH the
# scripted trajectory (10 seeds, must stay successful with reward not far off its ~20.8
# baseline) and grasp06's own real rollout (10 seeds, want its approach-speed total to become
# comparable to its distance total, not swamp it or stay negligible) found W=0.5: scripted
# stays 10/10 at reward~19.1 (barely down from ~20.8), while grasp06's approach-speed total
# (-14.6) is now the same order of magnitude as its distance total (-12.4), a real competing
# incentive -- unlike the naive 60x scale-up (W=6.0), which also blows up the scripted
# trajectory's reward to -11.8 despite it still succeeding, because the 60x ratio measured on
# one failed episode doesn't generalize to a successful one's much longer proximity dwell time.
W_APPROACH_SPEED = 0.5

# Every RL-trained checkpoint so far (grasp01/04/06), not just the fast ones, visibly "shakes"
# in its rendered rollout -- measured directly: consecutive-step action components flip sign
# 20-68% of the time (vs. bc01/the scripted controller's ~2%), and the raw (unweighted) sum of
# ||a_t - a_t-1||^2 over an arm-joint action sequence is 2-3 orders of magnitude higher for
# these policies (112-1675) than for the scripted trajectory (~0.05) -- a separate failure mode
# from approach SPEED (W_APPROACH_SPEED above): nothing in the reward penalizes reversing
# direction step to step, only the raw magnitude and (now) proximity-gated speed, so rapid
# oscillation is free as long as the average magnitude/proximity cost stays low. Calibrated the
# same way as W_APPROACH_SPEED -- checked the raw total against both the scripted trajectory
# (must stay negligible) and grasp01's real rollout (the worst oscillator measured, want it
# penalized on the same order as TERMINAL_BONUS/the distance term) -- W_SMOOTH=0.01 gives
# grasp01-level oscillation a ~9-17 penalty depending on severity while costing the scripted
# trajectory <0.001.
W_SMOOTH = 0.01


class GraspPixelEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, render_size=IMG_SIZE, curriculum_scale=1.0):
        super().__init__()
        # Scales how far from TABLE_XY_RANGE's center the object spawns, in [0, 1]. 1.0 (default)
        # is the full, final task difficulty -- used whenever this env isn't being driven by a
        # curriculum callback (eval, smoke tests, non-curriculum training runs).
        self.curriculum_scale = curriculum_scale
        self.model = mj.MjModel.from_xml_path(SCENE)
        # Shadow mapping turned out to be the dominant cost of rendering on this machine's
        # integrated GPU -- confirmed directly: ~47 renders/sec with it on vs. ~630/sec with it
        # off, at any resolution tried (32-64px), so it wasn't a fill-rate cost. A ~13x speedup
        # for a policy that never needs to interpret shadows anyway.
        self.model.light_castshadow[:] = 0
        self.data = mj.MjData(self.model)
        self.render_size = render_size
        self.renderer = mj.Renderer(self.model, height=render_size, width=render_size)

        self.site_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_SITE, SITE)
        self.obj_body = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, "object")
        obj_joint = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, "object_free")
        self.obj_qadr = self.model.jnt_qposadr[obj_joint]
        self.obj_dof = self.model.jnt_dofadr[obj_joint]
        obj_geom = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, "object_geom")
        self.table_geom = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, "table_top")
        self.obj_half_height = self.model.geom_size[obj_geom][2]
        self.jnt_lo = self.model.jnt_range[:N_ARM, 0]
        self.jnt_hi = self.model.jnt_range[:N_ARM, 1]

        dt = self.model.opt.timestep
        self.frame_skip = round(CONTROL_DT / dt)

        self.observation_space = spaces.Dict({
            "image": spaces.Box(0, 255, (render_size, render_size, 3), dtype=np.uint8),
            "wrist_image": spaces.Box(0, 255, (render_size, render_size, 3), dtype=np.uint8),
            "proprio": spaces.Box(-np.inf, np.inf, (2 * N_ARM + 1,), dtype=np.float32),
        })
        self.action_space = spaces.Box(-1.0, 1.0, (N_ARM + 1,), dtype=np.float32)

        self._np_rng = np.random.default_rng()
        self.q_target = np.zeros(N_ARM)
        self.gripper_target = GRIPPER_OPEN
        self.obj_pos0 = np.zeros(3)
        self.prev_arm_action = np.zeros(N_ARM)
        self.t = 0

    def set_curriculum_scale(self, scale):
        self.curriculum_scale = scale

    def _table_top_z(self):
        mj.mj_forward(self.model, self.data)
        return self.data.geom_xpos[self.table_geom][2] + self.model.geom_size[self.table_geom][2]

    def _get_obs(self):
        # Renderer.render() always allocates a fresh array when called with out=None (confirmed
        # in the mujoco source), so reusing one Renderer for both cameras back to back is safe --
        # no aliasing between the two returned images.
        self.renderer.update_scene(self.data, camera="workspace_cam")
        image = self.renderer.render()
        self.renderer.update_scene(self.data, camera="wrist_cam")
        wrist_image = self.renderer.render()
        gripper_qpos = self.data.qpos[N_ARM] / 0.04  # normalized finger_joint1 position
        proprio = np.concatenate([
            self.data.qpos[:N_ARM], self.data.qvel[:N_ARM], [gripper_qpos],
        ]).astype(np.float32)
        return {"image": image, "wrist_image": wrist_image, "proprio": proprio}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        rng = self._np_rng if seed is None else np.random.default_rng(seed)

        self.data.qpos[:N_ARM] = 0.0
        self.data.qvel[:N_ARM] = 0.0
        self.data.qpos[self.obj_qadr:self.obj_qadr + 3] = [2.0, 2.0, -1.0]
        self.data.qpos[self.obj_qadr + 3:self.obj_qadr + 7] = [1, 0, 0, 0]
        mj.mj_forward(self.model, self.data)

        table_top_z = self._table_top_z()
        # best_oriented_ik occasionally finds no candidate at all (none of n_candidates=40
        # random restarts both reach the target AND clear the table-contact check) -- rare
        # enough that eval_stage2_reach_grasp.py's own fixed 20-seed sweep never hit it, but
        # real during training's much wider resampling across random seeds (confirmed: a
        # SubprocVecEnv auto-reset crashed on exactly this, `best_q=None`). Resample a new
        # object position and retry rather than let a single unlucky table position crash an
        # entire training run.
        for _ in range(10):
            obj_xy = random_table_xy_scaled(rng, self.curriculum_scale)
            obj_pos0 = np.array([obj_xy[0], obj_xy[1], table_top_z + self.obj_half_height])
            above = obj_pos0 + np.array([0, 0, STANDOFF])
            q_pregrasp, _ = best_oriented_ik(self.model, above, rng, descend_check_pos=obj_pos0)
            if q_pregrasp is not None:
                break
        else:
            raise RuntimeError("best_oriented_ik found no valid candidate in 10 resampled "
                                "object positions -- investigate, this should be rare")
        self.obj_pos0 = obj_pos0
        yaw = object_yaw_for_gripper(self.model, q_pregrasp, self.site_id)

        self.data.qpos[:N_ARM] = q_pregrasp
        self.data.qvel[:N_ARM] = 0.0
        self.data.qpos[N_ARM] = 0.0
        self.data.qpos[N_ARM + 1] = 0.0
        self.data.qpos[self.obj_qadr:self.obj_qadr + 3] = self.obj_pos0
        self.data.qpos[self.obj_qadr + 3:self.obj_qadr + 7] = _yaw_quat(yaw)
        self.data.qvel[self.obj_dof:self.obj_dof + 6] = 0.0
        mj.mj_forward(self.model, self.data)

        self.q_target = q_pregrasp.copy()
        self.gripper_target = GRIPPER_OPEN
        self.prev_arm_action = np.zeros(N_ARM)
        self.t = 0
        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        self.q_target = np.clip(self.q_target + action[:N_ARM] * MAX_JOINT_RATE,
                                 self.jnt_lo, self.jnt_hi)
        self.gripper_target = (action[N_ARM] + 1.0) / 2.0 * 255.0

        for _ in range(self.frame_skip):
            pd_gravity_comp_step(self.model, self.data, self.q_target, self.gripper_target)
        self.t += 1

        obj_pos = self.data.xpos[self.obj_body]
        dist = np.linalg.norm(self.data.site_xpos[self.site_id] - obj_pos)
        rise = obj_pos[2] - self.obj_pos0[2]
        drift = np.linalg.norm(obj_pos[:2] - self.obj_pos0[:2])

        proximity = max(0.0, 1.0 - dist / PROXIMITY_RADIUS)
        arm_action_mag = float(np.linalg.norm(action[:N_ARM]))
        smoothness_cost = float(np.sum((action[:N_ARM] - self.prev_arm_action) ** 2))
        self.prev_arm_action = action[:N_ARM].copy()
        reward = (-W_DIST * dist + W_HEIGHT * max(0.0, rise)
                  - W_ACTION * float(np.sum(action ** 2))
                  - W_APPROACH_SPEED * proximity * arm_action_mag
                  - W_SMOOTH * smoothness_cost)
        terminated = self.t >= EPISODE_STEPS
        success = bool(rise > SUCCESS_RISE_FRAC * LIFT_HEIGHT and drift < SUCCESS_DRIFT_MAX)
        if terminated and success:
            reward += TERMINAL_BONUS

        info = {"success": success, "rise": float(rise), "drift": float(drift)}
        return self._get_obs(), reward, terminated, False, info

    def close(self):
        self.renderer.close()
