"""Gymnasium env for the actual project goal: grasp + lift the table-top sphere using ONLY a
camera image + joint proprioception as observation -- no object position, no contact force, no
privileged state of any kind reaches the policy. (Reward computation, during training only,
freely uses ground-truth object state -- that's standard practice and doesn't leak into what
the policy itself ever sees.)

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
            "proprio": spaces.Box(-np.inf, np.inf, (2 * N_ARM + 1,), dtype=np.float32),
        })
        self.action_space = spaces.Box(-1.0, 1.0, (N_ARM + 1,), dtype=np.float32)

        self._np_rng = np.random.default_rng()
        self.q_target = np.zeros(N_ARM)
        self.gripper_target = GRIPPER_OPEN
        self.obj_pos0 = np.zeros(3)
        self.t = 0

    def set_curriculum_scale(self, scale):
        self.curriculum_scale = scale

    def _table_top_z(self):
        mj.mj_forward(self.model, self.data)
        return self.data.geom_xpos[self.table_geom][2] + self.model.geom_size[self.table_geom][2]

    def _get_obs(self):
        self.renderer.update_scene(self.data, camera="workspace_cam")
        image = self.renderer.render()
        gripper_qpos = self.data.qpos[N_ARM] / 0.04  # normalized finger_joint1 position
        proprio = np.concatenate([
            self.data.qpos[:N_ARM], self.data.qvel[:N_ARM], [gripper_qpos],
        ]).astype(np.float32)
        return {"image": image, "proprio": proprio}

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

        reward = -W_DIST * dist + W_HEIGHT * max(0.0, rise) - W_ACTION * float(np.sum(action ** 2))
        terminated = self.t >= EPISODE_STEPS
        success = bool(rise > SUCCESS_RISE_FRAC * LIFT_HEIGHT and drift < SUCCESS_DRIFT_MAX)
        if terminated and success:
            reward += TERMINAL_BONUS

        info = {"success": success, "rise": float(rise), "drift": float(drift)}
        return self._get_obs(), reward, terminated, False, info

    def close(self):
        self.renderer.close()
