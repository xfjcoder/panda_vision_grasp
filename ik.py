"""Damped-least-squares Jacobian IK for the Panda arm's end-effector site ("gripper").

Numerical rather than closed-form, same reasoning as `barrett_wam/ik.py` (which this is
ported from almost unchanged): damped-least-squares handles the redundancy (7 arm joints
driving a 3-DOF position target) without a robot-specific closed-form derivation. Used here
only as a scripted sanity-check baseline (Stage 2) to confirm the grasp scene's geometry is
actually reachable before any pixel-based policy is trained on it -- NOT part of the final
vision policy, which (per this project's own scoping decision) must act from pixels alone,
not from an IK solve over privileged object/joint state.
"""
import mujoco as mj
import numpy as np

N_ARM = 7


def solve_ik(model, target_pos, q_init, site_name="gripper",
             max_iters=200, tol=1e-3, damping=0.05, max_step=0.2):
    """Returns (q_arm, reached, n_iters). q_arm is length-7, clipped to joint ranges."""
    data = mj.MjData(model)
    site_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, site_name)
    q = np.array(q_init[:N_ARM], dtype=float)
    lo = model.jnt_range[:N_ARM, 0]
    hi = model.jnt_range[:N_ARM, 1]
    jacp = np.zeros((3, model.nv))

    for it in range(max_iters):
        data.qpos[:N_ARM] = q
        mj.mj_forward(model, data)
        err = target_pos - data.site_xpos[site_id]
        if np.linalg.norm(err) < tol:
            return q, True, it
        mj.mj_jacSite(model, data, jacp, None, site_id)
        J = jacp[:, :N_ARM]
        dq = J.T @ np.linalg.solve(J @ J.T + damping ** 2 * np.eye(3), err)
        step = np.clip(dq, -max_step, max_step)
        q = np.clip(q + step, lo, hi)

    data.qpos[:N_ARM] = q
    mj.mj_forward(model, data)
    err = target_pos - data.site_xpos[site_id]
    return q, np.linalg.norm(err) < tol, max_iters


def solve_ik_multistart(model, target_pos, q_init, site_name="gripper",
                         n_restarts=20, restart_rng=None, **kwargs):
    """solve_ik, falling back to random-restart if the warm-started solve gets stuck.

    See barrett_wam's README (Stage 2/3 IK-convergence-gap addendum) for why this matters:
    damped-least-squares IK with joint-limit clipping can deadlock (an error-reducing direction
    needs an already-pinned joint to move further out), and a different starting posture almost
    always escapes it.
    """
    q, ok, it = solve_ik(model, target_pos, q_init, site_name=site_name, **kwargs)
    if ok:
        return q, True, it, 0
    if restart_rng is None:
        restart_rng = np.random.default_rng(0)
    lo = model.jnt_range[:N_ARM, 0]
    hi = model.jnt_range[:N_ARM, 1]
    for r in range(n_restarts):
        q_try = restart_rng.uniform(lo, hi)
        q2, ok2, it2 = solve_ik(model, target_pos, q_try, site_name=site_name, **kwargs)
        if ok2:
            return q2, True, it2, r + 1
    return q, False, it, n_restarts
