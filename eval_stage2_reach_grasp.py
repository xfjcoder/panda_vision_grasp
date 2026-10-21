"""Stage 2: scripted reach + grasp + lift, using PRIVILEGED ground-truth object position.

This is NOT the project's actual deliverable (per this repo's own scoping decision, the final
policy must act from pixels alone) -- it's a sanity-check baseline, same spirit as
barrett_wam's own staged approach: confirm the grasp *scene* (table height, object size,
gripper geometry, joint reach) is physically solvable at all with privileged state, before
spending any effort on the much harder pixels-only learning problem. If this baseline can't
reliably grasp, that's a scene-design bug to fix first, not a vision problem to debug around.

Five phases, same structure as barrett_wam's eval_stage4_grasp.py:
  1. pregrasp -- move to a standoff point above the object (best_oriented_ik below), gripper open
  2. descend  -- lower the grasp site onto the object's center, gripper still open
  3. close    -- hold position, command the gripper fully closed (ctrl=0); the object itself
                 mechanically stops full closure, same as a real parallel gripper squeezing
                 something -- no special "don't over-close" logic needed here, unlike
                 barrett_wam's coupled-linkage BHand fingers, which could swing back out past
                 full closure.
  4. lift     -- raise the grasp site, gripper held closed
  5. (verify) -- did the object rise with the hand, or get left behind / knocked away?

Usage:
    python3 eval_stage2_reach_grasp.py --trials 20
"""
import argparse

import mujoco as mj
import numpy as np

from control import pd_gravity_comp_step
from ik import solve_ik, N_ARM

SCENE = "assets/grasp_scene.xml"
SITE = "gripper"

TABLE_XY_RANGE = ((0.35, 0.65), (-0.2, 0.2))  # (x_lo, x_hi), (y_lo, y_hi), margined inside the
                                               # table's actual (0.2-0.8, -0.4-0.4) footprint

STANDOFF = 0.15
LIFT_HEIGHT = 0.15
GRIPPER_OPEN = 255.0
GRIPPER_CLOSE = 0.0

PREGRASP_SECONDS = 1.5
DESCEND_SECONDS = 1.5
CLOSE_SECONDS = 1.0
HOLD_SECONDS = 0.5
LIFT_SECONDS = 1.5
POST_LIFT_HOLD_SECONDS = 2.0  # see run_trial: a static hold immediately after lift is what
                              # actually exposed the sphere-slipping bug -- measuring rise right
                              # after the lift motion completed missed it entirely.


def random_table_xy(rng):
    (xlo, xhi), (ylo, yhi) = TABLE_XY_RANGE
    return np.array([rng.uniform(xlo, xhi), rng.uniform(ylo, yhi)])


def object_yaw_for_gripper(model, q_arm, site_id):
    """Returns the world-frame yaw (rotation about Z) that aligns the box object's flat faces
    with whichever orientation best_oriented_ik's IK solve actually landed on for this trial.

    The gripper's closing direction is the "hand" body's local Y axis (both finger joints slide
    along local Y, see assets/panda.xml); a flat-faced box needs that axis to line up with one
    of its own flat-face normals, or the jaws contact an edge/corner instead of a face. Since
    best_oriented_ik only scores how downward the approach is (never the roll about that
    approach axis), the gripper's actual closing direction varies table-position to
    table-position -- so the box's spawn yaw is computed per trial from the solved q_arm,
    not fixed to a single global convention.
    """
    data = mj.MjData(model)
    data.qpos[:N_ARM] = q_arm
    mj.mj_forward(model, data)
    local_y_world = data.site_xmat[site_id].reshape(3, 3)[:, 1]
    return np.arctan2(local_y_world[1], local_y_world[0])


def _yaw_quat(yaw):
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


def _has_arm_table_contact(model, data):
    for i in range(data.ncon):
        c = data.contact[i]
        b1 = model.body(model.geom_bodyid[c.geom1]).name
        b2 = model.body(model.geom_bodyid[c.geom2]).name
        if "table" in (b1, b2) and "object" not in (b1, b2):
            return True
    return False


def best_oriented_ik(model, target_pos, rng, n_candidates=40, descend_check_pos=None):
    """Position-only IK has 4 redundant DOF for a 7-joint arm driving a 3-DOF position target,
    and which one it lands on is path-dependent, not just a function of the target -- confirmed
    directly: a naive single IK solve from home for a table-height target landed on an
    elbow-high posture that drove the forearm clean through the table (link3/4/7/hand all in
    contact with table_top). Orientation-only scoring (pick whichever solution faces most
    nearly straight down) isn't sufficient by itself here, unlike barrett_wam: for some table
    positions the single best-scoring candidate still pins a joint at its hard limit and, when
    warm-started on down to the actual grasp height, plows the forearm into the table anyway
    (confirmed directly: link3/link4/hand all in contact at the final descend pose, even though
    the pregrasp pose alone looked fine). Fixed by making that check part of selection, not an
    afterthought: when `descend_check_pos` is given, a candidate is rejected outright if
    warm-starting it down to that position produces any arm/table contact, and orientation
    score is only used to rank the survivors.
    """
    site_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, SITE)
    data = mj.MjData(model)
    lo = model.jnt_range[:N_ARM, 0]
    hi = model.jnt_range[:N_ARM, 1]
    best_q, best_score = None, -2.0
    for i in range(n_candidates):
        q_init = np.zeros(N_ARM) if i == 0 else rng.uniform(lo, hi)
        q, ok, _ = solve_ik(model, target_pos, q_init, max_iters=150, tol=0.01)
        if not ok:
            continue
        if descend_check_pos is not None:
            q_descend, ok2, _ = solve_ik(model, descend_check_pos, q, max_iters=150, tol=0.01)
            if not ok2:
                continue
            data.qpos[:N_ARM] = q_descend
            mj.mj_forward(model, data)
            if _has_arm_table_contact(model, data):
                continue
        data.qpos[:N_ARM] = q
        mj.mj_forward(model, data)
        local_z_world = data.site_xmat[site_id].reshape(3, 3)[:, 2]
        score = -local_z_world[2]  # +1 if local z points straight down (0,0,-1), -1 if up
        if score > best_score:
            best_score, best_q = score, q
    return best_q, best_score


def run_trial(model, obj_xy, restart_rng):
    data = mj.MjData(model)
    site_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, SITE)
    obj_body = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "object")
    obj_joint = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, "object_free")
    obj_qadr = model.jnt_qposadr[obj_joint]
    obj_dof = model.jnt_dofadr[obj_joint]

    obj_geom = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, "object_geom")
    table_geom = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, "table_top")
    obj_half_height = model.geom_size[obj_geom][2]
    # table_top is a geom nested inside the "table" body (not directly in worldbody, unlike
    # barrett_wam's layout), so model.geom_pos is LOCAL to that body (0,0,0 here) -- the world
    # z needs data.geom_xpos (post mj_forward), not model.geom_pos, or this silently computes
    # the body's own origin height instead of its actual top surface height.
    mj.mj_forward(model, data)
    table_top_z = data.geom_xpos[table_geom][2] + model.geom_size[table_geom][2]

    obj_pos0 = np.array([obj_xy[0], obj_xy[1], table_top_z + obj_half_height])
    data.qpos[:N_ARM] = 0.0
    data.qvel[:N_ARM] = 0.0
    # Parked out of the way during pregrasp, placed once the arm has safely parked above the
    # table -- same ordering fix barrett_wam's Stage 4 needed (a collision-unaware pregrasp
    # jump can otherwise sweep straight through the object before descend even starts).
    data.qpos[obj_qadr:obj_qadr + 3] = [2.0, 2.0, -1.0]
    data.qpos[obj_qadr + 3:obj_qadr + 7] = [1, 0, 0, 0]
    mj.mj_forward(model, data)

    dt = model.opt.timestep

    above = obj_pos0 + np.array([0, 0, STANDOFF])
    # descend_check_pos rejects any pregrasp candidate that, warm-started down to the actual
    # grasp height, would plow the forearm into the table -- see best_oriented_ik's docstring:
    # orientation score alone wasn't enough, one specific table position's best-scoring
    # candidate still did this.
    q_pregrasp, orient_score = best_oriented_ik(model, above, restart_rng,
                                                 descend_check_pos=obj_pos0)
    n = int(PREGRASP_SECONDS / dt)
    for _ in range(n):
        pd_gravity_comp_step(model, data, q_pregrasp, GRIPPER_OPEN)
    q_arm = q_pregrasp.copy()

    yaw = object_yaw_for_gripper(model, q_pregrasp, site_id)
    data.qpos[obj_qadr:obj_qadr + 3] = obj_pos0
    data.qpos[obj_qadr + 3:obj_qadr + 7] = _yaw_quat(yaw)
    data.qvel[obj_dof:obj_dof + 6] = 0.0
    mj.mj_forward(model, data)

    # Solve the final descend target ONCE, warm-started from q_pregrasp (same validated,
    # table-clear branch -- already checked above), then interpolate in JOINT space rather
    # than re-solving IK on a slowly-moving Cartesian target every step. Per-step re-solving
    # was tried first and found to drift through the redundant null space over hundreds of
    # tiny re-solves with nothing pulling it back, occasionally landing the forearm in the
    # table well before the Cartesian target itself got anywhere near it.
    q_grasp, ok, _ = solve_ik(model, obj_pos0, q_pregrasp, max_iters=150, tol=0.01)
    n = int(DESCEND_SECONDS / dt)
    for i in range(n):
        frac = (i + 1) / n
        q_target = q_pregrasp + (q_grasp - q_pregrasp) * frac
        pd_gravity_comp_step(model, data, q_target, GRIPPER_OPEN)
    q_arm = q_grasp.copy()

    n = int((CLOSE_SECONDS + HOLD_SECONDS) / dt)
    for _ in range(n):
        pd_gravity_comp_step(model, data, q_arm, GRIPPER_CLOSE)

    obj_before_lift = data.xpos[obj_body].copy()

    lifted = obj_pos0 + np.array([0, 0, LIFT_HEIGHT])
    q_lifted, ok, _ = solve_ik(model, lifted, q_arm, max_iters=150, tol=0.01)
    n = int(LIFT_SECONDS / dt)
    for i in range(n):
        frac = (i + 1) / n
        q_target = q_arm + (q_lifted - q_arm) * frac
        pd_gravity_comp_step(model, data, q_target, GRIPPER_CLOSE)
    q_arm = q_lifted.copy()

    # A static hold immediately after lift, gripper target unchanged -- this is what actually
    # caught the sphere-slipping bug (see the object's own docstring note in grasp_scene.xml):
    # measuring success right after the lift motion completed, with no further hold, missed a
    # grip that looked fine for about a second and then silently let go.
    n = int(POST_LIFT_HOLD_SECONDS / dt)
    for _ in range(n):
        pd_gravity_comp_step(model, data, q_arm, GRIPPER_CLOSE)

    obj_after_lift = data.xpos[obj_body].copy()
    rise = obj_after_lift[2] - obj_before_lift[2]
    drift = float(np.linalg.norm(obj_after_lift[:2] - obj_pos0[:2]))
    grasped = (rise > 0.6 * LIFT_HEIGHT) and (drift < 0.04)
    return dict(obj_xy=obj_xy, rise=rise, drift=drift, grasped=grasped, orient_score=orient_score)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--base-seed", type=int, default=0)
    args = ap.parse_args()

    model = mj.MjModel.from_xml_path(SCENE)
    results = []
    for i in range(args.trials):
        seed = args.base_seed + i
        rng = np.random.default_rng(seed)
        obj_xy = random_table_xy(rng)
        restart_rng = np.random.default_rng(40_000 + seed)
        r = run_trial(model, obj_xy, restart_rng)
        results.append(r)
        print(f"trial {i:2d} seed={seed:4d} obj_xy={np.round(obj_xy, 3)} "
              f"rise={r['rise']:.4f}m drift={r['drift']:.4f}m grasped={r['grasped']}")

    n_grasped = sum(r["grasped"] for r in results)
    print(f"\nSummary over {args.trials} trials: {n_grasped}/{args.trials} grasped")


if __name__ == "__main__":
    main()
