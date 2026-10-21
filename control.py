"""Gravity-compensation feedforward on top of the Panda's own position actuators.

Same technique as `barrett_wam/control.py`: a plain position-servo loop left a real,
confirmed steady-state sag under gravity (Stage 1's finding, ~11 degrees on one joint at the
straight-down home pose). Rather than add a second feedforward actuator channel to the vendored
MJCF (barrett_wam's approach, since that URDF import had none at all), this applies the
compensating torque directly as an external generalized force (`qfrc_applied`), which the
existing actuators don't need any XML change to support.
"""
import mujoco as mj
import numpy as np

N_ARM = 7


def pd_gravity_comp_step(model, data, q_arm_target, gripper_target=None):
    """One control step: command q_arm_target (+ optional gripper ctrl), cancel gravity/
    Coriolis bias on the 7 arm joints via qfrc_applied, then mj_step."""
    mj.mj_forward(model, data)
    data.ctrl[:N_ARM] = q_arm_target
    if gripper_target is not None:
        data.ctrl[N_ARM] = gripper_target
    data.qfrc_applied[:N_ARM] = data.qfrc_bias[:N_ARM]
    mj.mj_step(model, data)
