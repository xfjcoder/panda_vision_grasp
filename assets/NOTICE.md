`panda.xml`, `hand.xml`, `scene.xml`, and everything under `assets/` are vendored unmodified
from [`google-deepmind/mujoco_menagerie`](https://github.com/google-deepmind/mujoco_menagerie)'s
`franka_emika_panda` package (local checkout: `~/robo/mujoco_menagerie`), licensed Apache 2.0
(`LICENSE.franka_emika_panda`, see also `CHANGELOG.franka_emika_panda.md` for that package's own
derivation history from the upstream Franka URDF). Copied in wholesale (33MB) rather than
re-derived or pruned, since this package is already a curated, ready-to-use MJCF model -- unlike
`barrett_wam`'s Barrett WAM, no MuJoCo-ready model existed for that robot anywhere in this
environment, which is why that project's vendoring was a real build step instead of a copy.

**One small modification to `panda.xml`**: added a `<site name="gripper" pos="0 0 0.1"/>` inside
the `hand` body, at the exact same local offset the same upstream package's own `mjx_panda.xml`
variant already uses for its own gripper site -- this file (the non-MJX variant) just didn't
have one. Used as the grasp-point/end-effector site for IK and camera-relative reasoning in this
project; no other change to the file.

**A second small modification to `panda.xml`**: added a `<camera name="wrist_cam" pos="0 0 0.02"
quat="0 1 0 0" fovy="70"/>` inside the same `hand` body, just behind the `gripper` site, oriented
180 degrees about the local x-axis so its view direction (the camera's own -z) points along the
hand's +z toward the fingers/grasp point rather than back up the arm. Purely for visualization
(an eye-in-hand "robot's own view" alongside the external `workspace_cam` in demo GIFs) -- the
policy's actual observation in `grasp_env.py` still renders from `workspace_cam` only, this
camera is never read by training/eval code.

Everything else in this repo (scenes/tasks built on top of this model, training code, docs) is
original to this project.
