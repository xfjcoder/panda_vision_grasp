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
hand's +z toward the fingers/grasp point rather than back up the arm. Originally added purely for
visualization (an eye-in-hand "robot's own view" alongside `workspace_cam` in demo GIFs); as of
the post-`bc02` Stage 5 pivot it's ALSO part of the policy's actual observation in
`grasp_env.py` (`wrist_image`, alongside `image` from `workspace_cam`) -- motivated by `bc02`'s
measured 0.03-0.12m gripper-close alignment offset being hard to judge from a single distant
external view. This breaks compatibility with every checkpoint trained before this change
(`bc01`/`bc02`/`grasp01`-`08`), which only ever saw `workspace_cam`.

Everything else in this repo (scenes/tasks built on top of this model, training code, docs) is
original to this project.
