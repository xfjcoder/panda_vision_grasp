`panda.xml`, `hand.xml`, `scene.xml`, and everything under `assets/` are vendored unmodified
from [`google-deepmind/mujoco_menagerie`](https://github.com/google-deepmind/mujoco_menagerie)'s
`franka_emika_panda` package (local checkout: `~/robo/mujoco_menagerie`), licensed Apache 2.0
(`LICENSE.franka_emika_panda`, see also `CHANGELOG.franka_emika_panda.md` for that package's own
derivation history from the upstream Franka URDF). Copied in wholesale (33MB) rather than
re-derived or pruned, since this package is already a curated, ready-to-use MJCF model -- unlike
`barrett_wam`'s Barrett WAM, no MuJoCo-ready model existed for that robot anywhere in this
environment, which is why that project's vendoring was a real build step instead of a copy.

Everything else in this repo (scenes/tasks built on top of this model, training code, docs) is
original to this project.
