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

   `assets/grasp_scene.xml` adds a static table, a free-floating graspable object, and a fixed
   external `workspace_cam` (computed via a look-at from `(1.3, 0, 1.05)` toward the table
   center) -- the same view the eventual pixels-only policy will actually observe from. One
   small, documented modification to the vendored `panda.xml` (see `assets/NOTICE.md`): added a
   `gripper` site at the same local offset the upstream package's own MJX variant already uses,
   needed as an IK/grasp-point reference.

   **Three real bugs found and fixed, not glossed over**:
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
   - **A sphere (tried first, same object choice as `barrett_wam` and for the same
     orientation-invariance reason) turned out not to work with this gripper.** It looked fine
     right after lifting but was found to slip free after roughly a second of an otherwise-
     completely-static hold -- gripper qpos collapsing from ~0.025 to ~0 and the object
     dropping back to the table -- a real geometric mismatch (pinching a smooth round object
     between two flat parallel pads is a classic non-form-closure grasp), not a tuning bug:
     higher friction wasn't the fix. The eval itself was missing this entirely at first too,
     since it only measured rise right after the lift motion completed with no further hold --
     fixed by adding a 2-second static `POST_LIFT_HOLD_SECONDS` before measuring success, which
     is what actually caught the slip. Switched the object to a small box instead (flat faces
     mate properly with the flat gripper pads), which brings an orientation requirement back
     in -- handled by computing the box's spawn yaw per trial (`object_yaw_for_gripper`) from
     the gripper's actual closing-axis direction at whichever posture `best_oriented_ik`
     happened to land on, rather than fixing one global orientation that would only sometimes
     line up.

   **Evaluated over 20 random table positions** (`eval_stage2_reach_grasp.py`), now including
   the 2-second post-lift static hold: **20/20 grasped** (object rises ~0.142-0.147m of a
   commanded 0.15m lift, settles within 6mm of its starting position) -- a clean, reliable
   baseline, genuinely stable under sustained holding, not just at the instant the lift motion
   finishes.

   <p align="center">
     <img src="media/stage2_reach_grasp.gif" width="480" alt="Franka Panda arm reaching down, closing its gripper around a small red box, and lifting it off the table, viewed from the fixed workspace camera">
   </p>

   *Stage 2's demo (`demo_stage2_reach_grasp.py`, seed 0): rendered from the exact same fixed
   `workspace_cam` the eventual pixels-only policy will observe from, not a cosmetic orbiting
   camera -- this is the view the vision task actually has to solve from.*

3. **The actual vision Gym environment -- DONE (env verified, policy training not yet
   launched).** `grasp_env.py`'s `GraspPixelEnv`: a Gymnasium env whose observation is
   `{"image": 64x64 RGB from workspace_cam, "proprio": 15-dim arm joint pos/vel + gripper
   position}` -- no object position, no contact force, no privileged state of any kind reaches
   the policy. Action is 8-dim: a per-step delta on the 7 arm joint targets (bounded by
   `MAX_JOINT_RATE`) plus a direct gripper ctrl command, same bounded-step reasoning as
   `barrett_wam`'s residual design (keeps early random-policy exploration from immediately
   slamming the arm into the table or a joint limit). Reward is dense shaping (gripper-to-object
   distance, object height while lifted) plus a terminal success bonus, computed from
   ground-truth object state during training -- that's standard practice and doesn't leak into
   what the policy itself observes.

   **Scope decision, made explicitly rather than left implicit**: episodes start from a
   `best_oriented_ik`-computed pregrasp posture above the (randomized) object, not from the
   arm's arbitrary home pose -- same reasoning as `barrett_wam`'s own first RL stage, which only
   tackled its close phase rather than the whole pick. Cold-start RL on a long, multi-phase
   manipulation task is a well-documented hard problem, and `barrett_wam` hit it directly
   (repeated from-scratch failures even WITH privileged state) -- so the first attempt here
   narrows the learned portion to descend + grasp + lift, deferring "vision-guided reach from an
   arbitrary starting pose" to a later stage once this narrower problem is shown to work at all.

   **Verified the env itself before spending any training compute**, same discipline as
   `barrett_wam`'s own residual-RL premise check: replayed Stage 2's scripted
   descend/close/lift trajectory (converted into this env's own action space) through
   `GraspPixelEnv.step()` end-to-end. Result: `success=True`, matching Stage 2's own grasp
   criterion exactly, with a sane positive total episode reward (+20.8, dominated by the
   terminal bonus) -- confirming the env's physics, reward wiring, and action/observation
   plumbing are all correct before any policy has been trained on it.

4. **First PPO training run (`grasp01`) -- DONE, honest partial result: 0/20, real but
   insufficient learning.** `train.py` mirrors `barrett_wam`/`go1_rl_walk`'s own conventions
   (PPO + `SubprocVecEnv` + `VecMonitor` + `VecNormalize`, per-run `args.json`), with two
   changes for a `Dict{image, proprio}` observation: `"MultiInputPolicy"` instead of
   `"MlpPolicy"`, and `norm_obs_keys=["proprio"]` so `VecNormalize`'s running-mean/std
   normalization is applied only to proprioception -- normalizing raw image pixels too would
   feed the CNN extractor's built-in `/255` assumption the wrong distribution entirely.

   **Found and fixed a rendering-speed bug before launching any real training budget.**
   Benchmarked raw env throughput first (same "verify premise" discipline as
   `barrett_wam`'s residual-RL check): only ~46 renders/sec no matter the resolution tried
   (32-64px) or how many parallel envs were used (8 or 14 gave the *same* total throughput,
   not more -- a shared-GPU-context bottleneck, not a CPU one, confirmed via a single-process
   render-loop benchmark landing at the identical number). Traced to shadow-map rendering
   specifically (an extra render pass MuJoCo does for cast shadows): disabling it
   (`model.light_castshadow[:] = 0` in `GraspPixelEnv.__init__`) gave a confirmed **~13x
   speedup** (46 -> 630 renders/sec) -- a policy that only ever sees pixels has no need to
   interpret shadows anyway. Also fixed a crash found via a short smoke-training run (the
   actual "verify before scaling" check): `best_oriented_ik` occasionally finds no valid
   candidate for an unlucky random table position -- never hit in Stage 2's fixed 20-seed
   sweep, but real across training's much wider resampling (confirmed: it crashed a
   `SubprocVecEnv` worker on exactly this). Fixed by resampling the object position and
   retrying instead of letting one unlucky position crash an entire run.

   **Result** (1M timesteps, 8 parallel envs, ~430->270 fps as other load on the machine
   varied): episode reward climbed from the random-policy baseline (~-20 to -27) to **-12.7**
   over training -- a genuine, large improvement, not noise. `eval_policy.py` (same
   `random_table_xy` seed convention as `eval_stage2_reach_grasp.py`, so the two numbers are
   directly comparable): **0/20** succeeded. Rendering a held-out rollout showed why: the
   policy reliably learned to descend the arm down to table height near the object (not
   flailing randomly), but doesn't reliably center the gripper on it or time the close/lift
   -- it drifts past the object rather than grasping it. The same shape of result
   `barrett_wam` hit on its own first RL attempts (real, partial learning, not yet enough to
   succeed), on a problem that's strictly harder here: full reach+grasp+lift from raw pixels,
   cold-start, no privileged-state residual baseline to correct like `barrett_wam` eventually
   used. `barrett_wam`'s own directly analogous finding ("more data and time: 2/20 -> 4/20")
   was resolved by scaling up budget before any bigger redesign was needed -- tried next here
   too, see below.

5. **Scaled up 4x (`grasp02`, resumed from `grasp01`'s checkpoint, +3M timesteps = 4M total)
   -- negative result: more budget alone did NOT help here, unlike `barrett_wam`'s analogous
   case.** Same training recipe, no reward/hyperparameter changes. Result: **still 0/20**,
   training reward actually *regressed* partway through (climbed to -9.4 around the 1.5M-step
   mark, then drifted down to -12 by the end) rather than continuing to improve or even
   plateauing cleanly.

   **Diagnosed, not just reported as a flat "didn't work"**: the policy's action-distribution
   `std` collapsed hard and fast throughout this run (0.43 -> 0.06), while `approx_kl` stayed
   persistently far above PPO's normal healthy range (0.8-1.6, vs. a typical 0.01-0.05) for
   the entire 3M timesteps -- a small std mechanically inflates measured KL for the same
   underlying policy shift, which is consistent with premature exploration collapse: the
   policy became highly confident in a specific behavior (descend-but-don't-center) very
   early and stopped exploring away from it, rather than genuinely converging on a good one.
   `ent_coef=0.0` (this project's and `barrett_wam`'s shared convention, chosen there
   specifically because nonzero `ent_coef` risked a policy-std *blowup* on that lower-dimensional
   MLP policy) looks like the wrong default for this much higher-dimensional CNN-based policy
   on a harder task -- the two projects' policies apparently sit on opposite sides of the same
   knob. **Next experiment, not yet tried**: a small nonzero `ent_coef` (e.g. 0.01) to keep
   exploration alive longer, rather than further scaling the same recipe that already showed
   it won't self-correct.

6. **`ent_coef=0.01` (`grasp03`, fresh 1M-step run, not resumed -- resuming from `grasp02`
   would have carried forward its already-collapsed std) -- the diagnosed mechanism was fixed,
   but that wasn't sufficient to solve the task.** `std` held steady around 1.2-1.3 for the
   entire run (no collapse), and `approx_kl` stayed in a healthy range (well under 0.1 for most
   of training, only trivially near-zero at the very end as the linear LR schedule decayed to
   ~0). So the fix did exactly what it was diagnosed to do. Result: still **0/20**.

   The failure mode changed shape, though, which matters: `rise` across all 20 eval trials sat
   at essentially 0 (~-0.01m, noise-level settling) instead of grasp01/02's more erratic
   values. The policy isn't attempting a descend-grasp-lift sequence at all -- it's staying
   near its start state rather than drifting past the object or flailing. That's consistent
   with a different, likely deeper problem than exploration collapse: the task is thrown at
   full difficulty from step 0 (`random_table_xy` samples the entire table range on every
   single episode, from the very first one), with reward signal that's mostly a sparse +20
   terminal bonus plus small dense shaping -- there may simply not be enough early reward
   gradient for PPO to discover a multi-phase motor skill cold, across every object position
   at once, within this budget.

   **Lesson**: exploration collapse and insufficient task scaffolding are separate problems
   that can both be present -- fixing one (the collapse) is necessary but was not alone
   sufficient here. **Next experiment, not yet tried**: a position curriculum -- start
   `reset()` with a fixed or narrow object-position range, widen `TABLE_XY_RANGE` once reward
   crosses a threshold, so the policy can lock in a working grasp motion before having to
   generalize it across the full table.

7. **Position curriculum (`grasp04`, fresh 1M-step run, `ent_coef=0.01` kept + a
   `CurriculumCallback` widening `TABLE_XY_RANGE`'s sampled scale 0.15 -> 0.4 -> 0.7 -> 1.0
   whenever rolling success rate over the last 20+ episodes clears 30%) -- the curriculum
   mechanism itself verified correct (confirmed via a smoke test forcing `threshold=0.0`,
   which advanced through all 4 stages as expected), but it never advanced a single stage
   during the real run: success rate never cleared 30% even at the easiest, narrowest object
   range. Still **0/20** -- both on the standard full-range eval (expected, since it never
   trained past the narrow stage) and, more tellingly, on a second eval run at the exact
   narrow scale (0.15) it actually trained on the whole time. That second number rules out
   "learned narrow, failed to generalize wide" -- the policy never learned the core
   descend-close-lift skill at all, even at the easiest difficulty.

   That said, this run is qualitatively different from grasp03: `ep_rew_mean` reached -8.59
   (the best of any run so far), and eval `drift` values are now consistently nonzero
   (0.03-0.31m, vs. grasp03's mostly-exact-zero) -- the arm is reaching in and contacting/
   pushing the object, not just sitting idle. So the distance-minimization part of the
   behavior clearly improved; it's specifically the precise "close the gripper at the right
   moment while in contact, then lift without losing it" sequence that's never being
   discovered.

   **Lesson: the bottleneck isn't task-difficulty breadth (what this curriculum targeted) --
   it's the fine-grained grasp-closing maneuver itself being too hard to stumble into from
   sparse reward + pure exploration, within this budget, cold-start.** Four attempts
   (`grasp01`-`grasp04`) have each correctly diagnosed and fixed a real, distinct problem
   (partial learning -> exploration collapse -> task scaffolding) without producing a single
   success -- strong evidence the remaining gap is structural, not a tunable hyperparameter.
   **Next experiment, not yet tried**: stop iterating on pure cold-start RL hyperparameters:
   warm-start the policy from Stage 2's already-working scripted trajectory (behavior cloning
   pretraining, or a residual-RL formulation on top of it), the same fix `barrett_wam` used
   for its own analogous cold-start RL difficulty -- let RL correct/adapt an already-working
   grasp motion from pixels, rather than discover the whole thing from scratch.

8. **Behavior cloning + PPO fine-tuning (`bc01` then `grasp05`) -- DONE, a genuinely new
   failure mode identified: the warm-start worked, the fine-tuning erased it.**
   `collect_demos.py` replayed Stage 2's scripted controller through `GraspPixelEnv`'s exact
   observation/action pipeline (200/200 scripted successes, full object-position range, not the
   narrow curriculum), producing 24,000 (image, proprio, action) transitions. `train_bc.py`
   supervised the same PPO policy architecture's action-mean against those demonstrations via
   plain MSE (no environment interaction) -- loss dropped cleanly from 0.09 to 0.002 over 20
   epochs. `bc01` alone: still 0/20, but the best result of any run -- small positive `rise`
   on several trials, much better rewards, and (measured directly) a per-step joint-delta
   magnitude of 0.0088 rad, matching the scripted controller's own 0.0119 rad pace almost
   exactly. BC genuinely learned the gentle, precise approach.

   `grasp05` then fine-tuned that checkpoint with PPO (1M steps, `ent_coef=0.01`, same
   VecNormalize stats carried over) -- still 0/20, but **worse** than `bc01` alone, not better.
   Measured the same per-step joint-delta metric: fine-tuning pushed it from 0.0088 to 0.116
   rad, a 13x jump landing back in the same near-saturated regime `grasp04`'s failure GIF
   shows. **Root cause, not just correlation**: nothing in the reward penalizes how fast the
   arm approaches -- the action-magnitude term (`W_ACTION=0.001`) is deliberately tiny (same
   reward-hacking-avoidance choice as the distance term), and `gamma=0.98` discounting rewards
   reaching low distance *sooner* rather than carefully. PPO fine-tuning found it more
   rewarding to rush toward the object and knock it around than to preserve BC's careful,
   slow alignment -- it didn't correct BC's small residual errors (the intended use of RL
   fine-tuning on a good init), it overwrote the whole behavior.

   **Lesson: a warm-start only survives RL fine-tuning if the reward doesn't actively reward
   abandoning it.** The missing piece isn't exploration, scaffolding, or initialization
   anymore -- it's that nothing in `GraspPixelEnv`'s reward distinguishes a careful approach
   from a reckless one. **Next experiment, not yet tried**: add an approach-speed penalty
   (e.g. an action-magnitude cost that scales up specifically when `dist` is small, rather
   than a uniform one that would also slow down the reach phase), and/or lower
   `MAX_JOINT_RATE` itself so fast, careless motion isn't available to the policy at all.

9. **Approach-speed penalty (`grasp06`, PPO fine-tune from `bc01` again, with a
   proximity-gated `-W_APPROACH_SPEED * proximity * ||action||` term added to the reward) --
   DONE, mechanism confirmed correctly implemented but weighted far too weakly to matter.**
   Kept `MAX_JOINT_RATE` unchanged deliberately (lowering it would have silently rescaled
   `bc01`'s already-correct, validated action calibration -- its raw outputs were learned
   against the current rate constant). Premise-checked before training: replaying the scripted
   trajectory still gives `success=True`, reward 20.94 (vs. ~20.8 pre-change) -- the new term
   barely dents a genuinely careful trajectory, as intended.

   Result: still **0/20**, and the fix didn't actually change behavior -- measured directly,
   action magnitude at each episode's closest approach to the object was 0.1147 rad for
   `grasp06` vs. 0.1097 for `grasp05`, i.e. no improvement (if anything, slightly worse).
   **Root cause, found by breaking the reward into its components over a real rollout**: the
   penalty activates correctly, but its total contribution over an episode (`-0.33`) is
   dwarfed by the distance term's (`-20.0`) by roughly 60x -- the gripper is only within
   `PROXIMITY_RADIUS` for a handful of the 120 steps, and even there `W_APPROACH_SPEED=0.1`
   produces a per-step penalty of ~0.02-0.03, far too small to compete against an episode-long
   incentive to approach fast.

   **Lesson: a proximity-gated penalty applied with the same weight scale as the always-on
   distance term is automatically dominated by it, roughly in proportion to how much of the
   episode each one is active for.** A fix needs a substantially larger weight (not the same
   order of magnitude as `W_DIST`), or a different mechanism that doesn't have to outweigh an
   accumulated 120-step term from a handful of steps near contact.

   **Before spending training compute on a re-weighted guess, swept it properly instead of
   just multiplying by 60x.** That single-episode 60x ratio turned out not to generalize: it
   depends heavily on how long a rollout spends inside `PROXIMITY_RADIUS`, which differs a lot
   between a *successful* trajectory (long dwell -- descend, close, hold, lift all happen
   close to the object) and a *failed* one (barely gets close at all). Swept
   `W_APPROACH_SPEED` in `{0.1, 0.3, 0.4, 0.5, 1.0, 2.0, 3.0, 6.0}` against **both** the
   scripted trajectory (10 seeds -- must stay successful, reward shouldn't move far from its
   ~20.8 baseline) and `grasp06`'s own real rollout (10 seeds -- want its approach-speed total
   to become comparable to its distance total, not stay negligible). The naive `W=6.0` does
   blow up the fast policy's penalty, but it blows up the scripted trajectory's just as much
   (reward -11.8, down from +21.2, despite still succeeding) -- confirming the 60x ratio was
   an artifact of comparing a failed episode's short dwell time against an assumed-equivalent
   successful one. **`W=0.5`** is the actual balance point: scripted stays 10/10 at
   reward~19.1 (barely down from ~20.8), while `grasp06`'s approach-speed total (-14.6) is now
   the same order of magnitude as its distance total (-12.4) instead of ~0.33 vs -20 -- a real
   competing incentive against rushing, without wrecking the careful baseline it has to coexist
   with. Set `W_APPROACH_SPEED = 0.5` in `grasp_env.py` and launched `grasp07` (same recipe as
   `grasp05`/`grasp06`: PPO fine-tune from `bc01`, `ent_coef=0.01`, 1M steps).

   <p align="center">
     <img src="media/stage5_rl_failure.gif" width="480" alt="Trained RL policy (grasp04) descending toward the object, nudging it sideways, then neither closing the gripper around it nor lifting it, viewed from the fixed workspace camera">
   </p>

   **What the failure actually looks like** (`grasp04`, seed 0, same view the policy itself
   sees): the arm descends and makes contact with the object -- it clearly isn't flailing or
   idle -- but drifts it sideways (`drift=0.106m`) rather than centering, closing, and lifting.
   This is representative of all four runs' shared shape: real, visible progress on
   reach-and-approach, with the precise close-then-lift sequence never discovered.

   | Run | Change tested | Result | Root cause identified |
   |---|---|---|---|
   | `grasp01` | First training, 1M steps | 0/20, reward -12.7 (up from ~-24 random) | Genuine partial learning, not yet enough -- descends near the object but doesn't center/time the grasp |
   | `grasp02` | Scaled to 4M steps (same recipe) | 0/20, reward regressed to -12 | Exploration collapse: action-std fell 0.43->0.06 while `approx_kl` stayed abnormally high (0.8-1.6) the whole run -- policy locked onto a bad behavior early, `ent_coef=0.0` (right for `barrett_wam`'s MLP, wrong for this CNN policy) let it happen |
   | `grasp03` | `ent_coef=0.01` | 0/20, reward -8.59 best segment, but `rise`~0 every trial | Collapse fixed (`std` held ~1.2-1.3) but policy never attempts a grasp at all -- full-difficulty random positions from step 0 left too little early reward gradient |
   | `grasp04` | Position curriculum (0.15->1.0 scale) on top of `ent_coef=0.01` | 0/20 at full range AND at the narrow scale actually trained on | Not a generalization gap -- the core descend-close-lift skill was never learned even at the easiest difficulty; the grasp-closing maneuver itself is the bottleneck, not task-difficulty breadth |
   | `bc01` | Behavior cloning on Stage 2's scripted controller (200/200 demo episodes, no RL at all) | 0/20, but qualitatively the best result so far: several trials showed small positive `rise` (up to 0.0115m, vs. every RL run's flat zero/negative), rewards mostly -2 to -14 | Classic BC compounding-error: imitates the demonstrator well locally but never saw how to recover from small deviations, so drifts off-distribution before completing the sequence |
   | `grasp05` | PPO fine-tuning (1M steps, `ent_coef=0.01`) on top of `bc01`'s checkpoint | 0/20, **worse** than `bc01` alone (rewards mostly -5 to -25) | Fine-tuning didn't correct BC's small errors -- it overwrote the whole gentle approach. Measured directly: `bc01`'s mean per-step joint delta is 0.0088 rad, matching the scripted controller's own pace (0.0119 rad); after fine-tuning it's 0.116 rad, a 13x jump back into the same near-rate-cap regime `grasp04` shows in the GIF above. Nothing in the reward penalizes approach speed, and `gamma=0.98` discounting rewards reaching low distance *sooner* -- PPO found it more rewarding to rush in and knock the object around than to preserve BC's careful alignment |
   | `grasp06` | Proximity-gated approach-speed penalty added to the reward, PPO fine-tune from `bc01` again | 0/20, action magnitude at closest approach unchanged (0.1147 vs `grasp05`'s 0.1097 rad) | Penalty correctly wired (premise check: scripted trajectory reward barely moves, 20.94 vs ~20.8) but its accumulated magnitude over an episode (-0.33) is ~60x smaller than the distance term's (-20.0) -- numerically negligible, not a logic error |
   | `grasp07` | Same as `grasp06` but `W_APPROACH_SPEED` raised 0.1->0.5 (swept and verified, not a naive 60x guess), PPO fine-tune from `bc01` again | 0/20, `drift`~0.0000m (vs `grasp04`/`05`/`06`'s 0.03-0.3m) | New failure mode: the penalty didn't teach a careful approach, it taught retreat. `dist` briefly tightens to 0.10m by step 10-20, then the arm backs off to 0.69m by episode end while action magnitude stays high (mean 1.37) -- cheaper to leave `PROXIMITY_RADIUS` than pay the approach-speed cost, since this policy was never on a path to the terminal bonus anyway |
   | `grasp08` | Same as `grasp07`, plus a new `W_SMOOTH * ||a_t - a_t-1||^2` action-rate penalty (see below) | Running | -- |

   **Every attempt's rollout, not just one representative GIF** -- same `workspace_cam` view
   the policy itself sees, seed 0 for all seven so the object's start position is identical
   across runs and only the policy's behavior differs:

   | Run | Rollout |
   |---|---|
   | `grasp01` | <img src="media/grasp01.gif" width="280"> |
   | `grasp02` | <img src="media/grasp02.gif" width="280"> |
   | `grasp03` | <img src="media/grasp03.gif" width="280"> |
   | `grasp04` | <img src="media/stage5_rl_failure.gif" width="280"> |
   | `bc01` | <img src="media/bc01.gif" width="280"> |
   | `grasp05` | <img src="media/grasp05.gif" width="280"> |
   | `grasp06` | <img src="media/grasp06.gif" width="280"> |
   | `grasp07` | <img src="media/grasp07.gif" width="280"> |

10. **Action-rate "shaking" -- a separate failure mode from approach speed, found by actually
    watching the GIFs above (user's own observation): the arm visibly vibrates/shakes rather
    than moving smoothly, in every RL-trained run, not just the fast ones.** Quantified
    directly: for each arm-joint action component, how often it flips sign between consecutive
    steps, and the raw (unweighted) `sum(||a_t - a_t-1||^2)` over an episode.

    | Run | mean\|action\| | mean\|Δaction\| | sign-flip rate | raw smoothness total |
    |---|---|---|---|---|
    | `bc01` (imitation) | 0.031 | 0.003 | 2% | 0.018 |
    | scripted controller | -- | -- | -- | 0.046 (mean over 10 seeds) |
    | `grasp06` | 0.460 | 0.244 | 20% | 112.5 |
    | `grasp04` | 0.813 | 0.726 | 40% | 907.9 |
    | `grasp01` | 0.756 | 1.232 | 68% | 1675.1 |

    `bc01` and the scripted controller move almost perfectly smoothly (direction barely
    changes step to step); every RL-trained policy reverses direction on a large fraction of
    steps, and for `grasp01` the step-to-step change in action is actually *larger* than the
    action itself -- the arm is flipping between near-opposite joint commands almost every
    control tick (20Hz), which is what reads as "shaking" in the GIF.

    **Root cause**: nothing in the reward penalizes *changing* the action between steps, only
    its raw magnitude (`W_ACTION=0.001`, deliberately tiny) and (as of `grasp06`) proximity-
    gated speed -- both allow high-frequency oscillation as long as the average stays low.
    `ent_coef=0.01` (needed to avoid the earlier exploration-collapse problem) also keeps the
    action distribution's std near 1.0 throughout training, actively working against smooth
    output. `go1_rl_walk` already has the equivalent fix in its own reward
    (`--velocity-smoothness-weight`) -- this project never added it.

    **Fix**: added `W_SMOOTH * sum((action[:N_ARM] - prev_arm_action) ** 2)` to the reward.
    Calibrated the same way as `W_APPROACH_SPEED` -- checked the raw (unweighted) total against
    both the scripted trajectory (must stay negligible) and `grasp01`'s real rollout (the worst
    oscillator measured, want it penalized on the same order as `TERMINAL_BONUS`/the distance
    term) before picking a weight. `W_SMOOTH=0.01` gives `grasp01`-level oscillation a ~9-17
    penalty (depending on severity) while costing the scripted trajectory <0.001 -- premise-
    checked: scripted trajectory still 5/5 successful, mean reward 18.93 (vs. ~18.9-20.8 at
    earlier stages, barely moved). Launched `grasp08` (PPO fine-tune from `bc01`, both the
    approach-speed and smoothness fixes together) alongside `grasp07` (approach-speed alone),
    since there's CPU headroom to run both at once (16 cores, each run uses 8).

## Repository layout

```
panda_vision_grasp/
├── assets/
│   ├── panda.xml, hand.xml, scene.xml   # vendored from mujoco_menagerie, see NOTICE.md
│   ├── assets/                          # meshes the above MJCF references
│   ├── LICENSE.franka_emika_panda
│   ├── CHANGELOG.franka_emika_panda.md
│   ├── NOTICE.md                        # attribution + what's vendored vs. original here
│   └── grasp_scene.xml                  # Stage 2: table + graspable box + workspace_cam
├── ik.py                             # Stage 2: damped-least-squares Jacobian IK
├── control.py                        # Stage 2: gravity-compensation feedforward
├── eval_stage2_reach_grasp.py        # Stage 2: 20-trial scripted reach/grasp/lift sanity check
├── grasp_env.py                      # Stage 3: pixels-to-actions Gym env, see writeup above
├── train.py                          # Stage 4: PPO training, see README's RL section
├── eval_policy.py                    # Stage 4: 20-trial held-out eval for a trained policy
├── collect_demos.py                  # Stage 5: BC dataset from the scripted controller
├── train_bc.py                       # Stage 5: behavior-cloning pretrain on PPO's own network
├── runs/                             # training checkpoints/logs (gitignored, not committed)
├── data/                             # BC demo datasets (gitignored, not committed)
├── demo_stage1_kinematic_sweep.py    # Stage 1: verification sweep, see media/ GIF above
├── demo_stage2_reach_grasp.py        # Stage 2: scripted grasp demo, see media/ GIF above
├── demo_stage5_rl_failure.py         # Stage 5: renders a trained RL policy's failure mode
├── media/
│   ├── stage1_kinematic_sweep.gif
│   ├── stage2_reach_grasp.gif
│   ├── stage5_rl_failure.gif          # grasp04's rollout
│   ├── bc01.gif, grasp01.gif, grasp02.gif, grasp03.gif,
│   └── grasp05.gif, grasp06.gif       # every other attempt's rollout, see table above
├── requirements.txt
└── README.md
```
