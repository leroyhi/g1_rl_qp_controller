> **Note (v19 bundle, `weak_v19_tqnorm`):** `weak.onnx`, `estimator.onnx`, `estimator.json`, `limit_kernel.json` and
> `manifest.json` come from `deploy_v19`. The estimator now reads 13 features (`tilt`, `feet`, `dcm`, `shoulders`,
> `ang_mom`, see `estimator.json`) with `thr` 0.9166, and the kernel's threshold is 47.09. The rest of this README
> describes the previous bundle (37 blind features, `thr` 0.825).

# G1 irrecoverable-fall switch — deployment bundle

Three ONNX graphs and one ellipsoid. A low-gain walking policy walks; an estimator
watches and says when switching to the catch controller is about to stop working;
the catch controller takes over; a limit kernel says when walking can resume.

Everything that is not weights is in this directory three times over: embedded in
each `.onnx` file's metadata (one `fall_switch` key holds the whole block as JSON,
so a single file is self-describing), in the JSON sidecars next to it, and
described here. The numbers were read out of the live simulation at export time,
not transcribed.

| file | role | input | output |
| --- | --- | --- | --- |
| `weak.onnx` | walking policy | `obs (batch, 98)` | `action (batch, 29)` |
| `recovery_v0.onnx` | catch controller | `obs (batch, 98)`, same layout | `action (batch, 29)` |
| `estimator.onnx` | when a switch stops working | `window (batch, 5, 37)` | `logits (batch, 5)` |
| `limit_kernel.json` | when walking can resume | 37 blind features | inside / outside |
| `plant.json` | the machine, for your QP | | |
| `observation.json` | how to build the 98-vector | | |
| `estimator.json` | standardisation and switch rule | | |
| `manifest.json` | hashes and export-time verification | | |

Both policies take **the same 98-wide observation in the same order**. That is
deliberate: the walking policy was trained with Unitree's observation layout
precisely so the two are interchangeable at the handover, and you build one vector
per control step regardless of who holds control.

## The loop

```
every 20 ms:
    obs   = build_observation()                  # observation.json, 98 floats
    phys  = build_phys_features()                # estimator.json, 37 floats
    push(window, standardise(phys))              # 5 frames, oldest first

    if holding_recovery:
        inside = mahalanobis2(phys) <= kernel.threshold
        dwell  = dwell + 1 if inside else 0
        if dwell >= 25:                          # 0.5 s
            holding_recovery = False             # hand control back
            restore_operator_command()
    else:
        p = sigmoid(estimator(window))[4]        # horizon 40 steps = 0.8 s
        if p > 0.825:
            holding_recovery = True
            command = 0                          # REQUIRED, see below
            dwell = 0

    a = (recovery_v0 if holding_recovery else weak)(obs)
    target = a * action_scale + action_offset    # plant.json, per joint
    tau = your_qp(target, q, qd, limits)         # see "What the QP has to do"
```

The action vector is in natural joint order, the same order as `plant.joints`
(`plant.action_order_equals_joint_order` is `true` — checked at export, not
assumed). `action_offset` equals the default pose, so a zero action commands the
reference crouch.

## Two things that will silently break it

**Command zero at the handover.** The estimator's labels were produced with a zero
twist commanded to the catch controller from the switch onward. A non-zero command
there does not degrade the result, it invalidates the label the threshold was set
against. It also matters mechanically: `phase` is forced to `(0, 0)` whenever
`norm(command) < 0.1`, so the gait clock stops, and the catch controller was
measured in that regime only.

**`actions` in the observation is the raw action of whoever acted last.** Not the
walking policy's action, not the post-scale target. After a switch it is the catch
controller's own output, which is what both policies saw in training.

## What the QP has to do

Neither policy knows anything about limits. They emit joint position targets, and
in simulation those targets are realised by a PD law that clips at the actuator's
torque ceiling:

```
tau = clip(kp * (target - q) - kd * qd, -torque_limit, +torque_limit)
```

`plant.json` carries, per joint: `kp`, `kd` (nominal — the walking policy trains
with ±10% randomisation around these, so a live sample is not what to ship),
`torque_limit`, `velocity_limit`, `pos_limit_hard` (the MJCF range) and
`pos_limit_soft` (mjlab's 0.9 shrink about the range centre, the band the training
penalties use and a sane bound for a QP).

Two measured facts that bear on the safety layer:

- MuJoCo enforces joint ranges as **compliant constraints, not walls**. Under a
  strong push we measured ankles driven about 0.05 rad past their hard stop, and a
  knee peaking at 140% of its velocity limit, in a run where the robot never fell.
  Nothing in these graphs prevents that; on hardware those bounds are mechanical.
  This is the main reason to put a QP between the policy and the motors.
- The torque ceiling is reached routinely during a catch. A joint pinned at its
  limit stops being a position source, so a QP that assumes the target is tracked
  will be wrong exactly when it matters most.

The limit figures above were measured on a *different* (stronger) recovery policy
than the `recovery_v0` shipped here; treat them as the order of magnitude to design
for, not as this bundle's numbers.

## Estimator input

37 blind features — IMU and joint encoders only, nothing needing the base linear
velocity, which the G1 does not measure. The full column-by-column definition, with
the two easy mistakes called out (`support` uses a gravity-aligned heading frame,
not the pelvis frame; `track_err` and `tau_sat` are not interchangeable), is in
`../estimator/README.md`. The reference implementation is `physics_features()` in
`scripts/fall_switch/common.py` — the exact code the training data came from.

Standardise before the graph: `z = clamp((x - mu) / sd, -8, 8)`, with `mu` and `sd`
in `estimator.json`. The graph does **not** include this step. Shape is
`(batch, 5, 37)`, oldest frame first; edge-pad with the oldest available frame
while the buffer fills.

The output is five **logits**. `sigmoid(logits[i])` is the probability that
switching stops working within `horizons[i]` steps at 50 Hz, for
`horizons = [0, 5, 10, 20, 40]`. Switch on the last one.

## Measured performance

Closed loop in simulation, 1024 environments, 9 s episodes, one instantaneous push
per episode of 0.3–1.0 m/s in a random direction, mild uneven terrain. This is
`estimator.onnx` at the threshold in `estimator.json` (0.825), i.e. exactly what is
shipped here.

| | falls | switched | walking time lost | false switches/min |
| --- | --- | --- | --- | --- |
| walking policy alone | 61.2% | — | — | — |
| + estimator, no hand-back | 13.0% | 62.1% | 38.6% | 0.89 |
| + estimator + limit kernel | **12.3%** | 60.7% | **15.1%** | 0.90 |

Read the two protected rows together: the hand-back is what makes the protection
affordable. Holding the catch controller to the end of the episode costs 38.6% of
walking time; releasing it when the state re-enters the limit kernel costs 15.1%,
and the fall rate does not pay for it. Control went back at least once in 48.5% of
episodes, 0.65 switches per episode.

The run-to-run spread of the unprotected control row is about 0.6 points (61.2 vs
60.8 across these two runs at identical seeds — the simulator is not bit-exact), so
differences below that are noise, including the apparent 13.0 → 12.3 improvement.

## Limits you should design around

**About 0.9 false switches per minute** of undisturbed walking at the shipped
threshold. Too high for long autonomous operation. Raising the threshold trades
falls for false alarms, but the false-alarm rate barely moves with it, so it is not
simply a dial.

**Roughly 13% of falls are already unrecoverable at the instant of the push**, and
no detector can address them: the catch controller's basin is too small. A further
~12% leave under 0.2 s to react. The lever on both is a wider-basin catch
controller, not a better estimator — `recovery_v0` is a walking policy trained with
small pushes, not a dedicated catch controller. It recovers instantaneous pushes to
~1.0 m/s reliably and 2.0 m/s in 83% of cases.

**The estimator is bound to this exact pair.** It was trained against
`weak.onnx` and `recovery_v0.onnx`, both frozen. Change either and the label — "the
last moment at which switching still works" — refers to something else. Retrain.

**Trained with a sensor-noise model**, not with the simulator's domain
randomisation: encoders 1e-3 rad, gyro 5e-3 rad/s, IMU attitude 9e-3 rad, link
positions 6 mm, motor current 2% of the limit. Noise within that cost nothing in
simulation; measured noise beyond it will degrade the estimator.

## Export verification

Each graph was checked against its torch reference on real observations from a
rollout, at batch 1 and at full batch. `manifest.json` records the maximum absolute
differences; they are at single-precision rounding (around 1e-6). The batch
dimension is symbolic in all three graphs, and batch-1 and batch-N results agree
exactly.

Regenerate with `uv run python scripts/fall_switch/export_onnx.py`.
