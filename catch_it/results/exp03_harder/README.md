# exp03: stronger per-throw disturbances, wind, and aimed throws

Run on 2026-11-11 with code commit `c403bf3` (clean tree; recorded in every `.json`). 5000 test throws per controller (seeds 10000–14999).

Each seed has the same target, launch point, flight time and ball as in exp01 and exp02 (checked on all 5000 seeds). Only the disturbances and the launch velocity differ, because the launch velocity is now aimed with the drag and wind.

## Env settings (DcmmCfg at `c403bf3`)

Drawn independently for each throw:

| Disturbance | exp02 | exp03 | Why this range |
|---|---|---|---|
| Action delay | 0, 1 or 2 steps (0–80 ms) | 0–4 steps (0–160 ms), equally likely | camera → detection → arm latency |
| Arm PID gains | ×0.75–1.25 per joint | ×0.5–1.5 per joint | payload and tuning differences |
| Wind | none | horizontal force 0–0.04 N, random direction, from release on | about a 5 m/s breeze on a 4 cm ball (½ρC_dAv²) |
| Air drag b | 0–0.005 (up to 13% speed lost per s) | 0–0.015 (up to 35% for a 35 g ball) | a light foam ball |
| Ball position noise std | 5 mm (fixed) | 5–20 mm | a good to mediocre depth camera at 2–3 m |
| Ball velocity noise std | 0.05–0.18 m/s | 0.1–0.4 m/s | lightly filtered differences of those positions |
| Torque noise | ×N(1, 0.025) | ×N(1, 0.10) | motor current ripple and friction |
| Throw aim | as if no drag (falls short) | with the drag and wind, through the target | keeps every throw catchable (see below) |
| Gravity | 9.81 | 9.81 | |

Unchanged from exp02: arm and hoop observation noise 0.001, ball radius 35–45 mm and mass 35–75 g, and the throw sampler.

**Why aim with the drag and wind.** With these ranges, a throw aimed as if neither existed would leave 26% of throws never passing through the catch zone (pilot, validation seeds). The catch rate would then mostly count impossible throws. Aimed with them, every throw passes through its target (`aim_miss` is 0 on all 5000 throws), but the path is not a parabola, which is what the classical catcher assumes.

## Classical catcher settings (ClassicalCfg at `c403bf3`)

The catcher's assumed sensor noise and release thresholds are set from the noise ranges, the way an engineer sets them from a sensor data sheet:

| Setting | exp02 | exp03 |
|---|---|---|
| `pos_sigma` | 5 mm | 12.5 mm |
| `vel_sigma` | 0.05 m/s | 0.25 m/s |
| `release_disp` | 3 cm | 8 cm |
| `release_speed` | 1.0 m/s | 1.6 m/s |

These values were checked only on validation seeds. With exp02's catcher settings, noise alone triggered the release detector on 5.8% of validation throws while the ball was still held, and the catch rate was 68.6% instead of 75.4%. The catcher's model is unchanged: it assumes no drag, no wind and no delay.

## How the settings were chosen

Each change was piloted alone on validation seeds 0–499 (500 throws, about ±4 points). The test seeds were not used until this run. `classical_multi` under exp02 settings caught 96% of these throws.

| Change alone | `classical_multi` |
|---|---|
| delay 0–4 steps | 88% (68% at 4 steps) |
| wind 0–0.04 N | 89% |
| PID ×0.5–1.5 | 93% |
| velocity noise 0.1–0.4 m/s | 94% |
| drag 0–0.015 | 96% |
| position noise 20 mm | 96% |
| torque noise 10% | 97% |

All changes together landed at 75% on validation; these are the settings tested here.

## Catch rate

| Controller | exp03 | 95% CI | exp02 |
|---|---|---|---|
| random | 5.0% | 4.5–5.7% | 5.9% |
| classical_single | 21.5% | 20.4–22.7% | 57.2% |
| classical_multi | 76.0% | 74.8–77.2% | 96.1% |
| classical_perfect | 81.4% | 80.3–82.5% | 96.2% |
| oracle | 84.4% | 83.3–85.4% | 90.8% |

Paired against `classical_multi` (same throws):
- `classical_perfect` catches 560 throws that multi misses, while multi catches 289 that perfect misses (p = 9·10⁻²¹).
- `oracle` catches 898 throws that multi misses, while multi catches 479 that the oracle misses (p = 7·10⁻³⁰).

By disturbance (each row averages over the others, since they are drawn independently):

| | single | multi | perfect | oracle |
|---|---|---|---|---|
| delay 0 / 1 / 2 / 3 / 4 steps | 23 / 22 / 20 / 22 / 21% | 99 / 91 / 76 / 65 / 51% | 99 / 96 / 87 / 69 / 56% | 96 / 95 / 86 / 73 / 72% |
| wind across the throw 0–0.01 / 0.01–0.02 / 0.02–0.04 N | 23 / 21 / 20% | 80 / 75 / 70% | 82 / 80 / 83% | 84 / 84 / 85% |
| weakest joint's PID gain 0.5–0.6 / 0.6–0.75 / 0.75–1.5 | 22 / 21 / 21% | 73 / 78 / 80% | 79 / 83 / 86% | 83 / 85 / 86% |
| drag speed loss 0–10 / 10–20 / 20–35% per s | 22 / 23 / 19% | 77 / 77 / 72% | 81 / 82 / 81% | 83 / 85 / 87% |
| velocity noise 0.1–0.2 / 0.2–0.3 / 0.3–0.4 m/s | 28 / 19 / 18% | 75 / 76 / 77% | 80 / 82 / 83% | 85 / 84 / 85% |
| position noise 5–10 / 10–15 / 15–20 mm | 22 / 21 / 22% | 76 / 77 / 75% | 80 / 82 / 83% | 85 / 84 / 85% |

## What the results say

- **Delay is still the biggest factor.**
  - `classical_multi` catches 99% with no delay and 51% at 4 steps (160 ms).
  - 843 of its 1200 misses are at 3–4 steps.
  - `classical_perfect`, which predicts exactly, drops almost as much (99 → 56%), so most of the delay loss comes from the arm arriving late, not from prediction.
- **Wind is where prediction matters.**
  - `classical_multi`'s parabola cannot represent a sideways push. As the crosswind grows, its median cross-path prediction error rises from 1.7 to 2.3 to 3.3 cm.
  - Its catch rate falls from 80% to 70%, while `classical_perfect` (which knows the wind) stays at about 82%.
  - The gap between perfect and multi grows from 2 to 4 to 13 points with crosswind.
- **Sensor noise does not affect `classical_multi`.** It stays at 75–77% across the position-noise and velocity-noise bins, because its fit averages every read since the release. `classical_single`, which relies on one read, drops from 28% to 18% with velocity noise.
- **Weak arm gains cost 7 points**: 73% when the weakest joint is at ×0.5–0.6, against 80% above ×0.75.
- **The oracle (84.4%) beats `classical_multi`.** It knows the aim point before the release, so it is already moving when the ball leaves: at 4 steps of delay it still catches 72%, against multi's 51%. It is a privileged reference, not a competitor.
- **Don't trust the miss-reason labels for `classical_multi`.**
  - `summary.md` puts most of its misses under "not reachable in time".
  - That flag is set when a late re-plan finds the new catch point too close in time. Under delay this happens constantly: 2.8 re-picks per throw at 0 delay, 5.5 at 4 steps.
  - 73% of all its throws carry the flag, and 69% of those are still caught.
  - The delay and wind tables above are the reliable evidence.

## For the RL comparison

- **Delay (2–4 steps) and wind are where the classical catcher loses.** There is room for a learned policy to move earlier under delay and to react to the ball's actual path in wind. At 0–1 steps of delay, `classical_multi` catches 94.5%.
- **The sensor noise barely touches the classical catcher but will hurt a memoryless policy.** Its fit averages about 20 reads per throw. A plain MLP sees one frame, and at 0.4 m/s velocity noise its estimate of where the ball will be 0.25 s later is off by about 10 cm. Give the policy a short observation history (for example, the last 4 ball observations).
- **The policy must be trained with these settings** (the `DcmmCfg.py` defaults at `c403bf3`). The env changes the RL branch needs are listed in `docs/classical_baseline.md` §8.
- **Freeze these settings before any RL test run, and keep exp02 in the report.**

Files:
- [summary.md](summary.md): all tables.
- `<controller>.csv`: one row per throw, including that throw's draws.
- `<controller>.json`: full env and catcher settings, plus the commit.

## Reproduce

```bash
git checkout c403bf3 && cd catch_it
EXP=exp03_harder SEEDS=10000:15000 bash run_eval.sh
```
