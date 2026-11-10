# exp02: every disturbance randomized per throw

Run on 2026-11-10 with code commit `df856c0` (clean tree; recorded in every `.json`). 5000 test throws per controller (seeds 10000–14999). Seeds 10000–10999 are the same throws as exp01 (same target, launch point, flight time and ball); only the disturbances differ.

## Env settings (DcmmCfg at `df856c0`)

Drawn independently for each throw:

| Disturbance | Draw |
|---|---|
| Ball velocity obs noise | Gaussian, std ~ U(0.05, 0.18) m/s |
| Air drag | ball damping b ~ U(0, 0.005), force −b·v; speed lost over 1 s = 1 − exp(−b/m), up to 13.3% for a 35 g ball. The throw is aimed without drag, so the ball falls short of the aim point (median 3.8 cm, max 17.5 cm). |
| Action delay | the arm follows the command from 0, 1 or 2 steps ago (0/40/80 ms), equally likely |
| Gravity | fixed at 9.81 m/s² |

Unchanged from exp01, on every throw: ball position noise 5 mm, arm/hoop obs noise 0.001, torque noise 2.5%, PID gains ×U(0.75, 1.25), ball radius 35–45 mm, mass 35–75 g. The classical catchers are unchanged from exp01 (they assume no drag and `vel_sigma` 0.05). `classical_perfect` gets the true drag, so its prediction is exact.

## Catch rate

| Controller | Catch | 95% CI |
|---|---|---|
| random | 5.9% | 5.3–6.6% |
| classical_single | 57.2% | 55.8–58.5% |
| classical_multi | 96.1% | 95.5–96.6% |
| classical_perfect | 96.2% | 95.7–96.7% |
| oracle | 90.8% | 89.9–91.5% |

By disturbance (each row averages over the others, since they are drawn independently):

| | single | multi | perfect | oracle |
|---|---|---|---|---|
| delay 0 / 1 / 2 steps | 65 / 57 / 50% | 100 / 98 / 91% | 100 / 98 / 91% | 95 / 92 / 85% |
| velocity noise 0.05–0.09 / 0.09–0.14 / 0.14–0.18 m/s | 77 / 55 / 40% | 97 / 96 / 96% | 96 / 96 / 96% | 90 / 91 / 91% |
| drag speed loss 0–4 / 4–8 / 8–13% per s | 59 / 57 / 53% | 96 / 96 / 96% | 95 / 97 / 97% | 94 / 91 / 79% |

## Reading the miss reasons

In `summary.md`, most `classical_multi` misses are labelled "not reachable in time". That label is a side effect of delay, not a real reachability problem:
- When the arm lags, the catcher re-picks its catch time late (1.5 re-picks per throw at 0 delay, 2.7 at 2 steps), and the late plan is logged as infeasible.
- 92% of the throws flagged that way are still caught.
- The better evidence is the delay split: 157 of multi's 194 misses are at 2-step delay, and `classical_perfect` (exact prediction) misses on the same pattern (156 of 189).

Files: [summary.md](summary.md) (all tables), `<controller>.csv` (one row per throw, including its draws), `<controller>.json` (full env and catcher settings, commit).

## Reproduce

```bash
git checkout df856c0 && cd catch_it
EXP=exp02_randomized SEEDS=10000:15000 bash run_eval.sh
```
