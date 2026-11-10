# exp01: one factor per condition

Run on 2026-11-09. Each condition changes one simulator setting from nominal; every controller faces the same 1000 test throws (seeds 10000–10999) in every condition.

**Code:** commit `ed79c76`. The `.json` files say `8510670` + dirty because the code was committed after the run; re-running `delay_2/classical_multi` at `ed79c76` reproduces its CSV byte for byte.

## Env settings (DcmmCfg at `ed79c76`)

| Setting | Nominal value |
|---|---|
| Ball position obs noise | 5 mm (Gaussian std) |
| Ball velocity obs noise | 0.05 m/s (Gaussian std) |
| Gravity | 9.81 ± 0.5 m/s² (uniform, per throw) |
| Air drag (ball damping) | 0 |
| Action delay | none |
| Always on | arm/hoop obs noise 0.001, action noise 2.5%, PID gains ×U(0.75, 1.25), ball radius 35–45 mm, mass 35–75 g |

## Conditions

| Condition | Override of nominal |
|---|---|
| `nominal` | none |
| `noise_x2` | ball obs noise 10 mm, 0.10 m/s |
| `noise_x4` | ball obs noise 20 mm, 0.20 m/s |
| `tracker_noise` | ball velocity noise 0.18 m/s |
| `gravity_fixed` | gravity exactly 9.81 |
| `delay_1` | arm follows the command from 1 step (40 ms) ago |
| `delay_2` | arm follows the command from 2 steps (80 ms) ago |

## Catch rate

| Controller | nominal | noise ×2 | noise ×4 | tracker noise | gravity fixed | delay 1 | delay 2 |
|---|---|---|---|---|---|---|---|
| random | 4.6% | | | | | | |
| classical_single | 95.7% | 71.5% | 33.7% | 39.0% | 99.2% | 82.6% | 70.2% |
| classical_multi | 99.7% | 99.7% | 99.6% | 99.4% | 99.8% | 97.8% | 89.3% |
| classical_perfect | 99.8% | 99.8% | 99.8% | 99.8% | 99.9% | 98.0% | 88.8% |
| oracle | 94.9% | 94.9% | 94.9% | 94.9% | 95.0% | 94.7% | 88.3% |

Full tables (CIs, paired tests, miss reasons, per-bin rates): [summary.md](summary.md). Per-episode data: `<condition>/<controller>.csv`; run settings: `<condition>/<controller>.json`.

## Reproduce

```bash
git checkout ed79c76 && cd catch_it
python eval_catch.py --controller classical_multi --condition delay_2 --out_dir /tmp/repro
bash run_eval_matrix.sh      # everything
```
