# exp04 checkpoint choice on the validation throws

Made by `validate_checkpoints.py` (`run_exp04.sh` stage 4). The picked checkpoint of each run is the only one run on the test throws.

## goal (500 validation throws, seeds 0:500)

Run: `outputs/rl_exp04_goal/2026-11-17/23:54:22/nn`

| checkpoint | step | catch | 95% CI |
|---|---|---|---|
| ep_01000_step_0001m_reward_-1.20.pth **(picked)** | 1M | 10.8% | 8.4-13.8% |
| ep_02000_step_0002m_reward_0.58.pth | 2M | 10.0% | 7.7-12.9% |
| ep_03000_step_0003m_reward_-5.55.pth | 3M | 4.8% | 3.2-7.0% |
| ep_04000_step_0004m_reward_-1.18.pth | 4M | 9.0% | 6.8-11.8% |
| ep_05000_step_0005m_reward_0.17.pth | 5M | 7.6% | 5.6-10.3% |
| ep_06000_step_0006m_reward_4.13.pth | 6M | 8.0% | 5.9-10.7% |
| ep_07000_step_0007m_reward_2.00.pth | 7M | 9.0% | 6.8-11.8% |
| ep_08000_step_0008m_reward_-1.76.pth | 8M | 4.8% | 3.2-7.0% |
| ep_09000_step_0009m_reward_-7.43.pth | 9M | 3.6% | 2.3-5.6% |
| ep_10000_step_0010m_reward_-9.45.pth | 10M | 1.0% | 0.4-2.3% |
| ep_11000_step_0011m_reward_-12.71.pth | 11M | 3.4% | 2.1-5.4% |
| ep_12000_step_0012m_reward_-15.29.pth | 12M | 0.6% | 0.2-1.7% |
| ep_13000_step_0013m_reward_-11.68.pth | 13M | 0.2% | 0.0-1.1% |
| ep_14000_step_0014m_reward_-10.28.pth | 14M | 0.0% | 0.0-0.8% |
| ep_15000_step_0015m_reward_-8.81.pth | 15M | 0.0% | 0.0-0.8% |
| ep_16000_step_0016m_reward_-10.53.pth | 16M | 0.0% | 0.0-0.8% |
| ep_17000_step_0017m_reward_-10.15.pth | 17M | 0.0% | 0.0-0.8% |
| ep_18000_step_0018m_reward_-8.94.pth | 18M | 0.0% | 0.0-0.8% |
| ep_19000_step_0019m_reward_-9.41.pth | 19M | 0.0% | 0.0-0.8% |
| ep_20000_step_0020m_reward_-11.61.pth | 20M | 0.0% | 0.0-0.8% |
| ep_21000_step_0021m_reward_-10.23.pth | 21M | 0.0% | 0.0-0.8% |
| ep_22000_step_0022m_reward_-7.48.pth | 22M | 0.4% | 0.1-1.4% |
| ep_23000_step_0023m_reward_-10.07.pth | 23M | 0.0% | 0.0-0.8% |
| ep_24000_step_0024m_reward_-9.80.pth | 24M | 0.0% | 0.0-0.8% |
| ep_25000_step_0025m_reward_-8.68.pth | 25M | 0.8% | 0.3-2.0% |
| ep_26000_step_0026m_reward_-10.65.pth | 26M | 0.2% | 0.0-1.1% |
| ep_27000_step_0027m_reward_-7.69.pth | 27M | 1.0% | 0.4-2.3% |
| ep_28000_step_0028m_reward_-6.17.pth | 28M | 0.8% | 0.3-2.0% |
| ep_29000_step_0029m_reward_-7.25.pth | 29M | 1.0% | 0.4-2.3% |
| best_reward_6.90.pth | - | 9.4% | 7.1-12.3% |
| last.pth | - | 0.8% | 0.3-2.0% |

## delta (500 validation throws, seeds 0:500)

Run: `outputs/rl_exp03/2026-11-16/09:27:29/nn`

| checkpoint | step | catch | 95% CI |
|---|---|---|---|
| ep_01000_step_0001m_reward_2.36.pth | 1M | 11.0% | 8.5-14.0% |
| ep_02000_step_0002m_reward_-6.35.pth | 2M | 8.4% | 6.3-11.2% |
| ep_03000_step_0003m_reward_-9.55.pth | 3M | 11.2% | 8.7-14.3% |
| ep_04000_step_0004m_reward_-8.81.pth | 4M | 5.4% | 3.7-7.7% |
| ep_05000_step_0005m_reward_-7.45.pth | 5M | 4.6% | 3.1-6.8% |
| ep_06000_step_0006m_reward_-7.57.pth | 6M | 7.8% | 5.8-10.5% |
| ep_07000_step_0007m_reward_-6.09.pth | 7M | 7.8% | 5.8-10.5% |
| ep_08000_step_0008m_reward_-4.39.pth | 8M | 5.0% | 3.4-7.3% |
| ep_09000_step_0009m_reward_-5.25.pth | 9M | 7.4% | 5.4-10.0% |
| ep_10000_step_0010m_reward_-4.92.pth | 10M | 6.6% | 4.7-9.1% |
| ep_11000_step_0011m_reward_-5.79.pth | 11M | 7.8% | 5.8-10.5% |
| ep_12000_step_0012m_reward_-11.77.pth | 12M | 6.2% | 4.4-8.7% |
| ep_13000_step_0013m_reward_-4.85.pth | 13M | 11.4% | 8.9-14.5% |
| ep_14000_step_0014m_reward_-4.62.pth | 14M | 11.6% | 9.1-14.7% |
| ep_15000_step_0015m_reward_-2.12.pth | 15M | 10.8% | 8.4-13.8% |
| ep_16000_step_0016m_reward_-6.09.pth | 16M | 8.2% | 6.1-10.9% |
| ep_17000_step_0017m_reward_-3.19.pth | 17M | 9.0% | 6.8-11.8% |
| ep_18000_step_0018m_reward_-5.49.pth | 18M | 7.4% | 5.4-10.0% |
| ep_19000_step_0019m_reward_-9.29.pth | 19M | 4.6% | 3.1-6.8% |
| ep_20000_step_0020m_reward_-5.98.pth | 20M | 5.2% | 3.6-7.5% |
| ep_21000_step_0021m_reward_-9.95.pth | 21M | 3.2% | 2.0-5.1% |
| ep_22000_step_0022m_reward_-8.10.pth | 22M | 8.8% | 6.6-11.6% |
| ep_23000_step_0023m_reward_-9.98.pth | 23M | 5.2% | 3.6-7.5% |
| ep_24000_step_0024m_reward_-7.96.pth | 24M | 5.6% | 3.9-8.0% |
| ep_25000_step_0025m_reward_-11.55.pth | 25M | 6.4% | 4.6-8.9% |
| ep_26000_step_0026m_reward_-6.27.pth | 26M | 2.8% | 1.7-4.6% |
| ep_27000_step_0027m_reward_-8.15.pth | 27M | 1.6% | 0.8-3.1% |
| ep_28000_step_0028m_reward_-10.32.pth | 28M | 3.4% | 2.1-5.4% |
| ep_29000_step_0029m_reward_-17.78.pth | 29M | 1.4% | 0.7-2.9% |
| best_reward_9.78.pth **(picked)** | - | 15.4% | 12.5-18.8% |
| last.pth | - | 3.0% | 1.8-4.9% |
