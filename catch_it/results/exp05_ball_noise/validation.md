# exp05 checkpoint choice on the validation throws

Made by `validate_checkpoints.py` (`run_exp05.sh` stage 5). The picked checkpoint is the only one run on the test throws.

## rl (500 validation throws, seeds 0:500)

Run: `outputs/rl_exp05/2026-11-19/08-48-56/nn`

| checkpoint | step | catch | 95% CI |
|---|---|---|---|
| ep_125_step_0001m_reward_17.31.pth | 1M | 32.6% | 28.6-36.8% |
| ep_375_step_0003m_reward_61.96.pth | 3M | 75.2% | 71.2-78.8% |
| ep_500_step_0004m_reward_79.10.pth | 4M | 90.6% | 87.7-92.9% |
| ep_625_step_0005m_reward_100.29.pth | 5M | 98.4% | 96.9-99.2% |
| ep_750_step_0006m_reward_103.98.pth | 6M | 99.4% | 98.3-99.8% |
| ep_875_step_0007m_reward_105.73.pth | 7M | 99.2% | 98.0-99.7% |
| ep_1000_step_0008m_reward_108.22.pth | 8M | 99.6% | 98.6-99.9% |
| ep_1125_step_0009m_reward_108.72.pth | 9M | 99.8% | 98.9-100.0% |
| ep_1250_step_0010m_reward_109.01.pth | 10M | 99.8% | 98.9-100.0% |
| ep_1375_step_0011m_reward_109.99.pth | 11M | 99.8% | 98.9-100.0% |
| ep_1500_step_0012m_reward_110.46.pth | 12M | 99.8% | 98.9-100.0% |
| ep_1625_step_0013m_reward_110.34.pth | 13M | 99.8% | 98.9-100.0% |
| ep_1750_step_0014m_reward_110.37.pth **(picked)** | 14M | 100.0% | 99.2-100.0% |
| ep_1875_step_0015m_reward_110.96.pth | 15M | 99.8% | 98.9-100.0% |
| ep_2000_step_0016m_reward_110.71.pth | 16M | 100.0% | 99.2-100.0% |
| ep_2125_step_0017m_reward_110.86.pth | 17M | 100.0% | 99.2-100.0% |
| ep_2250_step_0018m_reward_110.10.pth | 18M | 99.6% | 98.6-99.9% |
| ep_2375_step_0019m_reward_110.85.pth | 19M | 100.0% | 99.2-100.0% |
| ep_2500_step_0020m_reward_110.59.pth | 20M | 99.6% | 98.6-99.9% |
| ep_2625_step_0021m_reward_110.56.pth | 21M | 99.8% | 98.9-100.0% |
| ep_2750_step_0022m_reward_110.14.pth | 22M | 99.8% | 98.9-100.0% |
| ep_2875_step_0023m_reward_110.57.pth | 23M | 100.0% | 99.2-100.0% |
| ep_3000_step_0024m_reward_109.75.pth | 24M | 99.4% | 98.3-99.8% |
| best_reward_111.42.pth | - | 100.0% | 99.2-100.0% |
| last.pth | - | 100.0% | 99.2-100.0% |
