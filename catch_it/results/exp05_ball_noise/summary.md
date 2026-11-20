# HoopCatch evaluation summary: exp05_ball_noise

## Condition: exp05_ball_noise

| controller        | n    | catch | 95% CI      | ball_floor | ball_robot | collision | timeout | rim hits/ep | IK fails/ep | hoop travel (m) | action change^2 |
|-------------------|------|-------|-------------|------------|------------|-----------|---------|-------------|-------------|-----------------|-----------------|
| random            | 5000 | 4.8%  | 4.2%-5.4%   | 49.4%      | 38.8%      | 7.0%      | 0.0%    | 0.22        | 1.82        | 0.88            | 5.082           |
| classical_single  | 5000 | 31.8% | 30.6%-33.1% | 37.8%      | 30.1%      | 0.2%      | 0.0%    | 0.66        | 2.20        | 1.00            | 0.568           |
| classical_multi   | 5000 | 99.8% | 99.6%-99.9% | 0.2%       | 0.0%       | 0.0%      | 0.0%    | 0.01        | 0.48        | 1.00            | 0.921           |
| rl                | 5000 | 99.8% | 99.6%-99.9% | 0.2%       | 0.0%       | 0.0%      | 0.0%    | 0.03        | 0.02        | 0.88            | 0.684           |
| classical_perfect | 5000 | 99.8% | 99.6%-99.9% | 0.2%       | 0.0%       | 0.0%      | 0.0%    | 0.00        | 0.34        | 0.92            | 0.620           |
| oracle            | 5000 | 95.7% | 95.1%-96.2% | 4.0%       | 0.3%       | 0.0%      | 0.0%    | 0.07        | 4.01        | 0.65            | 0.356           |

Paired McNemar vs classical_multi (same seeds):

| controller        | paired n | only classical_multi caught | only other caught | p     |
|-------------------|----------|-----------------------------|-------------------|-------|
| random            | 5000     | 4750                        | 0                 | 0     |
| classical_single  | 5000     | 3400                        | 3                 | 0     |
| rl                | 5000     | 9                           | 10                | 1     |
| classical_perfect | 5000     | 4                           | 5                 | 1     |
| oracle            | 5000     | 207                         | 4                 | 5e-56 |

Classical diagnostics (prediction error = predicted catch point vs true path, across the path):

| controller        | pred err median (cm) | pred err p90 (cm) | hoop err median (cm) | first plan after release (ms) | plan reachable | g estimate (median) |
|-------------------|----------------------|-------------------|----------------------|-------------------------------|----------------|---------------------|
| classical_single  | 18.7                 | 42.7              | 3.0                  | 78                            | 61.7%          | 9.81                |
| classical_multi   | 0.7                  | 1.4               | 3.3                  | 78                            | 74.8%          | 9.81                |
| classical_perfect | 0.0                  | 0.0               | 2.0                  | 78                            | 97.6%          | 9.81                |

Why the classical controllers missed (count of misses):

| controller        | arm tracking | not reachable in time | prediction |
|-------------------|--------------|-----------------------|------------|
| classical_single  | 1            | 1673                  | 1734       |
| classical_multi   | 0            | 11                    | 0          |
| classical_perfect | 7            | 3                     | 0          |

Catch rate by position noise std (mm):

| position noise std (mm) | random | classical_single | classical_multi | rl   | classical_perfect | oracle | n    |
|-------------------------|--------|------------------|-----------------|------|-------------------|--------|------|
| 5-10                    | 5%     | 33%              | 100%            | 100% | 100%              | 95%    | 1685 |
| 10-15                   | 5%     | 31%              | 100%            | 100% | 100%              | 97%    | 1648 |
| 15-20                   | 4%     | 31%              | 100%            | 100% | 100%              | 96%    | 1667 |

Catch rate by velocity noise std (m/s):

| velocity noise std (m/s) | random | classical_single | classical_multi | rl   | classical_perfect | oracle | n    |
|--------------------------|--------|------------------|-----------------|------|-------------------|--------|------|
| 0.1-0.2                  | 4%     | 53%              | 100%            | 100% | 100%              | 95%    | 1697 |
| 0.2-0.3                  | 4%     | 25%              | 100%            | 100% | 100%              | 96%    | 1637 |
| 0.3-0.4                  | 5%     | 17%              | 100%            | 100% | 100%              | 96%    | 1666 |

Catch rate by weakest joint's PID gain scale:

| weakest joint's PID gain scale | random | classical_single | classical_multi | rl   | classical_perfect | oracle | n    |
|--------------------------------|--------|------------------|-----------------|------|-------------------|--------|------|
| 0.50-0.60                      | -      | -                | -               | -    | -                 | -      | 0    |
| 0.60-0.75                      | -      | -                | -               | -    | -                 | -      | 0    |
| 0.75-1.50                      | 5%     | 32%              | 100%            | 100% | 100%              | 96%    | 5000 |

Catch rate by flight time (s):

| flight time (s) | random | classical_single | classical_multi | rl   | classical_perfect | oracle | n    |
|-----------------|--------|------------------|-----------------|------|-------------------|--------|------|
| (0.599, 0.7]    | 5%     | 42%              | 99%             | 100% | 99%               | 94%    | 1217 |
| (0.7, 0.8]      | 4%     | 34%              | 100%            | 100% | 100%              | 96%    | 1256 |
| (0.8, 0.9]      | 5%     | 29%              | 100%            | 100% | 100%              | 96%    | 1278 |
| (0.9, 1.0]      | 5%     | 23%              | 100%            | 100% | 100%              | 97%    | 1249 |

Catch rate by launch azimuth (deg):

| launch azimuth (deg) | random | classical_single | classical_multi | rl   | classical_perfect | oracle | n    |
|----------------------|--------|------------------|-----------------|------|-------------------|--------|------|
| (-60.001, -30.0]     | 5%     | 31%              | 100%            | 99%  | 100%              | 97%    | 1256 |
| (-30.0, 0.0]         | 5%     | 31%              | 100%            | 100% | 100%              | 98%    | 1218 |
| (0.0, 30.0]          | 5%     | 33%              | 100%            | 100% | 100%              | 97%    | 1260 |
| (30.0, 60.0]         | 4%     | 32%              | 99%             | 100% | 99%               | 91%    | 1266 |

Catch rate by home-to-target distance (m):

| home-to-target distance (m) | random | classical_single | classical_multi | rl   | classical_perfect | oracle | n    |
|-----------------------------|--------|------------------|-----------------|------|-------------------|--------|------|
| (-0.001, 0.2]               | 9%     | 30%              | 100%            | 100% | 100%              | 99%    | 658  |
| (0.2, 0.3]                  | 6%     | 32%              | 100%            | 100% | 100%              | 93%    | 1153 |
| (0.3, 0.4]                  | 4%     | 32%              | 100%            | 100% | 100%              | 95%    | 1429 |
| (0.4, 0.5]                  | 2%     | 32%              | 100%            | 100% | 100%              | 97%    | 1246 |
| (0.5, 0.8]                  | 3%     | 32%              | 99%             | 99%  | 99%               | 96%    | 514  |
