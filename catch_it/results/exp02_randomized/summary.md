# HoopCatch evaluation summary: exp02_randomized

## Condition: exp02_randomized

| controller        | n    | catch | 95% CI      | ball_floor | ball_robot | collision | timeout | rim hits/ep | IK fails/ep | hoop travel (m) | action change^2 |
|-------------------|------|-------|-------------|------------|------------|-----------|---------|-------------|-------------|-----------------|-----------------|
| random            | 5000 | 5.9%  | 5.3%-6.6%   | 51.3%      | 35.9%      | 6.9%      | 0.0%    | 0.27        | 1.84        | 0.87            | 5.083           |
| classical_single  | 5000 | 57.2% | 55.8%-58.5% | 30.4%      | 12.2%      | 0.1%      | 0.1%    | 0.88        | 1.15        | 1.06            | 0.749           |
| classical_multi   | 5000 | 96.1% | 95.5%-96.6% | 3.3%       | 0.5%       | 0.0%      | 0.0%    | 0.22        | 0.92        | 1.03            | 0.991           |
| classical_perfect | 5000 | 96.2% | 95.7%-96.7% | 3.2%       | 0.5%       | 0.0%      | 0.0%    | 0.17        | 0.71        | 0.99            | 0.774           |
| oracle            | 5000 | 90.8% | 89.9%-91.5% | 8.3%       | 0.8%       | 0.0%      | 0.2%    | 0.41        | 4.50        | 0.84            | 0.559           |

Paired McNemar vs classical_multi (same seeds):

| controller        | paired n | only classical_multi caught | only other caught | p       |
|-------------------|----------|-----------------------------|-------------------|---------|
| random            | 5000     | 4521                        | 9                 | 0       |
| classical_single  | 5000     | 2008                        | 61                | 0       |
| classical_perfect | 5000     | 101                         | 106               | 0.78    |
| oracle            | 5000     | 425                         | 157               | 1.7e-29 |

Classical diagnostics (prediction error = predicted catch point vs true path, across the path):

| controller        | pred err median (cm) | pred err p90 (cm) | hoop err median (cm) | first plan after release (ms) | plan reachable | g estimate (median) |
|-------------------|----------------------|-------------------|----------------------|-------------------------------|----------------|---------------------|
| classical_single  | 9.7                  | 21.3              | 4.8                  | 68                            | 83.8%          | 9.81                |
| classical_multi   | 0.6                  | 1.1               | 5.1                  | 68                            | 61.7%          | 9.78                |
| classical_perfect | 0.0                  | 0.0               | 4.4                  | 68                            | 96.5%          | 9.81                |

Why the classical controllers missed (count of misses):

| controller        | arm tracking | not reachable in time | prediction |
|-------------------|--------------|-----------------------|------------|
| classical_single  | 74           | 514                   | 1553       |
| classical_multi   | 44           | 150                   | 0          |
| classical_perfect | 153          | 36                    | 0          |

Catch rate by velocity noise std (m/s):

| velocity noise std (m/s) | random | classical_single | classical_multi | classical_perfect | oracle | n    |
|--------------------------|--------|------------------|-----------------|-------------------|--------|------|
| 0.05-0.09                | 6%     | 77%              | 97%             | 96%               | 90%    | 1621 |
| 0.09-0.14                | 6%     | 55%              | 96%             | 96%               | 91%    | 1686 |
| 0.14-0.18                | 6%     | 40%              | 96%             | 96%               | 91%    | 1693 |

Catch rate by speed lost to drag over 1 s (%):

| speed lost to drag over 1 s (%) | random | classical_single | classical_multi | classical_perfect | oracle | n    |
|---------------------------------|--------|------------------|-----------------|-------------------|--------|------|
| 0-4                             | 5%     | 59%              | 96%             | 95%               | 94%    | 2239 |
| 4-8                             | 6%     | 57%              | 96%             | 97%               | 91%    | 2079 |
| 8-13                            | 6%     | 53%              | 96%             | 97%               | 79%    | 682  |

Catch rate by action delay (steps of 40 ms):

| action delay (steps of 40 ms) | random | classical_single | classical_multi | classical_perfect | oracle | n    |
|-------------------------------|--------|------------------|-----------------|-------------------|--------|------|
| 0                             | 6%     | 65%              | 100%            | 100%              | 95%    | 1634 |
| 1                             | 6%     | 57%              | 98%             | 98%               | 92%    | 1693 |
| 2                             | 6%     | 50%              | 91%             | 91%               | 85%    | 1673 |

Catch rate by flight time (s):

| flight time (s) | random | classical_single | classical_multi | classical_perfect | oracle | n    |
|-----------------|--------|------------------|-----------------|-------------------|--------|------|
| (0.599, 0.7]    | 6%     | 64%              | 92%             | 91%               | 91%    | 1217 |
| (0.7, 0.8]      | 5%     | 61%              | 96%             | 96%               | 91%    | 1256 |
| (0.8, 0.9]      | 6%     | 55%              | 98%             | 98%               | 91%    | 1278 |
| (0.9, 1.0]      | 7%     | 49%              | 98%             | 99%               | 90%    | 1249 |

Catch rate by launch azimuth (deg):

| launch azimuth (deg) | random | classical_single | classical_multi | classical_perfect | oracle | n    |
|----------------------|--------|------------------|-----------------|-------------------|--------|------|
| (-60.001, -30.0]     | 6%     | 58%              | 97%             | 98%               | 92%    | 1256 |
| (-30.0, 0.0]         | 7%     | 58%              | 96%             | 95%               | 94%    | 1218 |
| (0.0, 30.0]          | 7%     | 55%              | 95%             | 95%               | 91%    | 1260 |
| (30.0, 60.0]         | 4%     | 58%              | 96%             | 96%               | 86%    | 1266 |

Catch rate by home-to-target distance (m):

| home-to-target distance (m) | random | classical_single | classical_multi | classical_perfect | oracle | n    |
|-----------------------------|--------|------------------|-----------------|-------------------|--------|------|
| (-0.001, 0.2]               | 12%    | 60%              | 98%             | 98%               | 94%    | 658  |
| (0.2, 0.3]                  | 8%     | 58%              | 98%             | 98%               | 89%    | 1153 |
| (0.3, 0.4]                  | 4%     | 57%              | 96%             | 97%               | 91%    | 1429 |
| (0.4, 0.5]                  | 3%     | 57%              | 95%             | 95%               | 91%    | 1246 |
| (0.5, 0.8]                  | 4%     | 54%              | 93%             | 92%               | 89%    | 514  |
