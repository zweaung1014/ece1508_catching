# HoopCatch evaluation summary: exp03_harder

## Condition: exp03_harder

| controller        | n    | catch | 95% CI      | ball_floor | ball_robot | collision | timeout | rim hits/ep | IK fails/ep | hoop travel (m) | action change^2 |
|-------------------|------|-------|-------------|------------|------------|-----------|---------|-------------|-------------|-----------------|-----------------|
| random            | 5000 | 5.0%  | 4.5%-5.7%   | 50.0%      | 38.0%      | 6.9%      | 0.0%    | 0.25        | 1.85        | 0.88            | 5.083           |
| classical_single  | 5000 | 21.5% | 20.4%-22.7% | 46.0%      | 32.2%      | 0.2%      | 0.1%    | 0.68        | 3.39        | 1.02            | 0.647           |
| classical_multi   | 5000 | 76.0% | 74.8%-77.2% | 17.6%      | 6.3%       | 0.0%      | 0.1%    | 0.67        | 2.15        | 1.01            | 0.866           |
| classical_perfect | 5000 | 81.4% | 80.3%-82.5% | 13.3%      | 5.1%       | 0.0%      | 0.2%    | 0.52        | 1.89        | 0.97            | 0.689           |
| oracle            | 5000 | 84.4% | 83.3%-85.4% | 11.9%      | 3.6%       | 0.0%      | 0.1%    | 0.53        | 5.78        | 0.84            | 0.527           |

Paired McNemar vs classical_multi (same seeds):

| controller        | paired n | only classical_multi caught | only other caught | p       |
|-------------------|----------|-----------------------------|-------------------|---------|
| random            | 5000     | 3597                        | 49                | 0       |
| classical_single  | 5000     | 2825                        | 102               | 0       |
| classical_perfect | 5000     | 289                         | 560               | 9.1e-21 |
| oracle            | 5000     | 479                         | 898               | 7.1e-30 |

Classical diagnostics (prediction error = predicted catch point vs true path, across the path):

| controller        | pred err median (cm) | pred err p90 (cm) | hoop err median (cm) | first plan after release (ms) | plan reachable | g estimate (median) |
|-------------------|----------------------|-------------------|----------------------|-------------------------------|----------------|---------------------|
| classical_single  | 22.9                 | 49.4              | 11.5                 | 78                            | 49.9%          | 9.81                |
| classical_multi   | 2.2                  | 4.5               | 8.5                  | 78                            | 26.6%          | 9.70                |
| classical_perfect | 0.0                  | 0.0               | 7.0                  | 78                            | 95.9%          | 9.81                |

Why the classical controllers missed (count of misses):

| controller        | arm tracking | not reachable in time | prediction |
|-------------------|--------------|-----------------------|------------|
| classical_single  | 41           | 2127                  | 1755       |
| classical_multi   | 69           | 1125                  | 6          |
| classical_perfect | 783          | 146                   | 0          |

Catch rate by position noise std (mm):

| position noise std (mm) | random | classical_single | classical_multi | classical_perfect | oracle | n    |
|-------------------------|--------|------------------|-----------------|-------------------|--------|------|
| 5-10                    | 5%     | 22%              | 76%             | 80%               | 85%    | 1693 |
| 10-15                   | 5%     | 21%              | 77%             | 82%               | 84%    | 1592 |
| 15-20                   | 6%     | 22%              | 75%             | 83%               | 85%    | 1715 |

Catch rate by velocity noise std (m/s):

| velocity noise std (m/s) | random | classical_single | classical_multi | classical_perfect | oracle | n    |
|--------------------------|--------|------------------|-----------------|-------------------|--------|------|
| 0.1-0.2                  | 6%     | 28%              | 75%             | 80%               | 85%    | 1665 |
| 0.2-0.3                  | 5%     | 19%              | 76%             | 82%               | 84%    | 1720 |
| 0.3-0.4                  | 5%     | 18%              | 77%             | 83%               | 85%    | 1615 |

Catch rate by speed lost to drag over 1 s (%):

| speed lost to drag over 1 s (%) | random | classical_single | classical_multi | classical_perfect | oracle | n    |
|---------------------------------|--------|------------------|-----------------|-------------------|--------|------|
| 0-10                            | 6%     | 22%              | 77%             | 81%               | 83%    | 1909 |
| 10-20                           | 5%     | 23%              | 77%             | 82%               | 85%    | 2085 |
| 20-35                           | 4%     | 19%              | 72%             | 81%               | 87%    | 1006 |

Catch rate by wind force (N):

| wind force (N) | random | classical_single | classical_multi | classical_perfect | oracle | n    |
|----------------|--------|------------------|-----------------|-------------------|--------|------|
| 0-0.01         | 6%     | 23%              | 81%             | 80%               | 83%    | 1250 |
| 0.01-0.02      | 5%     | 23%              | 79%             | 82%               | 85%    | 1265 |
| 0.02-0.03      | 5%     | 21%              | 73%             | 81%               | 84%    | 1230 |
| 0.03-0.04      | 5%     | 19%              | 71%             | 82%               | 85%    | 1255 |

Catch rate by wind across the throw (N):

| wind across the throw (N) | random | classical_single | classical_multi | classical_perfect | oracle | n    |
|---------------------------|--------|------------------|-----------------|-------------------|--------|------|
| 0-0.01                    | 5%     | 23%              | 80%             | 82%               | 84%    | 2446 |
| 0.01-0.02                 | 5%     | 21%              | 75%             | 80%               | 84%    | 1313 |
| 0.02-0.04                 | 4%     | 20%              | 70%             | 83%               | 85%    | 1241 |

Catch rate by action delay (steps of 40 ms):

| action delay (steps of 40 ms) | random | classical_single | classical_multi | classical_perfect | oracle | n    |
|-------------------------------|--------|------------------|-----------------|-------------------|--------|------|
| 0                             | 5%     | 23%              | 99%             | 99%               | 96%    | 944  |
| 1                             | 5%     | 22%              | 91%             | 96%               | 95%    | 1017 |
| 2                             | 5%     | 20%              | 76%             | 87%               | 86%    | 1030 |
| 3                             | 4%     | 22%              | 65%             | 69%               | 73%    | 988  |
| 4                             | 6%     | 21%              | 51%             | 56%               | 72%    | 1021 |

Catch rate by weakest joint's PID gain scale:

| weakest joint's PID gain scale | random | classical_single | classical_multi | classical_perfect | oracle | n    |
|--------------------------------|--------|------------------|-----------------|-------------------|--------|------|
| 0.50-0.60                      | 5%     | 22%              | 73%             | 79%               | 83%    | 2321 |
| 0.60-0.75                      | 5%     | 21%              | 78%             | 83%               | 85%    | 1796 |
| 0.75-1.50                      | 5%     | 21%              | 80%             | 86%               | 86%    | 883  |

Catch rate by flight time (s):

| flight time (s) | random | classical_single | classical_multi | classical_perfect | oracle | n    |
|-----------------|--------|------------------|-----------------|-------------------|--------|------|
| (0.599, 0.7]    | 5%     | 27%              | 71%             | 73%               | 81%    | 1217 |
| (0.7, 0.8]      | 5%     | 26%              | 77%             | 81%               | 83%    | 1256 |
| (0.8, 0.9]      | 5%     | 20%              | 78%             | 84%               | 85%    | 1278 |
| (0.9, 1.0]      | 5%     | 14%              | 78%             | 87%               | 89%    | 1249 |

Catch rate by launch azimuth (deg):

| launch azimuth (deg) | random | classical_single | classical_multi | classical_perfect | oracle | n    |
|----------------------|--------|------------------|-----------------|-------------------|--------|------|
| (-60.001, -30.0]     | 5%     | 22%              | 77%             | 84%               | 85%    | 1256 |
| (-30.0, 0.0]         | 6%     | 22%              | 75%             | 79%               | 85%    | 1218 |
| (0.0, 30.0]          | 5%     | 21%              | 77%             | 80%               | 85%    | 1260 |
| (30.0, 60.0]         | 4%     | 21%              | 76%             | 82%               | 82%    | 1266 |

Catch rate by home-to-target distance (m):

| home-to-target distance (m) | random | classical_single | classical_multi | classical_perfect | oracle | n    |
|-----------------------------|--------|------------------|-----------------|-------------------|--------|------|
| (-0.001, 0.2]               | 12%    | 21%              | 82%             | 88%               | 93%    | 658  |
| (0.2, 0.3]                  | 6%     | 23%              | 82%             | 88%               | 86%    | 1153 |
| (0.3, 0.4]                  | 4%     | 23%              | 78%             | 83%               | 86%    | 1429 |
| (0.4, 0.5]                  | 2%     | 20%              | 69%             | 76%               | 82%    | 1246 |
| (0.5, 0.8]                  | 3%     | 20%              | 67%             | 69%               | 73%    | 514  |
