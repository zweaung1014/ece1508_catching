# HoopCatch evaluation summary

## Condition: nominal

| controller        | n    | catch | 95% CI      | ball_floor | ball_robot | collision | timeout | rim hits/ep | IK fails/ep | hoop travel (m) | action change^2 |
|-------------------|------|-------|-------------|------------|------------|-----------|---------|-------------|-------------|-----------------|-----------------|
| random            | 1000 | 4.6%  | 3.5%-6.1%   | 50.1%      | 37.3%      | 8.0%      | 0.0%    | 0.22        | 1.77        | 0.88            | 5.089           |
| classical_single  | 1000 | 95.7% | 94.3%-96.8% | 3.7%       | 0.6%       | 0.0%      | 0.0%    | 0.43        | 0.28        | 0.93            | 0.615           |
| classical_multi   | 1000 | 99.7% | 99.1%-99.9% | 0.3%       | 0.0%       | 0.0%      | 0.0%    | 0.00        | 0.42        | 0.96            | 0.739           |
| classical_perfect | 1000 | 99.8% | 99.3%-99.9% | 0.2%       | 0.0%       | 0.0%      | 0.0%    | 0.00        | 0.26        | 0.92            | 0.619           |
| oracle            | 1000 | 94.9% | 93.4%-96.1% | 4.9%       | 0.2%       | 0.0%      | 0.0%    | 0.09        | 4.29        | 0.65            | 0.352           |

Paired McNemar vs classical_multi (same seeds):

| controller        | paired n | only classical_multi caught | only other caught | p        |
|-------------------|----------|-----------------------------|-------------------|----------|
| random            | 1000     | 951                         | 0                 | 1.1e-286 |
| classical_single  | 1000     | 42                          | 2                 | 1.1e-10  |
| classical_perfect | 1000     | 0                           | 1                 | 1        |
| oracle            | 1000     | 49                          | 1                 | 9.1e-14  |

Classical diagnostics (prediction error = predicted catch point vs true path, across the path):

| controller        | pred err median (cm) | pred err p90 (cm) | hoop err median (cm) | first plan after release (ms) | plan reachable | g estimate (median) |
|-------------------|----------------------|-------------------|----------------------|-------------------------------|----------------|---------------------|
| classical_single  | 5.4                  | 10.4              | 2.0                  | 68                            | 92.5%          | 9.81                |
| classical_multi   | 0.3                  | 0.5               | 2.5                  | 68                            | 91.8%          | 9.81                |
| classical_perfect | 0.0                  | 0.0               | 1.8                  | 68                            | 97.6%          | 9.81                |

Why the classical controllers missed (count of misses):

| controller        | arm tracking | not reachable in time | prediction |
|-------------------|--------------|-----------------------|------------|
| classical_single  | 1            | 10                    | 32         |
| classical_multi   | 0            | 3                     | 0          |
| classical_perfect | 0            | 2                     | 0          |

Catch rate by flight time (s):

| flight time (s) | random | classical_single | classical_multi | classical_perfect | oracle | n   |
|-----------------|--------|------------------|-----------------|-------------------|--------|-----|
| (0.599, 0.7]    | 7%     | 97%              | 99%             | 99%               | 91%    | 218 |
| (0.7, 0.8]      | 3%     | 98%              | 100%            | 100%              | 96%    | 276 |
| (0.8, 0.9]      | 4%     | 96%              | 100%            | 100%              | 96%    | 239 |
| (0.9, 1.0]      | 4%     | 92%              | 100%            | 100%              | 96%    | 267 |

Catch rate by launch azimuth (deg):

| launch azimuth (deg) | random | classical_single | classical_multi | classical_perfect | oracle | n   |
|----------------------|--------|------------------|-----------------|-------------------|--------|-----|
| (-60.001, -30.0]     | 5%     | 95%              | 100%            | 100%              | 97%    | 259 |
| (-30.0, 0.0]         | 7%     | 96%              | 100%            | 100%              | 96%    | 215 |
| (0.0, 30.0]          | 4%     | 95%              | 100%            | 100%              | 96%    | 269 |
| (30.0, 60.0]         | 3%     | 96%              | 99%             | 100%              | 91%    | 257 |

Catch rate by home-to-target distance (m):

| home-to-target distance (m) | random | classical_single | classical_multi | classical_perfect | oracle | n   |
|-----------------------------|--------|------------------|-----------------|-------------------|--------|-----|
| (-0.001, 0.2]               | 9%     | 96%              | 100%            | 100%              | 99%    | 151 |
| (0.2, 0.3]                  | 5%     | 96%              | 100%            | 100%              | 90%    | 224 |
| (0.3, 0.4]                  | 5%     | 96%              | 100%            | 100%              | 95%    | 262 |
| (0.4, 0.5]                  | 2%     | 95%              | 100%            | 100%              | 97%    | 260 |
| (0.5, 0.8]                  | 2%     | 95%              | 98%             | 98%               | 94%    | 103 |

## Condition: delay_1

| controller        | n    | catch | 95% CI      | ball_floor | ball_robot | collision | timeout | rim hits/ep | IK fails/ep | hoop travel (m) | action change^2 |
|-------------------|------|-------|-------------|------------|------------|-----------|---------|-------------|-------------|-----------------|-----------------|
| classical_single  | 1000 | 82.6% | 80.1%-84.8% | 14.5%      | 2.9%       | 0.0%      | 0.0%    | 0.76        | 0.64        | 1.09            | 0.893           |
| classical_multi   | 1000 | 97.8% | 96.7%-98.5% | 2.0%       | 0.2%       | 0.0%      | 0.0%    | 0.15        | 0.84        | 1.05            | 0.980           |
| classical_perfect | 1000 | 98.0% | 96.9%-98.7% | 1.8%       | 0.2%       | 0.0%      | 0.0%    | 0.14        | 0.62        | 1.03            | 0.901           |
| oracle            | 1000 | 94.7% | 93.1%-95.9% | 5.1%       | 0.2%       | 0.0%      | 0.0%    | 0.16        | 4.45        | 0.91            | 0.672           |

Paired McNemar vs classical_multi (same seeds):

| controller        | paired n | only classical_multi caught | only other caught | p       |
|-------------------|----------|-----------------------------|-------------------|---------|
| classical_single  | 1000     | 158                         | 6                 | 2.2e-39 |
| classical_perfect | 1000     | 5                           | 7                 | 0.77    |
| oracle            | 1000     | 51                          | 20                | 0.0003  |

Classical diagnostics (prediction error = predicted catch point vs true path, across the path):

| controller        | pred err median (cm) | pred err p90 (cm) | hoop err median (cm) | first plan after release (ms) | plan reachable | g estimate (median) |
|-------------------|----------------------|-------------------|----------------------|-------------------------------|----------------|---------------------|
| classical_single  | 5.4                  | 10.4              | 5.5                  | 68                            | 92.8%          | 9.81                |
| classical_multi   | 0.3                  | 0.5               | 5.5                  | 68                            | 77.4%          | 9.79                |
| classical_perfect | 0.0                  | 0.0               | 5.2                  | 68                            | 97.5%          | 9.81                |

Why the classical controllers missed (count of misses):

| controller        | arm tracking | not reachable in time | prediction |
|-------------------|--------------|-----------------------|------------|
| classical_single  | 22           | 30                    | 122        |
| classical_multi   | 2            | 20                    | 0          |
| classical_perfect | 11           | 9                     | 0          |

Catch rate by flight time (s):

| flight time (s) | classical_single | classical_multi | classical_perfect | oracle | n   |
|-----------------|------------------|-----------------|-------------------|--------|-----|
| (0.599, 0.7]    | 83%              | 92%             | 92%               | 91%    | 218 |
| (0.7, 0.8]      | 83%              | 98%             | 100%              | 96%    | 276 |
| (0.8, 0.9]      | 84%              | 100%            | 99%               | 95%    | 239 |
| (0.9, 1.0]      | 81%              | 100%            | 100%              | 96%    | 267 |

Catch rate by launch azimuth (deg):

| launch azimuth (deg) | classical_single | classical_multi | classical_perfect | oracle | n   |
|----------------------|------------------|-----------------|-------------------|--------|-----|
| (-60.001, -30.0]     | 81%              | 98%             | 98%               | 97%    | 259 |
| (-30.0, 0.0]         | 86%              | 97%             | 97%               | 96%    | 215 |
| (0.0, 30.0]          | 79%              | 98%             | 98%               | 96%    | 269 |
| (30.0, 60.0]         | 85%              | 98%             | 98%               | 90%    | 257 |

Catch rate by home-to-target distance (m):

| home-to-target distance (m) | classical_single | classical_multi | classical_perfect | oracle | n   |
|-----------------------------|------------------|-----------------|-------------------|--------|-----|
| (-0.001, 0.2]               | 89%              | 99%             | 99%               | 99%    | 151 |
| (0.2, 0.3]                  | 85%              | 99%             | 99%               | 92%    | 224 |
| (0.3, 0.4]                  | 82%              | 99%             | 98%               | 94%    | 262 |
| (0.4, 0.5]                  | 79%              | 97%             | 98%               | 97%    | 260 |
| (0.5, 0.8]                  | 80%              | 91%             | 92%               | 91%    | 103 |

## Condition: delay_2

| controller        | n    | catch | 95% CI      | ball_floor | ball_robot | collision | timeout | rim hits/ep | IK fails/ep | hoop travel (m) | action change^2 |
|-------------------|------|-------|-------------|------------|------------|-----------|---------|-------------|-------------|-----------------|-----------------|
| classical_single  | 1000 | 70.2% | 67.3%-73.0% | 23.8%      | 5.7%       | 0.0%      | 0.3%    | 0.90        | 1.05        | 1.07            | 0.784           |
| classical_multi   | 1000 | 89.3% | 87.2%-91.1% | 9.3%       | 1.1%       | 0.0%      | 0.3%    | 0.46        | 1.42        | 1.02            | 0.867           |
| classical_perfect | 1000 | 88.8% | 86.7%-90.6% | 10.1%      | 1.1%       | 0.0%      | 0.0%    | 0.43        | 0.99        | 1.01            | 0.782           |
| oracle            | 1000 | 88.3% | 86.2%-90.1% | 9.8%       | 1.9%       | 0.0%      | 0.0%    | 0.44        | 5.41        | 0.91            | 0.636           |

Paired McNemar vs classical_multi (same seeds):

| controller        | paired n | only classical_multi caught | only other caught | p       |
|-------------------|----------|-----------------------------|-------------------|---------|
| classical_single  | 1000     | 223                         | 32                | 2.1e-36 |
| classical_perfect | 1000     | 49                          | 44                | 0.68    |
| oracle            | 1000     | 93                          | 83                | 0.5     |

Classical diagnostics (prediction error = predicted catch point vs true path, across the path):

| controller        | pred err median (cm) | pred err p90 (cm) | hoop err median (cm) | first plan after release (ms) | plan reachable | g estimate (median) |
|-------------------|----------------------|-------------------|----------------------|-------------------------------|----------------|---------------------|
| classical_single  | 5.4                  | 10.5              | 6.6                  | 68                            | 92.3%          | 9.81                |
| classical_multi   | 0.3                  | 0.5               | 7.6                  | 68                            | 37.9%          | 9.74                |
| classical_perfect | 0.0                  | 0.0               | 6.6                  | 68                            | 97.1%          | 9.81                |

Why the classical controllers missed (count of misses):

| controller        | arm tracking | not reachable in time | prediction |
|-------------------|--------------|-----------------------|------------|
| classical_single  | 77           | 50                    | 171        |
| classical_multi   | 25           | 82                    | 0          |
| classical_perfect | 91           | 21                    | 0          |

Catch rate by flight time (s):

| flight time (s) | classical_single | classical_multi | classical_perfect | oracle | n   |
|-----------------|------------------|-----------------|-------------------|--------|-----|
| (0.599, 0.7]    | 64%              | 78%             | 76%               | 83%    | 218 |
| (0.7, 0.8]      | 75%              | 91%             | 91%               | 91%    | 276 |
| (0.8, 0.9]      | 73%              | 91%             | 94%               | 90%    | 239 |
| (0.9, 1.0]      | 68%              | 94%             | 93%               | 89%    | 267 |

Catch rate by launch azimuth (deg):

| launch azimuth (deg) | classical_single | classical_multi | classical_perfect | oracle | n   |
|----------------------|------------------|-----------------|-------------------|--------|-----|
| (-60.001, -30.0]     | 71%              | 92%             | 92%               | 90%    | 259 |
| (-30.0, 0.0]         | 70%              | 90%             | 88%               | 89%    | 215 |
| (0.0, 30.0]          | 67%              | 84%             | 85%               | 90%    | 269 |
| (30.0, 60.0]         | 73%              | 93%             | 90%               | 84%    | 257 |

Catch rate by home-to-target distance (m):

| home-to-target distance (m) | classical_single | classical_multi | classical_perfect | oracle | n   |
|-----------------------------|------------------|-----------------|-------------------|--------|-----|
| (-0.001, 0.2]               | 72%              | 91%             | 91%               | 97%    | 151 |
| (0.2, 0.3]                  | 73%              | 94%             | 96%               | 89%    | 224 |
| (0.3, 0.4]                  | 74%              | 93%             | 93%               | 87%    | 262 |
| (0.4, 0.5]                  | 68%              | 88%             | 83%               | 88%    | 260 |
| (0.5, 0.8]                  | 59%              | 71%             | 75%               | 78%    | 103 |

## Condition: gravity_fixed

| controller        | n    | catch | 95% CI       | ball_floor | ball_robot | collision | timeout | rim hits/ep | IK fails/ep | hoop travel (m) | action change^2 |
|-------------------|------|-------|--------------|------------|------------|-----------|---------|-------------|-------------|-----------------|-----------------|
| classical_single  | 1000 | 99.2% | 98.4%-99.6%  | 0.8%       | 0.0%       | 0.0%      | 0.0%    | 0.19        | 0.28        | 0.92            | 0.620           |
| classical_multi   | 1000 | 99.8% | 99.3%-99.9%  | 0.2%       | 0.0%       | 0.0%      | 0.0%    | 0.00        | 0.29        | 0.96            | 0.742           |
| classical_perfect | 1000 | 99.9% | 99.4%-100.0% | 0.1%       | 0.0%       | 0.0%      | 0.0%    | 0.00        | 0.26        | 0.92            | 0.620           |
| oracle            | 1000 | 95.0% | 93.5%-96.2%  | 4.8%       | 0.2%       | 0.0%      | 0.0%    | 0.08        | 4.20        | 0.65            | 0.352           |

Paired McNemar vs classical_multi (same seeds):

| controller        | paired n | only classical_multi caught | only other caught | p       |
|-------------------|----------|-----------------------------|-------------------|---------|
| classical_single  | 1000     | 7                           | 1                 | 0.07    |
| classical_perfect | 1000     | 0                           | 1                 | 1       |
| oracle            | 1000     | 49                          | 1                 | 9.1e-14 |

Classical diagnostics (prediction error = predicted catch point vs true path, across the path):

| controller        | pred err median (cm) | pred err p90 (cm) | hoop err median (cm) | first plan after release (ms) | plan reachable | g estimate (median) |
|-------------------|----------------------|-------------------|----------------------|-------------------------------|----------------|---------------------|
| classical_single  | 4.1                  | 8.3               | 1.8                  | 68                            | 94.4%          | 9.81                |
| classical_multi   | 0.3                  | 0.5               | 2.4                  | 68                            | 92.8%          | 9.81                |
| classical_perfect | 0.0                  | 0.0               | 1.8                  | 68                            | 97.7%          | 9.81                |

Why the classical controllers missed (count of misses):

| controller        | not reachable in time | prediction |
|-------------------|-----------------------|------------|
| classical_single  | 4                     | 4          |
| classical_multi   | 2                     | 0          |
| classical_perfect | 1                     | 0          |

Catch rate by flight time (s):

| flight time (s) | classical_single | classical_multi | classical_perfect | oracle | n   |
|-----------------|------------------|-----------------|-------------------|--------|-----|
| (0.599, 0.7]    | 99%              | 99%             | 100%              | 92%    | 218 |
| (0.7, 0.8]      | 100%             | 100%            | 100%              | 96%    | 276 |
| (0.8, 0.9]      | 99%              | 100%            | 100%              | 96%    | 239 |
| (0.9, 1.0]      | 99%              | 100%            | 100%              | 96%    | 267 |

Catch rate by launch azimuth (deg):

| launch azimuth (deg) | classical_single | classical_multi | classical_perfect | oracle | n   |
|----------------------|------------------|-----------------|-------------------|--------|-----|
| (-60.001, -30.0]     | 99%              | 100%            | 100%              | 97%    | 259 |
| (-30.0, 0.0]         | 99%              | 100%            | 100%              | 97%    | 215 |
| (0.0, 30.0]          | 100%             | 100%            | 100%              | 96%    | 269 |
| (30.0, 60.0]         | 99%              | 100%            | 100%              | 91%    | 257 |

Catch rate by home-to-target distance (m):

| home-to-target distance (m) | classical_single | classical_multi | classical_perfect | oracle | n   |
|-----------------------------|------------------|-----------------|-------------------|--------|-----|
| (-0.001, 0.2]               | 100%             | 100%            | 100%              | 99%    | 151 |
| (0.2, 0.3]                  | 100%             | 100%            | 100%              | 90%    | 224 |
| (0.3, 0.4]                  | 98%              | 100%            | 100%              | 95%    | 262 |
| (0.4, 0.5]                  | 99%              | 100%            | 100%              | 97%    | 260 |
| (0.5, 0.8]                  | 98%              | 99%             | 99%               | 94%    | 103 |

## Condition: noise_x2

| controller        | n    | catch | 95% CI      | ball_floor | ball_robot | collision | timeout | rim hits/ep | IK fails/ep | hoop travel (m) | action change^2 |
|-------------------|------|-------|-------------|------------|------------|-----------|---------|-------------|-------------|-----------------|-----------------|
| classical_single  | 1000 | 71.5% | 68.6%-74.2% | 19.6%      | 8.9%       | 0.0%      | 0.0%    | 0.88        | 0.54        | 0.96            | 0.596           |
| classical_multi   | 1000 | 99.7% | 99.1%-99.9% | 0.3%       | 0.0%       | 0.0%      | 0.0%    | 0.01        | 0.48        | 1.00            | 0.899           |
| classical_perfect | 1000 | 99.8% | 99.3%-99.9% | 0.2%       | 0.0%       | 0.0%      | 0.0%    | 0.00        | 0.26        | 0.92            | 0.619           |
| oracle            | 1000 | 94.9% | 93.4%-96.1% | 4.9%       | 0.2%       | 0.0%      | 0.0%    | 0.09        | 4.29        | 0.65            | 0.352           |

Paired McNemar vs classical_multi (same seeds):

| controller        | paired n | only classical_multi caught | only other caught | p       |
|-------------------|----------|-----------------------------|-------------------|---------|
| classical_single  | 1000     | 283                         | 1                 | 1.8e-83 |
| classical_perfect | 1000     | 0                           | 1                 | 1       |
| oracle            | 1000     | 49                          | 1                 | 9.1e-14 |

Classical diagnostics (prediction error = predicted catch point vs true path, across the path):

| controller        | pred err median (cm) | pred err p90 (cm) | hoop err median (cm) | first plan after release (ms) | plan reachable | g estimate (median) |
|-------------------|----------------------|-------------------|----------------------|-------------------------------|----------------|---------------------|
| classical_single  | 8.8                  | 17.5              | 2.1                  | 68                            | 86.9%          | 9.81                |
| classical_multi   | 0.5                  | 1.0               | 3.0                  | 68                            | 88.5%          | 9.82                |
| classical_perfect | 0.0                  | 0.0               | 1.8                  | 68                            | 97.6%          | 9.81                |

Why the classical controllers missed (count of misses):

| controller        | not reachable in time | prediction |
|-------------------|-----------------------|------------|
| classical_single  | 77                    | 208        |
| classical_multi   | 3                     | 0          |
| classical_perfect | 2                     | 0          |

Catch rate by flight time (s):

| flight time (s) | classical_single | classical_multi | classical_perfect | oracle | n   |
|-----------------|------------------|-----------------|-------------------|--------|-----|
| (0.599, 0.7]    | 82%              | 99%             | 99%               | 91%    | 218 |
| (0.7, 0.8]      | 78%              | 100%            | 100%              | 96%    | 276 |
| (0.8, 0.9]      | 67%              | 100%            | 100%              | 96%    | 239 |
| (0.9, 1.0]      | 60%              | 100%            | 100%              | 96%    | 267 |

Catch rate by launch azimuth (deg):

| launch azimuth (deg) | classical_single | classical_multi | classical_perfect | oracle | n   |
|----------------------|------------------|-----------------|-------------------|--------|-----|
| (-60.001, -30.0]     | 69%              | 100%            | 100%              | 97%    | 259 |
| (-30.0, 0.0]         | 77%              | 100%            | 100%              | 96%    | 215 |
| (0.0, 30.0]          | 70%              | 100%            | 100%              | 96%    | 269 |
| (30.0, 60.0]         | 71%              | 99%             | 100%              | 91%    | 257 |

Catch rate by home-to-target distance (m):

| home-to-target distance (m) | classical_single | classical_multi | classical_perfect | oracle | n   |
|-----------------------------|------------------|-----------------|-------------------|--------|-----|
| (-0.001, 0.2]               | 77%              | 100%            | 100%              | 99%    | 151 |
| (0.2, 0.3]                  | 73%              | 100%            | 100%              | 90%    | 224 |
| (0.3, 0.4]                  | 68%              | 100%            | 100%              | 95%    | 262 |
| (0.4, 0.5]                  | 70%              | 100%            | 100%              | 97%    | 260 |
| (0.5, 0.8]                  | 72%              | 98%             | 98%               | 94%    | 103 |

## Condition: noise_x4

| controller        | n    | catch | 95% CI      | ball_floor | ball_robot | collision | timeout | rim hits/ep | IK fails/ep | hoop travel (m) | action change^2 |
|-------------------|------|-------|-------------|------------|------------|-----------|---------|-------------|-------------|-----------------|-----------------|
| classical_single  | 1000 | 33.7% | 30.8%-36.7% | 37.0%      | 28.9%      | 0.4%      | 0.0%    | 0.79        | 1.49        | 0.99            | 0.576           |
| classical_multi   | 1000 | 99.6% | 99.0%-99.8% | 0.4%       | 0.0%       | 0.0%      | 0.0%    | 0.04        | 0.60        | 1.04            | 1.141           |
| classical_perfect | 1000 | 99.8% | 99.3%-99.9% | 0.2%       | 0.0%       | 0.0%      | 0.0%    | 0.00        | 0.26        | 0.92            | 0.619           |
| oracle            | 1000 | 94.9% | 93.4%-96.1% | 4.9%       | 0.2%       | 0.0%      | 0.0%    | 0.09        | 4.29        | 0.65            | 0.352           |

Paired McNemar vs classical_multi (same seeds):

| controller        | paired n | only classical_multi caught | only other caught | p        |
|-------------------|----------|-----------------------------|-------------------|----------|
| classical_single  | 1000     | 660                         | 1                 | 1.4e-196 |
| classical_perfect | 1000     | 1                           | 3                 | 0.62     |
| oracle            | 1000     | 50                          | 3                 | 5.5e-12  |

Classical diagnostics (prediction error = predicted catch point vs true path, across the path):

| controller        | pred err median (cm) | pred err p90 (cm) | hoop err median (cm) | first plan after release (ms) | plan reachable | g estimate (median) |
|-------------------|----------------------|-------------------|----------------------|-------------------------------|----------------|---------------------|
| classical_single  | 16.7                 | 32.5              | 2.6                  | 68                            | 69.7%          | 9.81                |
| classical_multi   | 1.0                  | 2.0               | 3.7                  | 68                            | 74.7%          | 9.81                |
| classical_perfect | 0.0                  | 0.0               | 1.8                  | 68                            | 97.5%          | 9.81                |

Why the classical controllers missed (count of misses):

| controller        | not reachable in time | prediction |
|-------------------|-----------------------|------------|
| classical_single  | 261                   | 402        |
| classical_multi   | 4                     | 0          |
| classical_perfect | 2                     | 0          |

Catch rate by flight time (s):

| flight time (s) | classical_single | classical_multi | classical_perfect | oracle | n   |
|-----------------|------------------|-----------------|-------------------|--------|-----|
| (0.599, 0.7]    | 48%              | 99%             | 99%               | 91%    | 218 |
| (0.7, 0.8]      | 35%              | 100%            | 100%              | 96%    | 276 |
| (0.8, 0.9]      | 29%              | 100%            | 100%              | 96%    | 239 |
| (0.9, 1.0]      | 25%              | 100%            | 100%              | 96%    | 267 |

Catch rate by launch azimuth (deg):

| launch azimuth (deg) | classical_single | classical_multi | classical_perfect | oracle | n   |
|----------------------|------------------|-----------------|-------------------|--------|-----|
| (-60.001, -30.0]     | 31%              | 100%            | 100%              | 97%    | 259 |
| (-30.0, 0.0]         | 34%              | 100%            | 100%              | 96%    | 215 |
| (0.0, 30.0]          | 35%              | 100%            | 100%              | 96%    | 269 |
| (30.0, 60.0]         | 34%              | 100%            | 100%              | 91%    | 257 |

Catch rate by home-to-target distance (m):

| home-to-target distance (m) | classical_single | classical_multi | classical_perfect | oracle | n   |
|-----------------------------|------------------|-----------------|-------------------|--------|-----|
| (-0.001, 0.2]               | 38%              | 100%            | 100%              | 99%    | 151 |
| (0.2, 0.3]                  | 33%              | 100%            | 100%              | 90%    | 224 |
| (0.3, 0.4]                  | 34%              | 100%            | 100%              | 95%    | 262 |
| (0.4, 0.5]                  | 30%              | 99%             | 100%              | 97%    | 260 |
| (0.5, 0.8]                  | 36%              | 99%             | 98%               | 94%    | 103 |

## Condition: tracker_noise

| controller        | n    | catch | 95% CI      | ball_floor | ball_robot | collision | timeout | rim hits/ep | IK fails/ep | hoop travel (m) | action change^2 |
|-------------------|------|-------|-------------|------------|------------|-----------|---------|-------------|-------------|-----------------|-----------------|
| classical_single  | 1000 | 39.0% | 36.0%-42.1% | 36.1%      | 24.4%      | 0.5%      | 0.0%    | 0.80        | 1.23        | 0.99            | 0.579           |
| classical_multi   | 1000 | 99.4% | 98.7%-99.7% | 0.6%       | 0.0%       | 0.0%      | 0.0%    | 0.03        | 0.57        | 1.02            | 1.021           |
| classical_perfect | 1000 | 99.8% | 99.3%-99.9% | 0.2%       | 0.0%       | 0.0%      | 0.0%    | 0.00        | 0.26        | 0.92            | 0.620           |
| oracle            | 1000 | 94.9% | 93.4%-96.1% | 4.9%       | 0.2%       | 0.0%      | 0.0%    | 0.09        | 4.29        | 0.65            | 0.352           |

Paired McNemar vs classical_multi (same seeds):

| controller        | paired n | only classical_multi caught | only other caught | p        |
|-------------------|----------|-----------------------------|-------------------|----------|
| classical_single  | 1000     | 607                         | 3                 | 1.8e-176 |
| classical_perfect | 1000     | 0                           | 4                 | 0.12     |
| oracle            | 1000     | 48                          | 3                 | 2e-11    |

Classical diagnostics (prediction error = predicted catch point vs true path, across the path):

| controller        | pred err median (cm) | pred err p90 (cm) | hoop err median (cm) | first plan after release (ms) | plan reachable | g estimate (median) |
|-------------------|----------------------|-------------------|----------------------|-------------------------------|----------------|---------------------|
| classical_single  | 15.0                 | 29.2              | 2.5                  | 68                            | 74.3%          | 9.81                |
| classical_multi   | 0.4                  | 0.8               | 3.2                  | 68                            | 81.5%          | 9.81                |
| classical_perfect | 0.0                  | 0.0               | 1.8                  | 68                            | 97.5%          | 9.81                |

Why the classical controllers missed (count of misses):

| controller        | not reachable in time | prediction |
|-------------------|-----------------------|------------|
| classical_single  | 216                   | 394        |
| classical_multi   | 6                     | 0          |
| classical_perfect | 2                     | 0          |

Catch rate by flight time (s):

| flight time (s) | classical_single | classical_multi | classical_perfect | oracle | n   |
|-----------------|------------------|-----------------|-------------------|--------|-----|
| (0.599, 0.7]    | 54%              | 98%             | 99%               | 91%    | 218 |
| (0.7, 0.8]      | 42%              | 100%            | 100%              | 96%    | 276 |
| (0.8, 0.9]      | 33%              | 100%            | 100%              | 96%    | 239 |
| (0.9, 1.0]      | 29%              | 100%            | 100%              | 96%    | 267 |

Catch rate by launch azimuth (deg):

| launch azimuth (deg) | classical_single | classical_multi | classical_perfect | oracle | n   |
|----------------------|------------------|-----------------|-------------------|--------|-----|
| (-60.001, -30.0]     | 37%              | 100%            | 100%              | 97%    | 259 |
| (-30.0, 0.0]         | 38%              | 100%            | 100%              | 96%    | 215 |
| (0.0, 30.0]          | 40%              | 100%            | 100%              | 96%    | 269 |
| (30.0, 60.0]         | 41%              | 98%             | 100%              | 91%    | 257 |

Catch rate by home-to-target distance (m):

| home-to-target distance (m) | classical_single | classical_multi | classical_perfect | oracle | n   |
|-----------------------------|------------------|-----------------|-------------------|--------|-----|
| (-0.001, 0.2]               | 42%              | 100%            | 100%              | 99%    | 151 |
| (0.2, 0.3]                  | 38%              | 100%            | 100%              | 90%    | 224 |
| (0.3, 0.4]                  | 40%              | 100%            | 100%              | 95%    | 262 |
| (0.4, 0.5]                  | 37%              | 99%             | 100%              | 97%    | 260 |
| (0.5, 0.8]                  | 40%              | 98%             | 98%               | 94%    | 103 |
