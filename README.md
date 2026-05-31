## Learned PSO Hyperparameters

The table below summarizes the final hyperparameter values learned by the DRL agent for each RL-enhanced PSO variant.

| Parameter | RL-APSO |        RL-ARPSO | RL-SPSO |
| --------- | ------: | --------------: | ------: |
| w₁        |  0.7801 |               — |       — |
| w₂        | -0.1954 |               — |       — |
| c₁        |  1.3788 |          0.0676 |    5.00 |
| c₂        |  1.3788 |          3.5000 |    0.05 |
| wᵢ        |       — | 0.0617 ± 0.0305 |       — |

**Table:** Final hyperparameter values for the RL-enhanced APSO, ARPSO, and SPSO algorithms.

## Reward Function Hyperparameters

The following hyperparameters are used in the reward function design and reward normalization process.

| Category                       | Constant                         | Value |
| ------------------------------ | -------------------------------- | ----: |
| **Time / Iteration Penalties** | α (Time penalty)                 |  30.0 |
|                                | β (Iteration penalty)            |  1.25 |
|                                | γ (Shaping weight)               |   1.0 |
| **Event-based Rewards**        | R<sub>f</sub> (Found bonus)      | 300.0 |
|                                | R<sub>t</sub> (Timeout penalty)  | -20.0 |
|                                | η (Invalidity penalty)           |  80.0 |
| **Normalization**              | β<sub>r</sub> (Smoothing factor) | 0.999 |
|                                | C (Clipping threshold)           | 200.0 |
|                                | ε (Numerical stability)          |  1e-8 |

**Table:** Hyperparameters used for reward computation and normalization in the proposed DRL framework.


## ARPSO Results

The results for ARPSO can be found in the following figures:

### Average Number of Iterations vs Swarm Size
![ARPSO Iterations](arpso_vs_rlarpso_swarm_size_means_only_iterations.png)

### Average Swarm Distance vs Swarm Size
![ARPSO Swarm Distance](arpso_vs_rlarpso_swarm_size_means_only_swarm_distance.png)

### Average Source-Seeking Time vs Swarm Size
![ARPSO Source Seeking Time](arpso_vs_rlarpso_swarm_size_means_only_time.png)

## Source Speed Analysis

The table below reports the average convergence time (lower is better) for different PSO variants under varying source speeds.

| Method   |      0.00 |      0.05 |      0.10 |      0.15 |      0.20 |      0.25 |      0.30 |
| -------- | --------: | --------: | --------: | --------: | --------: | --------: | --------: |
| APSO     |     16.05 |     15.03 | **14.16** |     14.81 |     16.24 | **14.05** | **14.14** |
| RL-APSO  | **14.86** | **14.90** |     14.24 | **14.74** | **14.89** |     14.32 |     14.80 |
| ARPSO    |     31.14 |     35.06 |     38.75 |     30.72 |     34.65 |     30.87 |     32.35 |
| RL-ARPSO |     32.00 |     32.29 |     32.24 |     31.42 |     34.42 |     32.20 |     32.30 |
| SPSO     |     43.68 |     35.14 |     39.96 |     39.50 |     37.39 |     43.92 |     32.52 |
| RL-SPSO  |     42.35 |     33.23 |     42.38 |     40.73 |     35.57 |     40.80 |     39.59 |

**Table:** Average convergence time under varying source speeds. Lower values indicate faster convergence.

## Sensor Noise Analysis

The table below reports the average convergence time (lower is better) under different measurement noise levels.

| Method   |        0% |        5% |       10% |
| -------- | --------: | --------: | --------: |
| APSO     | **26.35** |     52.68 |     63.49 |
| RL-APSO  |     27.52 | **49.87** | **59.00** |
| ARPSO    |     45.51 |     75.72 |    106.74 |
| RL-ARPSO |    113.01 |     91.78 |    115.08 |
| SPSO     |     89.10 |    176.14 |     96.20 |
| RL-SPSO  |     76.73 |     77.58 |     90.06 |

**Table:** Average convergence time under different measurement noise levels. Lower values indicate faster convergence.

