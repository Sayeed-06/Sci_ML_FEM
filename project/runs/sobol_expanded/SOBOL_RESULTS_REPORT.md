# Sobol sensitivity analysis of spatial deflection statistics

## Executive interpretation

The analysis treats the plate deflection field as four scalar quantities of interest (QoIs): the spatial **mean**, **median**, **95th percentile**, and **maximum** of absolute deflection, `|w|`. Mean and average are mathematically the same statistic, so reporting both would be redundant. The median was added as a robust measure of typical response, while the 95th percentile and maximum describe increasingly local peak behavior.

Across all four QoIs, **Winkler foundation stiffness (`k_winkler`) is the dominant source of output variance**. Plate thickness is the main secondary factor for typical/global response (mean and median), whereas pressure becomes relatively more important for extreme response (95th percentile and maximum). Young's modulus, temperature change, and thermal expansion coefficient have small direct effects over the ranges studied.

## Final Sobol indices (base sample N = 1024)

| QoI | Parameter | First-order S1 | Total-order ST | ST - S1 | Interpretation |
|---|---|---:|---:|---:|---|
| Mean `|w|` | `k_winkler` | 0.726 | 0.810 | 0.084 | Dominant direct effect; some interaction involvement |
|  | thickness | 0.139 | 0.235 | 0.097 | Main secondary driver; interactions are material |
|  | E | 0.002 | 0.028 | 0.026 | Weak direct effect; small interaction involvement |
|  | pressure | 0.009 | 0.021 | 0.012 | Small influence on the spatial average |
| Median `|w|` | `k_winkler` | 0.818 | 0.866 | 0.048 | Stronger dominance than for the mean |
|  | thickness | 0.116 | 0.165 | 0.049 | Secondary driver |
|  | E | 0.004 | 0.016 | 0.012 | Small influence |
|  | pressure | -0.001 | 0.008 | 0.009 | Direct estimate is effectively zero; negative value is sampling noise |
| 95th percentile `|w|` | `k_winkler` | 0.825 | 0.968 | 0.144 | Dominant, with substantial interaction involvement |
|  | thickness | -0.001 | 0.139 | 0.140 | Little resolved direct effect but important through interactions |
|  | pressure | 0.021 | 0.040 | 0.019 | More relevant to high deflection than to the median |
| Maximum `|w|` | `k_winkler` | 0.922 | 0.963 | 0.040 | Almost all peak-response variance involves foundation stiffness |
|  | pressure | 0.039 | 0.040 | 0.001 | Clear direct contribution to the peak response |
|  | thickness | 0.002 | 0.040 | 0.038 | Mainly involved through interactions |

`delta_T` and `alpha_T` each have S1 and ST below approximately 0.02 for the mean/median and below 0.01 for the peak statistics. Their variation therefore contributes little to deflection uncertainty under this model, input range, and fixed geometry. This does not mean they are physically irrelevant to every response variable; they may matter more for stress than deflection.

## What the indices represent

- **S1** is the fraction of output variance explained by changing one input alone, averaged over the other inputs.
- **ST** is the fraction of variance involving that input, including its direct effect and all interactions.
- **ST - S1** is an interaction-involvement indicator. It is not an additional independent variance allocation and does not identify the interacting partner.
- An index near 1 indicates that most of the uncertainty in that QoI is associated with the input. An index near 0 means the input explains little variance over the specified range.
- Small negative S1 estimates are finite-sample estimator noise and should be interpreted as approximately zero, not as negative physical importance.

Sobol indices explain **variance, not the direction of response**. For example, a large index for `k_winkler` says it strongly controls variation in deflection, but the index alone does not say whether increasing stiffness increases or decreases deflection. A partial-dependence or response-curve plot is needed for direction.

## Why mean, median, p95, and maximum are useful together

- **Mean `|w|`**: overall/global deflection severity across the plate.
- **Median `|w|`**: typical nodal response; less affected by a small localized hot spot.
- **95th percentile `|w|`**: near-peak response without relying on a single node.
- **Maximum `|w|`**: worst predicted local response; most sensitive to local prediction error and mesh effects.

The results show that the choice of summary statistic changes the engineering conclusion. Thickness is important for global deflection but its direct contribution nearly disappears for the maximum, while pressure becomes more visible for the maximum. Consequently, a single scalar output would hide meaningful differences between typical and extreme behavior.

## Reliability checks

The expanded DeepONet includes all six varied quantities (`E`, thickness, `k_winkler`, `delta_T`, `alpha_T`, and pressure). On 900 held-out cases, QoI prediction performance was:

| QoI | R2 | Relative L2 error |
|---|---:|---:|
| Mean `|w|` | 0.980 | 0.090 |
| Median `|w|` | 0.972 | 0.115 |
| 95th percentile `|w|` | 0.870 | 0.226 |
| Maximum `|w|` | 0.833 | 0.260 |

The mean and median conclusions are therefore the strongest. The p95 and maximum rankings are still informative, but exact values should be stated with more caution because surrogate error is larger for localized extremes.

The input audit used all 20,000 generated cases. The largest absolute pairwise correlation was only 0.016, supporting the independent-input assumption used by classical Sobol analysis. Bootstrap means and medians are nearly identical for the leading indices, indicating little bootstrap skew. Mean and median estimates are stable across N = 256, 512, and 1024. Peak-statistic estimates change more at low N but are broadly stable from N = 512 to 1024; their wider confidence intervals should remain visible in presentations.

## Scope and caveats

These are **surrogate-based, conditional Sobol indices** for the audited input ranges and the fixed representative geometry `case_14510_EValue34998_Alpha9.34e-06_Kval0.185_pressure0.82`. They should not be generalized to other parameter ranges, boundary conditions, geometries, or response variables without rerunning the analysis. Pairwise second-order indices (S2) were not computed; the S1-ST gap indicates interaction involvement but cannot identify which pair is responsible.

## Recommended advisor-facing figures

1. `sobol_bootstrap_summary.png` — the main result: mean, median, p95, and maximum side by side, with bootstrap distributions.
2. `sobol_convergence.png` — evidence that the rankings and leading estimates have stabilized as sample size increases.
3. The four `*_variance_map.png` files — detailed views of direct effect, total influence, and interaction involvement for each QoI.

