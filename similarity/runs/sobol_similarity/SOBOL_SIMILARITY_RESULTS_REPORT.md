# Sobol sensitivity analysis: similarity DeepONet

## Scope

This analysis evaluates the trained `sim_deeponet_v1` similarity model. It varies the same six independent physical inputs and uses the same ranges, representative geometry, Sobol estimator, random seed, sample sizes (`N = 256, 512, 1024`), and 500 bootstrap resamples as the project DeepONet analysis. Each physical sample is converted to the model's similarity representation `[l, phi, P/Q]` before inference.

The quantities of interest are the spatial mean, median, 95th percentile, and maximum of absolute deflection `|w|`. Mean and average are the same statistic and are not duplicated.

## Main result

Winkler foundation stiffness remains the dominant uncertainty driver in the similarity representation. Pressure has a substantially larger role in near-peak and maximum response than it did in the project model. Thickness primarily affects mean and median response.

| QoI | Parameter | S1 | ST | ST - S1 |
|---|---|---:|---:|---:|
| Mean `|w|` | `k_winkler` | 0.798 | 0.904 | 0.106 |
|  | thickness | 0.061 | 0.110 | 0.050 |
|  | pressure | 0.039 | 0.072 | 0.033 |
| Median `|w|` | `k_winkler` | 0.704 | 0.876 | 0.172 |
|  | thickness | 0.120 | 0.236 | 0.116 |
|  | pressure | 0.014 | 0.045 | 0.031 |
| 95th percentile `|w|` | `k_winkler` | 0.817 | 0.895 | 0.078 |
|  | pressure | 0.096 | 0.144 | 0.049 |
|  | thickness | 0.006 | 0.016 | 0.010 |
| Maximum `|w|` | `k_winkler` | 0.780 | 0.850 | 0.071 |
|  | pressure | 0.121 | 0.171 | 0.050 |
|  | thickness | 0.021 | 0.027 | 0.006 |

Young's modulus has a small total-order effect of approximately 0.025-0.030. Temperature change and thermal expansion coefficient are near zero for all four deflection summaries within the studied ranges.

## Comparison with the project DeepONet

The two representations agree on the primary physical conclusion: `k_winkler` is the dominant factor. They also agree that thickness matters more for typical/global deflection and pressure matters more for peak deflection.

They differ in magnitude. The similarity model assigns more peak-response variance to pressure (`ST = 0.144` for p95 and `0.171` for maximum) than the project model (`0.040` for both). It assigns less p95 interaction involvement to thickness (`ST = 0.016`) than the project model (`0.139`). These discrepancies should be treated as model-form uncertainty, not averaged into one index.

## Surrogate reliability

Validation used the exact 3,000-case held-out test split:

| QoI | R2 | Relative L2 error |
|---|---:|---:|
| Mean `|w|` | 0.897 | 0.199 |
| Median `|w|` | 0.909 | 0.202 |
| 95th percentile `|w|` | 0.702 | 0.332 |
| Maximum `|w|` | 0.647 | 0.368 |

Mean and median rankings are the most defensible. P95 and maximum results are lower-confidence because the model is less accurate for localized extremes. Bootstrap and convergence plots retain estimator uncertainty, but that uncertainty does not include surrogate-model error.

## Important as-trained unit caveat

The existing `similarity_transform.py` expects unit weight values such as `23600 N/m3` and applies a `1e-9` conversion, while generated `.in2` files contain values such as `0.0000236`, which already appear to be in `N/mm3`. The Winkler conversion documentation and generated magnitudes also appear inconsistent. This analysis intentionally preserves the transform used to train the checkpoint, so the results characterize the **as-trained similarity model**.

The unit implementation should be verified against the ISLAB input specification before these similarity indices are treated as final physical evidence. Correcting the transform would require rebuilding cached graphs, retraining the similarity model, revalidating it, and rerunning Sobol analysis.

## Recommended figures

1. `project_similarity_comparison.png` for the aligned comparison.
2. `sobol_bootstrap_summary.png` for similarity-model uncertainty distributions.
3. `sobol_convergence.png` for sample-size stability.
4. The four `*_variance_map.png` figures for detailed per-QoI interpretation.
