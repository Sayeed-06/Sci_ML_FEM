# Sobol sensitivity analysis: representation and interpretation

## Important distinction in this repository

The existing `scripts/sensitivity_analysis.py` performs a one-factor-at-a-time
(OFAT) sweep. It changes one input while holding the others at a selected base
case. That is useful for showing direction and local response shape, but it is
**not a Sobol sensitivity analysis**.

Sobol analysis is global and variance based. It samples all uncertain inputs
over their stated distributions and decomposes the variance of a chosen model
output. The input ranges, probability distributions, dependencies, model, and
output statistic are therefore part of the result. Change any of them and the
indices can change.

## Proposed representation: the Sobol Variance Budget Map

The new representation is implemented in
`scripts/plot_sobol_variance_map.py`. It is a coordinated three-panel figure:

1. **Variance lanes** place the first-order estimate `S1` and total-order
   estimate `ST` on the same row. The distance between them exposes interaction
   involvement while confidence whiskers remain visible.
2. **Priority plane** plots direct effect (`S1`) horizontally and interaction
   involvement (`ST - S1`) vertically. This distinguishes inputs that matter by
   themselves from inputs that matter mainly through combinations.
3. **Pairwise interaction matrix** displays `S2` for each input pair. Circle
   area shows magnitude, color preserves the estimated sign, and a dagger marks
   a confidence interval that includes zero.

This is “new” as a representation for this project. It should not be presented
as a claim that the chart form is unprecedented in the sensitivity-analysis
literature. Recent work explicitly argues that visual treatment improves the
interpretation of global sensitivity results, and sensitivity toolkits such as
`sensobol` already include several visual summaries. The contribution here is
the particular coordinated layout and its decision-oriented reading rules.

## What the Sobol results represent

Let the model output be (Y=f(X_1,\ldots,X_d)), and assume the inputs are
independent for the classical Sobol decomposition.

### First-order index: `S1`

\[
S_i = \frac{\operatorname{Var}_{X_i}\left(\mathbb{E}[Y\mid X_i]\right)}
{\operatorname{Var}(Y)}.
\]

`S1` is the fraction of output variance attributable to an input acting alone,
after averaging over the distributions of the other inputs.

- `S1 = 0.46` means about 46% of the output variance is associated with that
  input's main effect under the stated uncertainty model.
- A large `S1` identifies a strong factor for **prioritisation**: learning or
  controlling that input can substantially reduce output uncertainty.
- `S1` does not state whether increasing the input increases or decreases the
  output. Sobol indices measure variance contribution, not direction.

### Second-order index: `S2`

\[
S_{ij}=\frac{V_{ij}}{\operatorname{Var}(Y)}.
\]

`S2` is the fraction of variance caused by the non-additive interaction between
two inputs beyond their separate main effects. A large `S2` means the effect of
one input depends strongly on the value of the other.

### Total-order index: `ST`

\[
S_{T_i}=S_i+\sum_{j\ne i}S_{ij}+\sum_{j<k;\,j,k\ne i}S_{ijk}+\cdots.
\]

`ST` includes the main effect of input (i) and every interaction of every
order containing (i).

- A small `ST` supports **factor fixing**: within the analysed range, fixing
  that input is unlikely to remove much output variance.
- If `ST` is much larger than `S1`, the input is interaction dependent.
- `ST - S1` is useful as an interaction-involvement diagnostic, but it is not a
  unique additive share. The same pairwise interaction contributes to the total
  index of both participating inputs.

For an exact decomposition with independent inputs, first-order indices sum to
at most one, and all variance components across every order sum to one. Total
indices may sum to more than one because interactions are counted for each
input involved.

## How to read the map

| Pattern | Interpretation | Practical response |
|---|---|---|
| High `S1`, `ST` close to `S1` | Strong, mostly additive driver | Prioritise measurement or control of this input |
| Low `S1`, high `ST` | Weak alone, important through interactions | Do not fix it without checking its partners |
| High `S1`, high `ST - S1` | Strong alone and in combinations | Treat as a dominant design variable; inspect `S2` |
| Low `ST` | Little contribution within the chosen distributions | Candidate for fixing or simplifying |
| Large `S2(i,j)` | Non-additive pair | Explore a two-dimensional response surface for the pair |

## What the results do *not* establish

- **Causality:** the indices describe the supplied model and uncertainty setup.
- **Direction:** a large index does not say whether the response rises or falls.
- **Importance outside the chosen ranges:** a narrow range can make a physically
  important parameter appear insensitive.
- **Validity under correlated inputs:** classical Sobol interpretation assumes
  independence. For dependent inputs, use a dependence-aware method such as
  Shapley effects or an explicitly documented dependent-input decomposition.
- **Numerical certainty:** small negative estimates or `S1 > ST` can occur from
  Monte Carlo error. They are diagnostics to increase the base sample and check
  convergence, not evidence of negative importance.

## Confidence intervals and reporting checklist

Confidence intervals quantify estimator uncertainty from finite sampling. A
result is more trustworthy when the ranking is stable as the base sample size
increases and the intervals become narrow. Always report:

- the model and exact scalar output (for example, maximum absolute deflection),
- every input distribution and range,
- whether inputs are assumed independent,
- the Sobol estimator and base sample size,
- first-, total-, and—when needed—second-order indices,
- confidence level and resampling method,
- a convergence check over increasing sample sizes.

SALib returns `S1`, `ST`, and their confidence half-widths, plus `S2` when
second-order calculation is enabled. Its documentation also notes that small
negative estimates can result from computational error and normally shrink as
the sample size increases.

## Using the plotting script

The accepted JSON structure matches common SALib output:

```json
{
  "names": ["E", "h", "k", "delta_T"],
  "S1": [0.10, 0.40, 0.20, 0.05],
  "S1_conf": [0.02, 0.04, 0.03, 0.02],
  "ST": [0.22, 0.60, 0.32, 0.12],
  "ST_conf": [0.03, 0.05, 0.04, 0.03],
  "S2": [[null, 0.08, 0.02, 0.01], [null, null, 0.05, 0.02], [null, null, null, 0.01], [null, null, null, null]],
  "S2_conf": [[null, 0.03, 0.02, 0.02], [null, null, 0.03, 0.02], [null, null, null, 0.02], [null, null, null, null]],
  "metadata": {"output": "maximum absolute deflection |w|", "confidence_level": 0.95, "base_sample_size": 1024}
}
```

Run:

```bash
python scripts/plot_sobol_variance_map.py \
  --input examples/sobol_results_illustrative.json \
  --output runs/sobol_variance_budget_map.png
```

The included example is explicitly illustrative. It is not evidence about any
trained model in this repository.

## Expanded model and spatial-summary workflow

The production workflow now uses `configs/deeponet_mse_expanded.yaml`. It adds
two physically varying inputs that the original 11-feature model omitted:
thermal-expansion coefficient `alpha_T` and pressure magnitude `pressure_q`.
Legacy cached graphs are upgraded in memory from 11 to 13 features; the cache
and original checkpoints are not overwritten.

`scripts/run_sobol_qoi_analysis.py` performs separate Sobol analyses for:

- mean absolute deflection,
- median absolute deflection,
- 95th-percentile absolute deflection, and
- maximum absolute deflection.

It varies `E`, thickness, Winkler `k`, temperature change, `alpha_T`, and
pressure independently over their observed training ranges. Plate geometry and
the load footprint are fixed at a representative dataset case, so the indices
must be reported as **conditional on that geometry**.

The analysis runs at increasing base sample sizes and retains bootstrap
replicates. `scripts/plot_sobol_bootstrap_summary.py` represents their
variation as boxplots: the box is the interquartile range, the center line is
the median, and the diamond is the mean. This directly addresses mean-versus-
median reporting without treating the two as different definitions of
sensitivity.

## Project and similarity analyses

There are now two separate production Sobol analyses in this repository:

- `project/runs/sobol_expanded` evaluates the expanded physical-feature
  DeepONet.
- `similarity/runs/sobol_similarity` evaluates the as-trained similarity
  DeepONet after mapping each physical sample to `[l, phi, P/Q]`.

Both analyses use the same six physical input ranges, representative geometry,
sample sizes, random seed, and bootstrap settings, making their rankings
directly comparable. They must nevertheless be reported as separate
surrogate-model results. Agreement supports a robust physical conclusion;
disagreement is model-form uncertainty and should not be removed by averaging
the indices.

The similarity implementation has an important unit-audit caveat. Its transform
expects unit-weight and Winkler-modulus magnitudes that appear inconsistent with
the generated `.in2` files. The reported similarity indices preserve the
as-trained transform so they are valid for the existing checkpoint, but the
unit convention should be verified before presenting those indices as final
physical evidence. A correction would require graph regeneration and model
retraining.

## Sources

- Sobol, I. M. (2001), “Global sensitivity indices for nonlinear mathematical
  models and their Monte Carlo estimates”: https://doi.org/10.1016/S0378-4754(00)00270-6
- SALib Sobol analyser documentation and returned fields:
  https://salib.readthedocs.io/en/latest/_modules/SALib/analyze/sobol.html
- Zhang et al. (2015), explanation and applied examples of first-, second-, and
  total-order indices: https://pmc.ncbi.nlm.nih.gov/articles/PMC5006244/
- Iooss and Prieur (2019), Shapley effects for correlated inputs:
  https://doi.org/10.1137/18M1186259
- Puy et al. (2022), `sensobol`, including visual representation of
  variance-based sensitivity indices: https://doi.org/10.18637/jss.v102.i05
- Svitelman and Saveleva (2024), visualization for understanding global
  sensitivity-analysis results: https://doi.org/10.1016/j.jenvrad.2024.107483
