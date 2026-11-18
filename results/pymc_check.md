# One-spread model: grid against PyMC

One synthetic world (seed 1), a typical 3 past cases per surgeon-procedure pair, 132 past cases in 25 pairs. Grid: 4,000 draws. PyMC: NUTS, 4 chains of 1,000 draws after 1,000 tuning steps each.

## tau (how far pairs differ from their procedure and surgeon averages, log scale; true value 0.08)

| | 5% | 25% | median | 75% | 95% |
|---|---|---|---|---|---|
| Grid | 0.006 | 0.030 | 0.061 | 0.105 | 0.184 |
| PyMC, non-centred | 0.006 | 0.030 | 0.059 | 0.099 | 0.171 |
| PyMC, centred | 0.022 | 0.044 | 0.072 | 0.111 | 0.182 |

## Pair averages (log scale)

- Largest difference between the grid's and PyMC's posterior medians: 0.014 (the pair averages themselves range from 2.81 to 5.18).
- Largest difference in the ends of the 90% intervals: 0.038.
- 90% intervals containing the true value: grid 76%, PyMC 72%.

## Diagnostics

| Form | Divergences | Largest R-hat | Smallest effective sample size | ESS for tau | Seconds |
|---|---|---|---|---|---|
| Non-centred | 1 | 1.005 | 860 | 860 | 21 |
| Centred | 190 | 1.040 | 190 | 190 | 31 |

The grid fit takes about a second.
