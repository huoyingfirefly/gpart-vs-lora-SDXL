| config | recipe | modules | covered_weights_M | params_M | alpha | dW_norm | rms_per_weight | score_p0 | n_seeds |
|---|---|---|---|---|---|---|---|---|---|
| base | - | - | 0 | 0 | - | 0 | 0 | 0.08 | 6 |
| lo_char_kohya | kohya lr1e-4 cosine | all-linear r16 | 2184.0 | 42.7 | 1.0 | 8.2 | 0.000103 | 0.05 | 6 |
| lo_char_matched | matched lr5e-4 const | all-linear r8 | 2184.0 | 20.9 | 1.0 | 68.77 | None | 0.7 | 6 |
| gp_char | matched lr5e-4 const | attn2 | 545.6 | 23.2 | 1.0 | 78.29 | 0.003355 | 0.43 | 6 |
| gp_char_alllinear | matched lr5e-4 const | all-linear | 2184.0 | 23.2 | 1.0 | None | None | 0.4 | 6 |
| gp_char_alllinear | matched lr5e-4 const | all-linear | 2184.0 | 23.2 | 1.3 | None | 0.0029 | 0.55 | 6 |
| gp_char_alllinear | matched lr5e-4 const | all-linear | 2184.0 | 23.2 | 1.5 | None | None | 0.57 | 6 |
| gp_char_alllinear | matched lr5e-4 const | all-linear | 2184.0 | 23.2 | 1.8 | None | None | 0.4 | 6 |
