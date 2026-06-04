# Two-Household Ramsey Equilibrium Dashboard

This repository contains a Streamlit dashboard for the two-household logarithmic Ramsey equilibrium model with Cobb-Douglas production.

## Inputs

- `alpha` in `(0, 1)`
- `delta_1` in `(0, 1)`
- `delta_2` in `(0, delta_1)`
- `x^1_0 >= 0`
- `x^2_0 >= 0`

The TeX model assumes `K_0 = x^1_0 + x^2_0 <= 1`.

## Outputs

For a selected finite horizon, the app reports and plots:

- aggregate capital `K_t`
- household 1 capital `x^1_t`
- household 2 capital `x^2_t`
- household 1 capital share `x^1_t / K_t`
- household 2 capital share `x^2_t / K_t`
- the exit time `T` such that `x^2_T = 0`
- the active-block length and final-exit ratio selected by the backward-shooting construction

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy

Push these files to a GitHub repository and deploy the app through Streamlit Community Cloud. Use `app.py` as the main file.
