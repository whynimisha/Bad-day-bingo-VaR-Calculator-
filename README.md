# Monte Carlo VaR / CVaR Engine

[![Live Demo – Streamlit](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://bad-day-bingo.streamlit.app/)

WHAT IS IT?

Bad Day Bingo — a tiny app that answers one question: “If the market throws a tantrum, how much could I lose?”
Upload a portfolio and run. It simulates a bunch of alternate tomorrows, then shows where your “bad-day” losses land. Toggle spicy tails (more crashy days), switch 1/5/10/20-day horizons, and peek at stress tests. It won’t predict the future—but it’ll stop you from pretending you don’t have one. 

HIRE ME MANIFESTO

Monte Carlo VaR/CVaR engine with multi-model risk and automated reporting.
Implements Gaussian, Student-t (fat-tail), and Historical bootstrap simulations with Cholesky-preserved correlations. Supports multi-day horizons (1/5/10/20d), stress tests (±kσ), and a rolling backtest that reports 95% VaR exceedances. Artifacts (plots + JSON) are auto-generated daily via GitHub Actions; a Streamlit UI enables interactive runs and toggling model assumptions.

Run:
python var_engine.py --mode t --df 6 --horizon 1 --scenarios 20000 --kde

Streamlit demo:
streamlit run app.py

![Daily VaR Report](https://github.com/whynimisha/var-engine/actions/workflows/var.yml/badge.svg)
