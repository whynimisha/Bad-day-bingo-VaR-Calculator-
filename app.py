# app.py
import os, json, tempfile, subprocess, sys
import streamlit as st
import pandas as pd  # optional, but fine to keep

# ---------- debug vars live in session state ----------
st.session_state.setdefault("dbg_cmd", None)
st.session_state.setdefault("dbg_cwd", None)

ENGINE = os.path.join(os.path.dirname(__file__), "var_engine.py")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
RESULTS_JSON = os.path.join(OUTPUT_DIR, "results.json")

st.set_page_config(page_title="VaR Engine", layout="wide")
st.title("📊 VaR Engine Web App")

# --- Fast sanity info (helps debug path issues) ---
with st.expander("Debug info"):
    st.write("CWD:", os.getcwd())
    st.write("Engine path exists:", os.path.exists(ENGINE))
    st.write("Outputs dir:", OUTPUT_DIR)
    st.write("Python exe:", sys.executable)

with st.expander("Debug: child process"):
    cmd_shown = (
        " ".join(st.session_state["dbg_cmd"])
        if isinstance(st.session_state["dbg_cmd"], (list, tuple))
        else (st.session_state["dbg_cmd"] or "(not run yet)")
    )
    cwd_shown = st.session_state["dbg_cwd"] or ""
    st.write({"cmd": cmd_shown, "cwd": cwd_shown})

# --- Controls ---
colA, colB, colC = st.columns(3)
with colA:
    mode = st.selectbox("Mode", ["t", "normal", "historical"], index=0)
    df = st.slider("Student-t df", 3, 30, 6)
    horizon = st.selectbox("Horizon (days)", [1, 5, 10, 20], index=0)
with colB:
    scenarios = st.number_input("Scenarios", 5_000, 200_000, 20_000, step=5_000)
    window = st.number_input("Window (days)", 60, 1000, 250, step=10)
    alpha = st.selectbox("Confidence", [0.95, 0.99], index=0)
with colC:
    rho_stress = st.slider("Correlation stress (0..1)", 0.0, 1.0, 0.6, 0.05)
    drift = st.selectbox("Drift", ["0", "auto", "custom %/yr"], index=1)
    drift_val = st.number_input("Annual drift % (if custom)", -50.0, 50.0, 8.0, step=0.5)

kde = st.checkbox("KDE overlay", value=True)
seed = st.number_input("Random seed (blank=0 disables)", 0, 1_000_000, 42, step=1)
liq_bp = st.number_input("Liquidity haircut (bp)", 0.0, 500.0, 20.0, step=5.0)
ci = st.number_input("VaR CI level (0=off)", 0, 99, 90, step=5)

uploaded = st.file_uploader(
    "Upload portfolio.csv (symbol,qty,[price],[multiplier])", type=["csv"]
)
run_btn = st.button("Run VaR")

def run_engine(args_list, portfolio_bytes=None):
    """
    Run var_engine.py with args_list.
    If portfolio_bytes is provided, save to a temp CSV and add --portfolio <tmp>.
    Returns: (returncode, stdout, stderr, cmd_debug, cwd_debug)
    """
    final_args = [str(x) for x in args_list]

    engine_path = os.path.abspath(ENGINE)
    workdir = os.path.abspath(os.path.dirname(engine_path) or os.getcwd())
    if not os.path.isdir(workdir):
        workdir = None  # inherit current CWD instead of passing a bad path

    tmp_path = None
    try:
        if portfolio_bytes:
            with tempfile.NamedTemporaryFile(
                mode="wb", suffix=".csv", delete=False, dir=workdir or None
            ) as tmp:
                tmp.write(portfolio_bytes)
                tmp.flush()
                tmp_path = tmp.name
            final_args = ["--portfolio", tmp_path] + final_args

        cmd = [sys.executable, engine_path, *final_args]
        proc = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True)
        return proc.returncode, proc.stdout, proc.stderr, cmd, workdir

    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass

def load_results():
    if not os.path.exists(RESULTS_JSON):
        return None
    with open(RESULTS_JSON, "r") as f:
        return json.load(f)

if run_btn:
    if not os.path.exists(ENGINE):
        st.error("var_engine.py not found next to app.py. Fix the path or move the files together.")
    else:
        args = [
            "--mode", mode,
            "--df", str(df),
            "--horizon", str(horizon),
            "--scenarios", str(scenarios),
            "--window", str(window),
            "--alpha", str(alpha),
            "--rho_stress", str(rho_stress),
        ]
        if kde:
            args += ["--kde"]
        if drift == "0":
            args += ["--drift", "0"]
        elif drift == "auto":
            args += ["--drift", "auto"]
        else:
            args += ["--drift", str(drift_val)]
        if seed > 0:
            args += ["--seed", str(seed)]
        if liq_bp > 0:
            args += ["--liq_haircut_bp", str(liq_bp)]
        if ci > 0:
            args += ["--ci", str(ci)]

        with st.status("Running engine…", expanded=False) as status:
            code, out, err, dbg_cmd, dbg_cwd = run_engine(
                args, uploaded.getvalue() if uploaded else None
            )
            st.session_state["dbg_cmd"] = dbg_cmd
            st.session_state["dbg_cwd"] = dbg_cwd
            status.update(
                label=("Run complete ✅" if code == 0 else "Run failed ❌"),
                state=("complete" if code == 0 else "error"),
            )

        st.subheader("Console output")
        st.code(out or "(no stdout)")
        if err:
            st.subheader("Errors / stderr")
            st.code(err)

        res = load_results()
        if not res:
            st.error("No outputs/results.json produced. Check the stderr above.")
        else:
            st.success("Loaded outputs/results.json")
            col1, col2 = st.columns(2)
            with col1:
                st.json(res.get("flags", {}))
                st.metric("Portfolio value", f"{res['portfolio_value']:,.0f}")
                st.write(
                    f"VaR 95%: {res['VaR']['95']:.0f} "
                    f"({res['as_pct_of_portfolio']['VaR95_pct']:.2f}%)"
                )
                st.write(
                    f"VaR 99%: {res['VaR']['99']:.0f} "
                    f"({res['as_pct_of_portfolio']['VaR99_pct']:.2f}%)"
                )
            with col2:
                st.subheader("Backtest")
                st.json(res.get("backtest", {}))

            st.subheader("Plots")
            main_plot = os.path.join(OUTPUT_DIR, f"pnl_hist_{horizon}d_{mode}.png")
            if os.path.exists(main_plot):
                st.image(main_plot, use_container_width=True)
            for h in [1, 5, 10, 20]:
                p = os.path.join(OUTPUT_DIR, f"pnl_hist_{h}d_{mode}.png")
                if os.path.exists(p):
                    st.image(p, use_container_width=True)
