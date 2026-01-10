# pages/3_Analysis.py

import streamlit as st
from datetime import date
from typing import Any, Dict, List
import matplotlib.pyplot as plt

from db_init import init_db, get_all_trades
from portfolio import build_portfolio_view
from market import get_underlying_spot, compute_historical_volatility

from analysis import (
    Load_Contract_Context,
    Run_Analysis_Pipeline,
    Render_Findings,
    Visualise_Results,
)


def _rerun() -> None:
    fn = getattr(st, "rerun", None)
    if callable(fn):
        fn()
        return
    fn = getattr(st, "experimental_rerun", None)
    if callable(fn):
        fn()
        return
    raise RuntimeError("No rerun function available in this Streamlit version")


def _resolve_position(positions: List[Dict[str, Any]], contract_id: str) -> Dict[str, Any]:
    cid = str(contract_id).strip()
    if cid == "":
        raise ValueError("contract_id must be non-empty")
    for p in positions:
        if isinstance(p, dict) and str(p.get("contract_id", "")) == cid:
            return p
    raise ValueError("Selected contract_id not found in positions")


init_db()
st.set_page_config(page_title="Analysis", layout="wide")
st.title("Analysis")
st.caption("Monte Carlo GBM valuation pipeline with findings and visualisation.")


# Load positions

try:
    all_trades = get_all_trades()
except Exception as e:
    st.error(str(e))
    st.stop()

positions = build_portfolio_view(all_trades)

if len(positions) == 0:
    st.info("No open positions. Create a position from the Market page first.")
    st.stop()

contract_ids = [p["contract_id"] for p in positions]

# Selection + inputs

left, right = st.columns([1.2, 1.0])

with left:
    st.subheader("1) Select contract")
    selected_cid = st.selectbox("Open contracts", contract_ids)

    try:
        pos = _resolve_position(positions, selected_cid)
    except Exception as e:
        st.error(str(e))
        st.stop()

    st.write("Selected contract details")
    st.json(
        {
            "contract_id": pos["contract_id"],
            "ticker": pos["ticker"],
            "expiry": pos["expiry"],
            "strike": pos["strike"],
            "type": pos["type"],
            "net_quantity": pos["net_quantity"],
            "average_cost": pos.get("average_cost"),
        }
    )

with right:
    st.subheader("2) Market and modelling inputs")

    use_live_spot = st.checkbox(
        "Use live spot (market layer)",
        value=True,
        help="When enabled, spot is fetched from the market layer. When disabled, a manual spot is used.",
    )

    manual_spot = st.number_input(
        "Manual spot (used only if live spot is disabled)",
        min_value=0.0,
        value=0.0,
        step=1.0,
        format="%.2f",
    )

    # Auto-populate premium reference from average cost
    default_premium = float(pos.get("average_cost") or 0.0)
    
    premium_ref = st.number_input(
        "Premium reference (used for net profit probability)",
        min_value=0.0,
        value=default_premium,
        step=0.1,
        format="%.4f",
        help=f"Defaults to your average cost (${default_premium:.4f}). Compared against discounted payoff to estimate probability of clearing the premium.",
    )

    N = st.number_input("Simulations (N)", min_value=100, max_value=20000, value=3000, step=100)
    seed = st.number_input("Random seed", min_value=0, max_value=10**9, value=42, step=1)

    st.caption("Historical volatility is always pulled through the cached market path.")
    ensure_prices = st.button("Ensure historical prices exist for ticker")


# Ensure cached prices

if ensure_prices:
    try:
        tkr = str(pos["ticker"]).strip().upper()
        _ = compute_historical_volatility(tkr)
        st.success("Price history and volatility path is available for this ticker.")
    except Exception as e:
        st.error(str(e))


# Run pipeline

st.subheader("3) Run analysis pipeline")

run_btn = st.button("Run Monte Carlo analysis")

if run_btn:
    try:
        tkr = str(pos["ticker"]).strip().upper()

        if use_live_spot:
            spot = float(get_underlying_spot(tkr))
        else:
            spot = float(manual_spot)
            if spot <= 0:
                raise ValueError("Manual spot must be > 0 when live spot is disabled")

        ctx = Load_Contract_Context(
            ticker=tkr,
            expiry=str(pos["expiry"]),
            strike=float(pos["strike"]),
            option_type=str(pos["type"]),
            spot=float(spot),
            lookback_min_points=60,
        )

        ctx["premium_ref"] = float(premium_ref)
        ctx["N"] = int(N)
        ctx["seed"] = int(seed)

        out = Run_Analysis_Pipeline(ctx)
        inputs = out["inputs"]
        sim = out["sim"]
        metrics = out["metrics"]

        findings = Render_Findings(ctx, inputs, metrics)
        viz = Visualise_Results(sim, max_paths=30, bins=30)

        st.success("Pipeline completed.")

        # Results: findings

        st.subheader("Findings")
        for line in findings["summary_lines"]:
            st.write(line)

        if len(findings["flags"]) > 0:
            st.warning("Flags")
            for f in findings["flags"]:
                st.write(f)


        # Results: key numbers

        st.subheader("Key metrics")
        nums = findings["numbers"]
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.metric("Mean payoff", f"{nums['mc_mean']:.4f}")
        with k2:
            st.metric("Median payoff", f"{nums['mc_median']:.4f}")
        with k3:
            st.metric("P(ITM)", f"{nums['p_itm']:.4f}")
        with k4:
            st.metric("P(payoff > premium)", f"{nums['p_netprofit']:.4f}")

        st.write(f"5th percentile: {nums['q05']:.4f}")
        st.write(f"95th percentile: {nums['q95']:.4f}")

  
        # Visualisation: paths
    
        st.subheader("Simulated paths (subset)")
        paths = viz["paths"]
        if len(paths) == 0:
            st.info("No paths available for plotting.")
        else:
            K = float(ctx["strike"])
            fig = plt.figure()
            for p in paths:
                plt.plot(p)
            
            plt.axhline(y=K, color="r", linestyle="--", label="Strike price")
            
            
            plt.xlabel("Step")
            plt.ylabel("Underlying price")
            st.pyplot(fig)
        

   
        # Visualisation: histogram
    
        st.subheader("Discounted payoff histogram")
        hist = viz["payoff_hist"]
        edges = hist["bin_edges"]
        counts = hist["counts"]

        if len(edges) >= 2 and len(counts) >= 1:
            fig2 = plt.figure()
            widths = []
            mids = []
            for i in range(len(edges) - 1):
                widths.append(edges[i + 1] - edges[i])
                mids.append((edges[i + 1] + edges[i]) / 2.0)
            plt.bar(mids, counts, width=widths, align="center")
            plt.xlabel("Discounted payoff")
            plt.ylabel("Count")
            st.pyplot(fig2)
        else:
            st.info("Histogram data not available.")

  
        # Debug panel 
   
        with st.expander("Raw pipeline output", expanded=False):
            st.write("Inputs")
            st.json(inputs)
            st.write("Metrics")
            st.json(metrics)
            st.write("Simulation summary")
            st.json(
                {
                    "N": sim.get("N"),
                    "steps": sim.get("steps"),
                    "discount_factor": sim.get("discount_factor"),
                    "paths_subset_count": len(sim.get("paths_subset", [])),
                    "discounted_payoffs_count": len(sim.get("discounted_payoffs", [])),
                }
            )

    except Exception as e:
        st.error(str(e))

with st.expander("Monte Carlo metrics explained", expanded=False):
    st.markdown(
        """
Mean discounted payoff
Average payoff at expiry, discounted back to today using the risk free rate.

Median discounted payoff 
Middle outcome. Less sensitive to extreme tail paths than the mean.

5th and 95th percentiles
A rough downside and upside band. It describes the spread of outcomes, not a guarantee.

P(ITM)  
Fraction of simulations where payoff at expiry is positive.

P(payoff > premium)
Fraction of simulations where the discounted payoff exceeds the reference premium.
        """
    )

