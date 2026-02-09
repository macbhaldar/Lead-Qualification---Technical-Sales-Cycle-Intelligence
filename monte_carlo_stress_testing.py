# Monte Carlo Stress Testing for Sales Pipeline Volatility

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

## Simulation Function

def monte_carlo_pipeline(df, n_simulations=10000):
    results = []

    for _ in range(n_simulations):
        # Random shocks
        win_shock = np.random.normal(1, 0.2, len(df))
        value_shock = np.random.normal(1, 0.25, len(df))
        cost_shock = np.random.normal(1, 0.15, len(df))

        sim_win_prob = np.clip(
            df["win_probability"] * win_shock, 0, 1
        )
        sim_value = df["estimated_deal_value_usd"] * value_shock
        sim_cost = df["total_rfp_cost"] * cost_shock

        sim_expected_profit = (
            sim_win_prob * sim_value - sim_cost
        ).sum()

        results.append(sim_expected_profit)

    return np.array(results)


# Placeholder DataFrame - replace with your actual data
data = {
    'win_probability': [0.7, 0.5, 0.9, 0.3, 0.8],
    'estimated_deal_value_usd': [100000, 50000, 200000, 30000, 150000],
    'total_rfp_cost': [5000, 2000, 10000, 1000, 7000]
}
df = pd.DataFrame(data)

simulated_profits = monte_carlo_pipeline(df, n_simulations=10000)

## Risk Metrics

summary = {
    "Mean Profit": np.mean(simulated_profits),
    "Median Profit": np.median(simulated_profits),
    "5th Percentile (Worst Case)": np.percentile(simulated_profits, 5),
    "95th Percentile (Best Case)": np.percentile(simulated_profits, 95),
    "Probability of Loss": np.mean(simulated_profits < 0)
}

summary

## Visualize Distribution

plt.hist(simulated_profits, bins=50)
plt.axvline(np.mean(simulated_profits), linestyle="--")
plt.axvline(np.percentile(simulated_profits, 5), linestyle=":")
plt.show()

## Budget Cut Shock

df_budget = df.copy()
df_budget["total_rfp_cost"] *= 1.3

budget_sim = monte_carlo_pipeline(df_budget)

## Hiring Freeze Shock

df_freeze = df.copy()
df_freeze["win_probability"] *= 0.85

freeze_sim = monte_carlo_pipeline(df_freeze)

## Compare Outcomes

pd.DataFrame({
    "Baseline": simulated_profits,
    "Budget Cut": budget_sim,
    "Hiring Freeze": freeze_sim
}).describe(percentiles=[0.05, 0.5, 0.95])

## Risk-Adjusted ROI

risk_adjusted_profit = (
    np.mean(simulated_profits)
    - 0.5 * np.std(simulated_profits)
)
