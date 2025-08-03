import pandas as pd
import QuantLib as ql
from datetime import datetime
import sys
import math
import os
sys.path.append(r"C:\Users\ksharma12\fin_research\scripts")
from fetch_options_to_excel import save_with_formatting

INPUT_FILE = r"C:\Users\ksharma12\fin_research\scripts\excels\avgo_options_2025-08-15.xlsx"
OUTPUT_FILE = r"C:\Users\ksharma12\fin_research\scripts\excels\avgo_options_2025-08-15_heston.xlsx"

# Helper: Calibrate Heston model to market prices for a given expiry
def calibrate_heston(df, valuation_date, risk_free_rate=0.05, dividend_yield=0.0):
    S = float(df["Current Price"].iloc[0])
    calendar = ql.UnitedStates(ql.UnitedStates.NYSE)
    day_count = ql.Actual365Fixed()
    ql_date = ql.Date(valuation_date.day, valuation_date.month, valuation_date.year)
    ql.Settings.instance().evaluationDate = ql_date
    flat_ts = ql.YieldTermStructureHandle(ql.FlatForward(ql_date, risk_free_rate, day_count))
    dividend_ts = ql.YieldTermStructureHandle(ql.FlatForward(ql_date, dividend_yield, day_count))
    spot_handle = ql.QuoteHandle(ql.SimpleQuote(S))
    # Initial guess for Heston params: v0, kappa, theta, sigma, rho
    v0 = (df["Implied Volatility"].mean() / 100) ** 2
    kappa = 2.0
    theta = v0
    sigma = 0.3
    rho = -0.7
    heston_process = ql.HestonProcess(flat_ts, dividend_ts, spot_handle, v0, kappa, theta, sigma, rho)
    heston_model = ql.HestonModel(heston_process)
    engine = ql.AnalyticHestonEngine(heston_model)
    # Build calibration helpers
    helpers = []
    market_prices = []
    for _, row in df.iterrows():
        K = float(row["Strike"])
        T = (datetime.strptime(row["Expiration"], "%Y-%m-%d") - valuation_date).days / 365.0
        try:
            vol = float(row["Implied Volatility"]) / 100
        except (ValueError, TypeError):
            continue
        if T <= 0 or math.isnan(K) or math.isnan(T) or math.isnan(vol) or vol <= 0:
            continue
        option_type = ql.Option.Call if row["Option Type"] == "Call" else ql.Option.Put
        payoff = ql.PlainVanillaPayoff(option_type, K)
        expiry_ql = ql.Date(datetime.strptime(row["Expiration"], "%Y-%m-%d").day, datetime.strptime(row["Expiration"], "%Y-%m-%d").month, datetime.strptime(row["Expiration"], "%Y-%m-%d").year)
        exercise = ql.EuropeanExercise(expiry_ql)
        helper = ql.HestonModelHelper(
            ql.Period(int(T * 365), ql.Days),
            calendar,
            S,
            K,
            ql.QuoteHandle(ql.SimpleQuote(vol)),
            flat_ts,
            dividend_ts
        )
        helper.setPricingEngine(engine)
        helpers.append(helper)
        # Save market price for diagnostics
        try:
            market_prices.append(float(row["PX_LAST"]))
        except (ValueError, TypeError):
            market_prices.append(float('nan'))
    lm = ql.LevenbergMarquardt()
    heston_model.calibrate(helpers, lm, ql.EndCriteria(1000, 500, 1e-8, 1e-8, 1e-8))
    params = heston_model.params()
    # Diagnostics: compute model and market prices for helpers
    model_prices = [h.modelValue() for h in helpers]
    market_prices = [h.marketValue() for h in helpers]
    # Compute RMSE, ignoring NaNs
    import numpy as np
    diffs = [mp - mpkt for mp, mpkt in zip(model_prices, market_prices) if not math.isnan(mp) and not math.isnan(mpkt)]
    rmse = np.sqrt(np.mean([d**2 for d in diffs])) if diffs else float('nan')
    print(f"Calibration RMSE: {rmse:.4f}")
    return params, heston_model, helpers, model_prices, market_prices

def heston_price_row(row, heston_model, valuation_date, risk_free_rate=0.05, dividend_yield=0.0):
    S = float(row["Current Price"])
    K = float(row["Strike"])
    expiry = datetime.strptime(row["Expiration"], "%Y-%m-%d")
    option_type = ql.Option.Call if row["Option Type"] == "Call" else ql.Option.Put
    calendar = ql.UnitedStates(ql.UnitedStates.NYSE)
    day_count = ql.Actual365Fixed()
    ql_date = ql.Date(valuation_date.day, valuation_date.month, valuation_date.year)
    ql.Settings.instance().evaluationDate = ql_date
    expiry_ql = ql.Date(expiry.day, expiry.month, expiry.year)
    payoff = ql.PlainVanillaPayoff(option_type, K)
    exercise = ql.EuropeanExercise(expiry_ql)
    spot_handle = ql.QuoteHandle(ql.SimpleQuote(S))
    flat_ts = ql.YieldTermStructureHandle(ql.FlatForward(ql_date, risk_free_rate, day_count))
    dividend_ts = ql.YieldTermStructureHandle(ql.FlatForward(ql_date, dividend_yield, day_count))
    engine = ql.AnalyticHestonEngine(heston_model)
    option = ql.VanillaOption(payoff, exercise)
    option.setPricingEngine(engine)
    try:
        price = option.NPV()
    except Exception:
        price = float('nan')
    return pd.Series({"Heston_Price": price})

def main():
    df = pd.read_excel(INPUT_FILE)
    valuation_date = datetime.today()
    risk_free_rate = 0.0453  # 4.53% from Bloomberg
    if df["Current Price"].isna().any():
        raise ValueError("Current Price column contains NaN values. Please check your input data.")
    params, heston_model, helpers, model_prices, market_prices = calibrate_heston(df, valuation_date, risk_free_rate=risk_free_rate)
    print(f"Calibrated Heston params: v0={params[0]}, kappa={params[1]}, theta={params[2]}, sigma={params[3]}, rho={params[4]}")
    # Map model prices back to the DataFrame for calibration error
    heston_calib_error = []
    helper_idx = 0
    for _, row in df.iterrows():
        try:
            vol = float(row["Implied Volatility"])
        except (ValueError, TypeError):
            heston_calib_error.append(float('nan'))
            continue
        if vol > 0 and helper_idx < len(model_prices):
            heston_calib_error.append(market_prices[helper_idx] - model_prices[helper_idx])
            helper_idx += 1
        else:
            heston_calib_error.append(float('nan'))
    df["Heston_Calib_Error"] = heston_calib_error
    heston_results = df.apply(lambda row: heston_price_row(row, heston_model, valuation_date, risk_free_rate=risk_free_rate), axis=1)
    df_heston = pd.concat([df, heston_results], axis=1)
    save_with_formatting(df_heston, OUTPUT_FILE)
    # Find and output all strikes where Heston_Price < PX_LAST and Heston_Price is valid and > 0 and Option Type is Call
    strikes_below_market = df_heston[(df_heston["Heston_Price"].notna()) & (df_heston["Heston_Price"] > 0) & (df_heston["Heston_Price"] < df_heston["PX_LAST"]) & (df_heston["Option Type"] == "Call")]
    # Use the original Option Ticker from the DataFrame
    if "Option Ticker" not in strikes_below_market.columns and "Ticker" in strikes_below_market.columns:
        strikes_below_market = strikes_below_market.rename(columns={"Ticker": "Option Ticker"})
    # Only keep relevant columns
    keep_cols = [col for col in ["Strike", "Heston_Price", "PX_LAST", "Option Ticker"] if col in strikes_below_market.columns]
    strikes_below_market = strikes_below_market[keep_cols]
    print("Strikes where Heston_Price < PX_LAST and Heston_Price > 0 (Calls only):")
    print(strikes_below_market)
    # Optionally, save to Excel with absolute path
    strikes_below_market_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "excels", "avgo_strikes_below_market.xlsx"))
    strikes_below_market.to_excel(strikes_below_market_path, index=False)

if __name__ == "__main__":
    main() 