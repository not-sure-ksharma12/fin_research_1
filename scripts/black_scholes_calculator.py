import pandas as pd
import QuantLib as ql
from datetime import datetime
import sys
sys.path.append(r"C:\Users\ksharma12\fin_research\scripts")
from fetch_options_to_excel import save_with_formatting

INPUT_FILE = r"C:\Users\ksharma12\fin_research\scripts\amd_options_2025-12-19.xlsx"
OUTPUT_FILE = r"C:\Users\ksharma12\fin_research\scripts\amd_options_2025-12-19_bs.xlsx"

# Black-Scholes calculation function
def black_scholes_row(row, valuation_date, risk_free_rate=0.046, dividend_yield=0.0):
    S = float(row["Current Price"])
    K = float(row["Strike"])
    expiry = datetime.strptime(row["Expiration"], "%Y-%m-%d")
    T = (expiry - valuation_date).days / 365.0
    sigma = float(row["Implied Volatility"]) / 100 if row["Implied Volatility"] else 0.2
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
    vol_ts = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(ql_date, calendar, sigma, day_count))
    process = ql.BlackScholesMertonProcess(spot_handle, dividend_ts, flat_ts, vol_ts)
    engine = ql.AnalyticEuropeanEngine(process)
    option = ql.VanillaOption(payoff, exercise)
    option.setPricingEngine(engine)
    try:
        price = option.NPV()
        delta = option.delta()
        gamma = option.gamma()
        vega = option.vega()
        theta = option.theta()
        rho = option.rho()
    except Exception:
        price = delta = gamma = vega = theta = rho = float('nan')
    return pd.Series({
        "BS_Price": price,
        "BS_Delta": delta,
        "BS_Gamma": gamma,
        "BS_Vega": vega,
        "BS_Theta": theta,
        "BS_Rho": rho
    })

def main():
    df = pd.read_excel(INPUT_FILE)
    valuation_date = datetime.today()
    bs_results = df.apply(lambda row: black_scholes_row(row, valuation_date), axis=1)
    df_bs = pd.concat([df, bs_results], axis=1)
    save_with_formatting(df_bs, OUTPUT_FILE)

if __name__ == "__main__":
    main() 