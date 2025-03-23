import pandas as pd
import numpy as np
from neuralprophet import NeuralProphet
import warnings
import time
from datetime import timedelta

warnings.filterwarnings("ignore")

# User Configuration
FILE_PATH = ""
START_DATE = ""
END_DATE = ""
GROUP_BY_COL = ""
TARGET_COL = ""
FORECAST_PERIODS = 6
MIN_HISTORY_MONTHS = 12
SPECIFIC_COMPANY = None
EXCLUDE_PAYMENT_TYPES = None


def load_data(file_path, start_date, end_date):
    df = pd.read_csv(file_path, parse_dates=["Y_Month"])
    return df[(df["Y_Month"] >= start_date) & (df["Y_Month"] <= end_date)]


def calculate_overall_mape(actual, predicted):
    mask = actual.notna() & (actual != 0)
    if not mask.any():
        return np.nan
    return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100


def prepare_data_for_forecast(df, target_col):
    df_agg = df.groupby("Y_Month")[target_col].sum().reset_index()
    start_date = df_agg["Y_Month"].min()
    end_date = df_agg["Y_Month"].max()
    all_dates = pd.date_range(start=start_date, end=end_date, freq="MS")
    return pd.DataFrame({"ds": all_dates}).merge(
        df_agg.rename(columns={"Y_Month": "ds", target_col: "y"}), on="ds", how="left"
    )


def create_forecast(df, company_name, target_col, forecast_periods=6):
    prophet_df = prepare_data_for_forecast(df, target_col)
    if prophet_df["y"].notna().sum() < MIN_HISTORY_MONTHS:
        return None, None, None, False

    try:
        model = NeuralProphet(
            n_forecasts=forecast_periods,
            n_lags=min(6, prophet_df["y"].notna().sum() // 2),
            yearly_seasonality=prophet_df["y"].notna().sum() >= 12,
            weekly_seasonality=False,
            daily_seasonality=False,
            learning_rate=0.1,
            epochs=100,
        )
        metrics = model.fit(prophet_df.fillna({"y": 0}), freq="MS")
        in_sample = model.predict(prophet_df)
        future = model.make_future_dataframe(prophet_df, periods=forecast_periods)
        future_forecast = model.predict(future)
        return prophet_df, in_sample, future_forecast, True
    except Exception as e:
        print(f"Error forecasting for {company_name}: {str(e)}")
        return None, None, None, False


def prepare_results(prophet_df, in_sample, future_forecast, company_name):
    forecast_cols = [col for col in in_sample.columns if col.startswith("yhat")]
    in_sample["yhat"] = in_sample[forecast_cols].mean(axis=1)
    future_forecast["yhat"] = future_forecast[forecast_cols].mean(axis=1)

    historical = pd.DataFrame(
        {
            "company_name": company_name,
            "date": prophet_df["ds"],
            "actual": prophet_df["y"],
            "yhat": np.nan,
        }
    )

    for idx, row in historical.iterrows():
        date = row["date"]
        matching_pred = in_sample[in_sample["ds"] == date]
        if not matching_pred.empty:
            historical.at[idx, "yhat"] = matching_pred["yhat"].values[0]

    overall_mape = calculate_overall_mape(historical["actual"], historical["yhat"])
    future = pd.DataFrame(
        {
            "company_name": company_name,
            "date": future_forecast["ds"],
            "actual": np.nan,
            "yhat": future_forecast["yhat"],
        }
    )

    results = pd.concat([historical, future[future["date"] > historical["date"].max()]])
    results["overall_mape"] = overall_mape
    return results


def main():
    start_time = time.time()
    df = load_data(FILE_PATH, START_DATE, END_DATE)
    df = df[~df["payment_type"].isin(EXCLUDE_PAYMENT_TYPES)]

    companies = (
        SPECIFIC_COMPANY if SPECIFIC_COMPANY else df[GROUP_BY_COL].unique().tolist()
    )
    companies = (
        [item for sublist in companies for item in sublist]
        if isinstance(companies[0], list)
        else companies
    )

    all_results = pd.DataFrame()
    insufficient_data_companies = []

    for i, company in enumerate(companies):
        print(f"Processing company {i+1}/{len(companies)}: {company}")
        company_data = df[df[GROUP_BY_COL] == company]
        prophet_df, in_sample, future_forecast, success = create_forecast(
            df=company_data,
            company_name=company,
            target_col=TARGET_COL,
            forecast_periods=FORECAST_PERIODS,
        )
        if success:
            company_results = prepare_results(
                prophet_df, in_sample, future_forecast, company
            )
            all_results = pd.concat([all_results, company_results])
        else:
            insufficient_data_companies.append(
                (company, len(company_data["Y_Month"].unique()))
            )
            print(
                f"  - Insufficient data for {company}: only {len(company_data['Y_Month'].unique())} months available (need {MIN_HISTORY_MONTHS})"
            )

    if not all_results.empty:
        output_filename = f"_forecasts_{TARGET_COL}.csv"
        if SPECIFIC_COMPANY:
            output_filename = f"{SPECIFIC_COMPANY}_forecasts_{TARGET_COL}.csv"

        all_results.to_csv(output_filename, index=False)

        if insufficient_data_companies:
            pd.DataFrame(
                insufficient_data_companies,
                columns=["company_name", "available_months"],
            ).to_csv("companies_insufficient_data.csv", index=False)

        total_time = time.time() - start_time
        print(f"\nForecast completed in {str(timedelta(seconds=int(total_time)))}!")
        print(
            f"Successfully processed {len(companies) - len(insufficient_data_companies)} companies"
        )
        print(
            f"Companies with insufficient historical data: {len(insufficient_data_companies)}"
        )
        print(f"\nFiles saved:\n- {output_filename}")
        if insufficient_data_companies:
            print(f"- companies_insufficient_data.csv")
    else:
        print("\nNo results generated. All selected companies had insufficient data.")


if __name__ == "__main__":
    main()
