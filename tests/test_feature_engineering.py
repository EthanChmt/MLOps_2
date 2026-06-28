import pandas as pd

from api.main import apply_feature_engineering


def test_apply_feature_engineering_creates_main_ratios():
    """
    Vérifie que le feature engineering crée les principaux ratios attendus.
    """

    df = pd.DataFrame({
        "AMT_CREDIT": [2000],
        "AMT_INCOME_TOTAL": [1000],
        "AMT_ANNUITY": [50],
        "DAYS_BIRTH": [-10000],
        "DAYS_EMPLOYED": [-500]
    })

    df_engineered = apply_feature_engineering(df)

    assert "CREDIT_INCOME_PERCENT" in df_engineered.columns
    assert "ANNUITY_INCOME_PERCENT" in df_engineered.columns
    assert "CREDIT_TERM" in df_engineered.columns
    assert "DAYS_EMPLOYED_PERCENT" in df_engineered.columns

    assert df_engineered["CREDIT_INCOME_PERCENT"].iloc[0] == 2.0
    assert df_engineered["ANNUITY_INCOME_PERCENT"].iloc[0] == 0.05
    assert df_engineered["CREDIT_TERM"].iloc[0] == 0.025


def test_apply_feature_engineering_handles_days_employed_anomaly():
    """
    Vérifie que la valeur aberrante DAYS_EMPLOYED = 365243 est bien identifiée.
    """

    df = pd.DataFrame({
        "DAYS_EMPLOYED": [365243],
        "DAYS_BIRTH": [-10000]
    })

    df_engineered = apply_feature_engineering(df)

    assert "DAYS_EMPLOYED_ANOM" in df_engineered.columns
    assert df_engineered["DAYS_EMPLOYED_ANOM"].iloc[0] == True
    assert pd.isna(df_engineered["DAYS_EMPLOYED"].iloc[0])