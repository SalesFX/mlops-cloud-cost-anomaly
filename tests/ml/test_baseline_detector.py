import numpy as np
import pandas as pd
import pytest

from src.ml.baseline_detector import (
    BASELINE_OUTPUT_COLUMNS,
    REASON_SCORE,
    REQUIRED_INPUT_COLUMNS,
    THRESHOLDS,
    detect,
    load_features,
)

# ---------------------------------------------------------------------------
# Fixture: 10 records with exact values to test each rule and priority
# ---------------------------------------------------------------------------
#
# Record 0: normal — no rule fires
# Record 1: cost_above_7d only  (cost > avg_7d*2.5, NOT > avg_30d*3.0)
# Record 2: cost_above_30d only (cost > avg_30d*3.0, NOT > avg_7d*2.5)
# Record 3: high_cost_change only
# Record 4: high_usage_change only
# Record 5: missing_tag only
# Record 6: priority — both 7d and 30d fire → 30d wins
# Record 7: priority — 7d and high_cost_change fire → 7d wins
# Record 8: priority — 30d and missing_tag fire → 30d wins
# Record 9: NaN in cost_change_percent → no rule fires


@pytest.fixture
def sample_df() -> pd.DataFrame:
    records = [
        # (daily_cost, avg_7d, avg_30d, cost_pct, usage_pct, missing, is_anomaly, atype)
        (100.0, 100.0, 100.0, 10.0, 5.0, False, False, "none"),          # 0 normal
        (300.0, 100.0, 200.0, 10.0, 5.0, False, False, "none"),          # 1 7d only
        (350.0, 200.0, 100.0, 10.0, 5.0, False, False, "none"),          # 2 30d only
        (100.0, 100.0, 100.0, 160.0, 5.0, False, False, "none"),         # 3 cost_change only
        (100.0, 100.0, 100.0, 10.0, 210.0, False, False, "none"),        # 4 usage_change only
        (100.0, 100.0, 100.0, 10.0, 5.0, True, False, "none"),           # 5 missing_tag only
        (400.0, 100.0, 100.0, 10.0, 5.0, False, True, "cost_spike"),     # 6 7d+30d → 30d
        (300.0, 100.0, 200.0, 160.0, 5.0, False, False, "none"),         # 7 7d+cost_change → 7d
        (400.0, 100.0, 100.0, 10.0, 5.0, True, True, "cost_spike"),      # 8 30d+missing → 30d
        (100.0, 100.0, 100.0, np.nan, 5.0, False, False, "none"),        # 9 NaN cost_change
    ]
    columns = [
        "daily_cost", "avg_cost_7d", "avg_cost_30d",
        "cost_change_percent", "usage_change_percent",
        "is_missing_tag", "is_anomaly", "anomaly_type",
    ]
    return pd.DataFrame(records, columns=columns)


# ---------------------------------------------------------------------------
# Rule detection
# ---------------------------------------------------------------------------

class TestDetectRules:
    def test_normal_record_is_not_anomaly(self, sample_df):
        df = detect(sample_df)
        assert df.loc[0, "baseline_anomaly"] == False
        assert df.loc[0, "baseline_reason"] == "none"

    def test_cost_above_7d_average(self, sample_df):
        # daily=300 > avg_7d=100 * 2.5=250 ✓   avg_30d=200 * 3.0=600, 300<600 ✗
        df = detect(sample_df)
        assert df.loc[1, "baseline_reason"] == "cost_above_7d_average"
        assert df.loc[1, "baseline_anomaly"] == True

    def test_cost_above_30d_average(self, sample_df):
        # daily=350 > avg_30d=100 * 3.0=300 ✓   avg_7d=200 * 2.5=500, 350<500 ✗
        df = detect(sample_df)
        assert df.loc[2, "baseline_reason"] == "cost_above_30d_average"
        assert df.loc[2, "baseline_anomaly"] == True

    def test_high_cost_change(self, sample_df):
        # cost_pct=160 >= 150 ✓   cost rules: 100 < 250 and 100 < 300 ✗
        df = detect(sample_df)
        assert df.loc[3, "baseline_reason"] == "high_cost_change"
        assert df.loc[3, "baseline_anomaly"] == True

    def test_high_usage_change(self, sample_df):
        # usage_pct=210 >= 200 ✓   no cost rule fires
        df = detect(sample_df)
        assert df.loc[4, "baseline_reason"] == "high_usage_change"
        assert df.loc[4, "baseline_anomaly"] == True

    def test_missing_tag(self, sample_df):
        # is_missing_tag=True, no cost/change rules
        df = detect(sample_df)
        assert df.loc[5, "baseline_reason"] == "missing_tag"
        assert df.loc[5, "baseline_anomaly"] == True

    def test_nan_cost_change_does_not_trigger_rule(self, sample_df):
        # NaN >= 150 must evaluate to False, not raise
        df = detect(sample_df)
        assert df.loc[9, "baseline_reason"] == "none"
        assert df.loc[9, "baseline_anomaly"] == False


# ---------------------------------------------------------------------------
# Priority: higher-priority rule overwrites lower-priority one
# ---------------------------------------------------------------------------

class TestRulePriority:
    def test_30d_beats_7d_when_both_fire(self, sample_df):
        # record 6: daily=400 > avg_7d=100*2.5=250 AND daily=400 > avg_30d=100*3.0=300
        df = detect(sample_df)
        assert df.loc[6, "baseline_reason"] == "cost_above_30d_average"

    def test_7d_beats_high_cost_change(self, sample_df):
        # record 7: daily=300 > avg_7d=100*2.5=250 AND cost_pct=160>=150
        df = detect(sample_df)
        assert df.loc[7, "baseline_reason"] == "cost_above_7d_average"

    def test_30d_beats_missing_tag(self, sample_df):
        # record 8: 30d fires AND missing_tag fires → 30d wins
        df = detect(sample_df)
        assert df.loc[8, "baseline_reason"] == "cost_above_30d_average"


# ---------------------------------------------------------------------------
# Score and risk level
# ---------------------------------------------------------------------------

class TestScoreAndRisk:
    def test_none_score_is_zero(self, sample_df):
        df = detect(sample_df)
        assert df.loc[0, "baseline_score"] == pytest.approx(0.0)

    def test_missing_tag_score(self, sample_df):
        df = detect(sample_df)
        assert df.loc[5, "baseline_score"] == pytest.approx(REASON_SCORE["missing_tag"])

    def test_high_usage_change_score(self, sample_df):
        df = detect(sample_df)
        assert df.loc[4, "baseline_score"] == pytest.approx(REASON_SCORE["high_usage_change"])

    def test_cost_above_7d_score(self, sample_df):
        df = detect(sample_df)
        assert df.loc[1, "baseline_score"] == pytest.approx(REASON_SCORE["cost_above_7d_average"])

    def test_cost_above_30d_score(self, sample_df):
        df = detect(sample_df)
        assert df.loc[2, "baseline_score"] == pytest.approx(REASON_SCORE["cost_above_30d_average"])

    def test_30d_score_greater_than_7d_score(self):
        assert REASON_SCORE["cost_above_30d_average"] > REASON_SCORE["cost_above_7d_average"]

    def test_scores_in_range_0_to_1(self, sample_df):
        df = detect(sample_df)
        assert (df["baseline_score"] >= 0.0).all()
        assert (df["baseline_score"] <= 1.0).all()

    def test_none_risk_is_low(self, sample_df):
        df = detect(sample_df)
        assert df.loc[0, "baseline_risk_level"] == "low"

    def test_missing_tag_risk_is_medium(self, sample_df):
        df = detect(sample_df)
        assert df.loc[5, "baseline_risk_level"] == "medium"

    def test_high_usage_change_risk_is_medium(self, sample_df):
        df = detect(sample_df)
        assert df.loc[4, "baseline_risk_level"] == "medium"

    def test_high_cost_change_risk_is_high(self, sample_df):
        df = detect(sample_df)
        assert df.loc[3, "baseline_risk_level"] == "high"

    def test_cost_above_7d_risk_is_high(self, sample_df):
        df = detect(sample_df)
        assert df.loc[1, "baseline_risk_level"] == "high"

    def test_cost_above_30d_risk_is_high(self, sample_df):
        df = detect(sample_df)
        assert df.loc[2, "baseline_risk_level"] == "high"

    def test_risk_level_only_valid_values(self, sample_df):
        df = detect(sample_df)
        assert set(df["baseline_risk_level"].unique()).issubset({"low", "medium", "high"})

    def test_reason_only_valid_values(self, sample_df):
        df = detect(sample_df)
        valid = set(REASON_SCORE.keys())
        assert set(df["baseline_reason"].unique()).issubset(valid)


# ---------------------------------------------------------------------------
# Output schema and preservation
# ---------------------------------------------------------------------------

class TestOutputSchema:
    def test_all_baseline_columns_present(self, sample_df):
        df = detect(sample_df)
        for col in BASELINE_OUTPUT_COLUMNS:
            assert col in df.columns, f"Missing: {col}"

    def test_no_rows_lost(self, sample_df):
        df = detect(sample_df)
        assert len(df) == len(sample_df)

    def test_is_anomaly_preserved(self, sample_df):
        df = detect(sample_df)
        pd.testing.assert_series_equal(
            df["is_anomaly"].reset_index(drop=True),
            sample_df["is_anomaly"].reset_index(drop=True),
        )

    def test_anomaly_type_preserved(self, sample_df):
        df = detect(sample_df)
        pd.testing.assert_series_equal(
            df["anomaly_type"].reset_index(drop=True),
            sample_df["anomaly_type"].reset_index(drop=True),
        )

    def test_input_dataframe_not_mutated(self, sample_df):
        original_cols = list(sample_df.columns)
        detect(sample_df)
        assert list(sample_df.columns) == original_cols


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

class TestLoadFeatures:
    """load_features reads the full cloud_cost_features.csv which includes a date column."""

    def _csv_df(self, sample_df: pd.DataFrame) -> pd.DataFrame:
        """Add the date column required by load_features to the minimal fixture."""
        df = sample_df.copy()
        df["date"] = "2025-01-01"
        df["tag_project"] = ""
        return df

    def test_empty_strings_preserved(self, tmp_path, sample_df):
        df_with_tags = self._csv_df(sample_df)
        csv = tmp_path / "features.csv"
        df_with_tags.to_csv(csv, index=False)
        df = load_features(str(csv))
        assert (df["tag_project"] == "").all()
        assert not df["tag_project"].isna().any()

    def test_is_anomaly_loaded_as_bool(self, tmp_path, sample_df):
        csv = tmp_path / "features.csv"
        self._csv_df(sample_df).to_csv(csv, index=False)
        df = load_features(str(csv))
        assert df["is_anomaly"].dtype == bool

    def test_date_parsed_as_datetime(self, tmp_path, sample_df):
        csv = tmp_path / "features.csv"
        self._csv_df(sample_df).to_csv(csv, index=False)
        df = load_features(str(csv))
        assert pd.api.types.is_datetime64_any_dtype(df["date"])
