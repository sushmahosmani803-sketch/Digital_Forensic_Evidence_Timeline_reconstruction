import pandas as pd
import pytest

from evidence.normalize import (
    SCHEMA,
    display_activity,
    infer_source_type,
    normalize_raw_csv,
    normalize_row,
    split_case_label,
)


@pytest.mark.parametrize(
    "activity,expected",
    [
        ("consult http", "http"),
        ("http://wikileaks.org/x", "http"),
        ("send email", "email"),
        ("consult file", "file"),
        ("Logon", "logon"),
        ("logoff", "logon"),
        ("Connect", "device"),
        ("Disconnect", "device"),
        ("print document", "unknown"),
        ("", "unknown"),
        (None, "unknown"),
    ],
)
def test_infer_source_type(activity, expected):
    assert infer_source_type(activity) == expected


@pytest.mark.parametrize(
    "value,expected",
    [("17;1", ("17", "1")), ("201;0", ("201", "0")), ("5", ("5", "")), ("", ("", "")), (" 3 ; 1 ", ("3", "1"))],
)
def test_split_case_label(value, expected):
    assert split_case_label(value) == expected


def test_display_activity_shortens_urls():
    assert display_activity("http://wikileaks.org/a/b.php") == "visit wikileaks.org"
    assert display_activity("send email") == "send email"


def test_normalize_standard_row():
    rec = normalize_row(["0", "logon", "{X}", "02/24/2011 19:49:41", "U1", "PC-1", "Logon", "U1", "17;1"])
    assert rec["record_format"] == "standard"
    assert rec["source_type"] == "logon"
    assert rec["user"] == "U1" and rec["pc"] == "PC-1"
    assert rec["case_id"] == "17" and rec["label"] == "1"


def test_normalize_short_row_reorders_fields():
    rec = normalize_row(["9", "{Y}", "2010-02-23 15:43:45", "PC-0151", "send email", "SBN0971", "17;0"])
    assert rec["record_format"] == "short"
    assert rec["user"] == "SBN0971"
    assert rec["pc"] == "PC-0151"
    assert rec["source_type"] == "email"
    assert rec["resource"] == ""
    assert rec["label"] == "0"


def test_normalize_rejects_unexpected_layout():
    assert normalize_row(["a", "b"]) is None


def test_normalize_sample_file(sample_df):
    assert list(sample_df.columns[: len(SCHEMA)]) == SCHEMA
    assert len(sample_df) == 7
    assert sample_df.attrs["skipped_rows"] == 1
    assert pd.api.types.is_datetime64_any_dtype(sample_df["timestamp"])
    assert sample_df["timestamp"].notna().all()
    # Mixed date formats are parsed correctly (month-first US format).
    assert sample_df.loc[0, "timestamp"] == pd.Timestamp("2011-02-24 19:49:41")
    assert sample_df["label"].tolist() == [1, 1, 1, 0, 0, 0, 0]


def test_normalize_accepts_bytes_and_file_objects(raw_sample_text):
    import io

    a = normalize_raw_csv(raw_sample_text.encode())
    b = normalize_raw_csv(io.BytesIO(raw_sample_text.encode()))
    assert len(a) == len(b) == 7


def test_full_dataset_normalisation(full_normalized):
    df = full_normalized
    assert len(df) == 4615
    assert df["timestamp"].isna().sum() == 0
    assert df["user"].isna().sum() == 0
    assert df["event_id"].is_unique
    assert df["label"].value_counts().to_dict() == {0: 3341, 1: 1274}
    assert set(df["source_type"]) <= {"http", "email", "file", "logon", "device"}
    assert df["timestamp"].min() >= pd.Timestamp("2010-01-01")
    assert df["timestamp"].max() <= pd.Timestamp("2011-12-31")


def test_label_is_confounded_with_record_format(full_normalized):
    """Documents WHY record_format must never be a model feature."""
    ct = pd.crosstab(full_normalized["record_format"], full_normalized["label"])
    assert ct.loc["short", 1] == 0 and ct.loc["standard", 0] == 0
