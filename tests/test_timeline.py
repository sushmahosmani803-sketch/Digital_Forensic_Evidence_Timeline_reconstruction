import pandas as pd

from evidence.timeline import assign_sessions, detect_gaps, reconstruct_timeline, summarize_sessions, user_risk_profiles
from tests.conftest import make_events


def _events():
    return make_events(
        [
            ("a", "p1", "2024-01-01 10:00", "Logon", "logon"),
            ("a", "p1", "2024-01-01 10:10", "consult http", "http"),
            ("a", "p1", "2024-01-01 12:00", "send email", "email"),  # new session (>30 min)
            ("b", "p2", "2024-01-01 09:00", "Logon", "logon"),
            ("a", "p1", "2024-01-06 12:00", "Logoff", "logon"),  # 5-day gap
        ]
    ).assign(case_id=[1, 1, 1, 2, 1])


def test_reconstruct_timeline_is_chronological_and_numbered():
    t = reconstruct_timeline(_events())
    assert t["timestamp"].is_monotonic_increasing
    assert t["sequence"].tolist() == [1, 2, 3, 4, 5]
    assert t.loc[0, "user"] == "b"


def test_reconstruct_timeline_filters():
    ev = _events()
    assert set(reconstruct_timeline(ev, user="a")["user"]) == {"a"}
    assert len(reconstruct_timeline(ev, case_id=2)) == 1
    assert len(reconstruct_timeline(ev, pc="p1")) == 4
    win = reconstruct_timeline(ev, start="2024-01-01 10:05", end="2024-01-02")
    assert len(win) == 2


def test_assign_sessions():
    s = assign_sessions(_events(), gap_minutes=30)
    a = s[s["user"] == "a"].sort_values("timestamp")
    assert a["session_no"].tolist() == [1, 1, 2, 3]
    assert s.loc[3, "session_id"] == "b-S001"
    assert list(s.index) == list(_events().index)


def test_summarize_sessions():
    summ = summarize_sessions(_events().assign(risk_score=[0.9, 0.1, 0.2, 0.0, 0.7]))
    assert len(summ) == 4
    first_a = summ[summ["session_id"] == "a-S001"].iloc[0]
    assert first_a["events"] == 2
    assert first_a["duration_min"] == 10.0
    assert first_a["flagged"] == 1
    assert summ["start"].is_monotonic_increasing


def test_detect_gaps():
    g = detect_gaps(_events(), min_gap_hours=72)
    assert len(g) == 1
    assert g.loc[0, "user"] == "a"
    assert g.loc[0, "gap_hours"] == 120.0
    assert detect_gaps(_events(), min_gap_hours=1000).empty


def test_user_risk_profiles():
    ev = _events().assign(risk_score=[0.9, 0.8, 0.7, 0.1, 0.65], off_hours=[0, 0, 0, 0, 1])
    prof = user_risk_profiles(ev)
    assert prof.loc[0, "user"] == "a"
    a = prof.iloc[0]
    assert a["events"] == 4 and a["flagged_events"] == 4
    assert a["flag_rate"] == 1.0
    assert 0 <= a["threat_score"] <= 1
    assert a["risk_level"] in {"Critical", "High"}
    assert prof["threat_score"].is_monotonic_decreasing
    assert pd.api.types.is_datetime64_any_dtype(prof["first_seen"])
