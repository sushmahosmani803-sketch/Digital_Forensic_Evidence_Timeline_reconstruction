import json

import pandas as pd
import pytest

pytest.importorskip("plotly")

from evidence import charts, config  # noqa: E402
from evidence.timeline import reconstruct_timeline, user_risk_profiles  # noqa: E402

OUT = config.OUTPUT_DIR
pytestmark = pytest.mark.skipif(not (OUT / config.SCORED_FILE).exists(), reason="run `python pipeline.py` first")


@pytest.fixture(scope="module")
def scored():
    return pd.read_csv(OUT / config.SCORED_FILE, parse_dates=["timestamp"])


@pytest.fixture(scope="module")
def models():
    return json.loads((OUT / config.METRICS_FILE).read_text())["models"]


def test_overview_figures(scored):
    for builder in (charts.density_strip, charts.activity_over_time, charts.hour_weekday_heatmap,
                    charts.source_breakdown, charts.risk_histogram):
        fig = builder(scored)
        assert len(fig.data) >= 1
        fig.to_json()  # must serialise for the browser


def test_heatmap_counts_flagged_events(scored):
    fig = charts.hour_weekday_heatmap(scored)
    assert int(sum(sum(r) for r in fig.data[0].z)) == int(scored["flagged"].sum())


def test_timeline_and_people_figures(scored):
    user = scored.groupby("user")["risk_score"].max().idxmax()
    tl = reconstruct_timeline(scored, user=user)
    fig = charts.timeline_scatter(tl)
    assert len(fig.data[0].x) == len(tl)
    fig.to_json()
    charts.user_bubble(user_risk_profiles(scored)).to_json()


def test_model_figures(models):
    keys = list(models)
    charts.metric_comparison(models, keys).to_json()
    charts.roc_curves(models, keys).to_json()
    charts.confusion_heatmap(models["enhanced_rf"]["confusion_matrix"]).to_json()
    charts.importance_bars(models["enhanced_rf"]["feature_importance"]).to_json()
