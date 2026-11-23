import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evidence import config  # noqa: E402
from evidence.features import build_features  # noqa: E402
from evidence.normalize import normalize_raw_csv  # noqa: E402

RAW_SAMPLE = """,file,id,timestamp,user,pc,activity,resource,case id;Cattivi
0,logon,{A1},02/24/2011 19:49:41,WDD0366,PC-0155,Logon,WDD0366,17;1
1,device,{A2},02/24/2011 22:07:28,WDD0366,PC-0155,Connect,WDD0366,17;1
2,http,{A3},02/24/2011 22:10:00,WDD0366,PC-0155,http://wikileaks.org/page.php,WDD0366,17;1
10,{B1},2010-02-23 15:43:45,PC-0151,consult http,SBN0971,17;0
11,{B2},2010-02-23 15:50:00,PC-0151,send email,SBN0971,17;0
12,{B3},2010-02-27 09:00:00,PC-0151,consult file,SBN0971,17;0
13,{B4},2010-03-01 08:00:00,PC-0999,Logoff,SBN0971,18;0
bad,row
"""


@pytest.fixture
def raw_sample_text():
    return RAW_SAMPLE


@pytest.fixture
def raw_sample_file(tmp_path):
    p = tmp_path / "raw.csv"
    p.write_text(RAW_SAMPLE, encoding="utf-8")
    return p


@pytest.fixture
def sample_df(raw_sample_file):
    return normalize_raw_csv(raw_sample_file)


@pytest.fixture(scope="session")
def full_normalized():
    return normalize_raw_csv(config.RAW_FILE)


@pytest.fixture(scope="session")
def full_features(full_normalized):
    return build_features(full_normalized)


@pytest.fixture(scope="session")
def small_raw_file(tmp_path_factory, full_normalized):
    """A ~900 row raw file (both layouts) for fast end-to-end tests."""
    src = config.RAW_FILE.read_text(encoding="utf-8-sig").splitlines()
    header, rows = src[0], src[1:]
    pick = rows[::5]
    p = tmp_path_factory.mktemp("raw") / "small_raw.csv"
    p.write_text("\n".join([header] + pick) + "\n", encoding="utf-8")
    return p


def make_events(rows):
    """Helper: build a small event frame from (user, pc, timestamp, activity, source) tuples."""
    return pd.DataFrame(rows, columns=["user", "pc", "timestamp", "activity", "source_type"]).assign(
        timestamp=lambda d: pd.to_datetime(d["timestamp"])
    )
