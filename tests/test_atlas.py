"""Atlas logic on a small synthetic atlas (offline)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from monitor_nepal import boundaries
from monitor_nepal.analysis import atlas
from monitor_nepal.config import CLASS_KEYS


@pytest.fixture()
def fake_atlas(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    rows = []
    for d in boundaries.regions(2)[:12]:
        for lon, lat in boundaries.random_points(d, 6, seed=1):
            rows.append({"lon": lon, "lat": lat, "province": d.province, "district": d.name,
                         "district_code": d.code, "stratum_km2": d.area_km2, "local": None, "local_code": None})
    sites = pd.DataFrame(rows)
    sites.insert(0, "id", [f"a{i:05d}" for i in range(len(sites))])
    results = []
    for epoch, forest in (("eox-2016", 0.50), ("eox-2025", 0.55)):
        for sid in sites.id:
            f = rng.dirichlet(np.ones(7) * 0.5)
            f = f * (1 - forest) / (1 - f[3]) if f[3] < 1 else f
            f[3] = forest
            results.append({"site": sid, "epoch": epoch, "model": "paper", **dict(zip(CLASS_KEYS, f / f.sum()))})
    sites.to_csv(tmp_path / "sites.csv", index=False)
    pd.DataFrame(results).to_csv(tmp_path / "results.csv.gz", index=False, compression="gzip")
    meta = {"epochs": ["eox-2016", "eox-2025"], "epoch_labels": {"eox-2016": "2016", "eox-2025": "2025"}}
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    monkeypatch.setattr(atlas, "ATLAS_DIR", tmp_path)
    atlas.load.cache_clear()
    yield sites
    atlas.load.cache_clear()


def test_estimates_trend_change(fake_atlas):
    nepal = boundaries.country()
    assert atlas.available()
    assert atlas.epochs("paper") == ["eox-2016", "eox-2025"]
    est = atlas.estimate(nepal, "eox-2016", "paper")
    assert est["share"].sum() == pytest.approx(100, abs=1e-6)
    forest = est.set_index("key").loc["forest"]
    assert forest.share == pytest.approx(50, abs=0.5)
    trend = atlas.trend(nepal, "paper")
    assert set(trend.label) == {"2016", "2025"}
    ch = atlas.change(nepal, "paper", "eox-2016", "eox-2025").set_index("key")
    assert ch.loc["forest", "change_pp"] == pytest.approx(5, abs=0.5)
    assert ch.loc["forest", "p_value"] < 0.01


def test_district_table(fake_atlas):
    t = atlas.district_table("paper", "eox-2025")
    assert len(t) == 12 and (t.n == 6).all()
    assert t.forest.between(54, 56).all()
    one = boundaries.get(t.district_code.iloc[0])
    est = atlas.estimate(one, "eox-2025", "paper")
    assert est.set_index("key").loc["forest", "share"] == pytest.approx(55, abs=0.5)
