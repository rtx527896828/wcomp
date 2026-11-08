"""Phase 2 presentation tools on synthetic fixtures (tests/probe_fixtures.py): every figure renders, the structure
table has the expected rows, readers tolerate missing files / stats, plot_rd --extra draws only full-model rows."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

import plot_probe  # noqa: E402
import plot_rd  # noqa: E402
import probe_fixtures as fx  # noqa: E402
import probe_io as io  # noqa: E402
import structure_table as st  # noqa: E402

MODELS = ["qwen3.5-0.8b", "qwen3.5-2b", "minicpm5-2b"]


@pytest.fixture(scope="module")
def fixtures(tmp_path_factory) -> Path:
    rdir = tmp_path_factory.mktemp("results")
    for m in MODELS:
        fx.write_fixtures(rdir, m)
    return rdir


# ---------------------------------------------------------------- fixtures follow the contract schema


def test_fixture_schema(fixtures):
    for m in MODELS:
        for probe in io.PROBES:
            doc = json.loads((fixtures / "probe" / m / f"{probe}.json").read_text(encoding="utf-8"))
            assert set(doc) >= {"model", "probe", "code_version", "timestamp", "params", "records", "aggregate"}
            assert doc["model"] == m and doc["probe"] == probe
            for r in doc["records"]:
                assert set(r) >= {"tensor", "layer", "type", "stat", "control", "gain_bits", "gain_bits_control", "note"}
                if r["type"] in io.TYPES:
                    assert r["type"] == _weight_type(r["tensor"])
        rows = [json.loads(l) for l in (fixtures / "probe" / m / "ablation.jsonl").read_text(encoding="utf-8").splitlines()]
        from run_baselines import ROW_FIELDS

        assert all(tuple(r) == ROW_FIELDS for r in rows)
        assert any("@only=" in r["method"] for r in rows) and any("+keepcols=" in r["method"] for r in rows)


def _weight_type(name: str) -> str:
    from wcomp.models import weight_type

    return weight_type(name)


def test_fixture_layer_kinds_match_models():
    assert fx.layer_kinds("qwen3.5-0.8b").count("attn") == 6 and len(fx.layer_kinds("qwen3.5-0.8b")) == 24
    assert set(fx.layer_kinds("minicpm5-2b")) == {"attn"} and len(fx.layer_kinds("minicpm5-2b")) == 42
    types = {t for _, _, t, _, _ in fx.all_linears("qwen3.5-0.8b")}
    assert types == set(io.TYPES)
    assert {t for _, _, t, _, _ in fx.all_linears("minicpm5-2b")} == {"q", "k", "v", "o", "gate", "up", "down"}


# ---------------------------------------------------------------- probe_io


def test_parse_method_and_full_model():
    p = io.parse_method("rtn-sym-g128-4@only=down")
    assert p["family"] == "rtn-sym-g128" and p["bits"] == 4 and p["qual"] == "only" and p["qual_value"] == "down"
    p = io.parse_method("rtn-sym-g128-3@layer=7")
    assert p["qual"] == "layer" and p["qual_value"] == "7"
    p = io.parse_method("rtn-sym-g128-4+keepcols=0.5")
    assert p["plus"] == "keepcols" and p["plus_value"] == "0.5" and p["qual"] is None
    p = io.parse_method("rtn-sym-g128-4+keepsuper=results/probe/m/superweight.json")
    assert p["plus"] == "keepsuper" and p["plus_value"].endswith("superweight.json")
    assert io.parse_method("rtn-sym-g32-4")["group"] == 32
    assert io.parse_method("had-gptq-sym-g128-3")["family"] == "had-gptq-sym-g128"
    assert io.parse_method("fp16") is None and io.parse_method("had-fp16") is None
    assert io.is_full_model("rtn-sym-g128-4+keepcols=1") and not io.is_full_model("rtn-sym-g128-4@except=q")


def test_aggregate_is_numel_weighted_and_skips_missing():
    recs = [{"type": "q", "layer": 0, "numel": 1, "stat": {"a": 1.0}},
            {"type": "q", "layer": 1, "numel": 3, "stat": {"a": 5.0}},
            {"type": "q", "layer": 2, "numel": 100, "stat": {}},  # missing stat: skipped, not zero
            {"type": "q", "layer": 3, "numel": 100, "stat": {"a": float("nan")}}]
    assert io.aggregate(recs, "stat", "a") == pytest.approx(4.0)
    assert io.aggregate(recs, "stat", "b") is None
    assert io.per_layer(recs, "stat", "a") == [(0, 1.0), (1, 5.0)]
    assert io.types_present([{"type": "down"}, {"type": "embed"}, {"type": "q"}]) == ["q", "down", "embed"]


def test_load_probe_tolerates_missing_and_broken(tmp_path, capsys):
    assert io.load_probe(tmp_path / "nope.json") is None
    (tmp_path / "bad.json").write_text("{not json", encoding="utf-8")
    assert io.load_probe(tmp_path / "bad.json") is None
    (tmp_path / "norec.json").write_text(json.dumps({"records": 3}), encoding="utf-8")
    assert io.load_probe(tmp_path / "norec.json") is None
    err = capsys.readouterr().err
    assert "nope.json" in err and "bad.json" in err and "norec.json" in err
    assert io.load_rows(tmp_path / "none.jsonl") == []


def test_load_rows_drops_smoke_rows_unless_asked(tmp_path, capsys):
    """A later smoke row of the same method must not replace the full-protocol row (latest-timestamp dedup)."""
    full = fx._row("m", "rtn-sym-g128-4@only=down", 13.9, 15.5, 0.03, note="bf16, not quantized; rtn-sym-g128-4: f16 scale", ts="2026-11-08T10:00:00+00:00")
    smoke = fx._row("m", "rtn-sym-g128-4@only=down", 13.9, 15.68, 0.031, note="limit-windows 2; kl-windows 2; rtn-sym-g128-4: f16 scale", ts="2026-11-08T11:00:00+00:00")
    old_calib = fx._row("m", "gptq-sym-g128-3", 3.125, 25.0, 0.3, calib_seed=0, note="calib 16x2048 seed 0; gptq 3b", ts="2026-11-08T11:00:00+00:00")
    p = tmp_path / "ablation.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in (full, smoke, old_calib)) + "\n", encoding="utf-8")
    rows = io.load_rows(p)
    assert len(rows) == 1 and rows[0]["ppl"] == 15.5
    assert io.ablation_only(rows, 4)["down"]["ppl"] == 15.5
    assert "2 smoke rows ignored" in capsys.readouterr().err
    rows = io.load_rows(p, include_smoke=True)  # the smoke tag is part of the key: the full row stays
    assert len(rows) == 3 and {r["ppl"] for r in rows} == {15.5, 15.68, 25.0}
    assert io.is_smoke(smoke) and io.is_smoke(old_calib) and not io.is_smoke(full)
    assert io.smoke_tag(smoke["note"]) == "kl-windows 2; limit-windows 2"


# ---------------------------------------------------------------- plot_probe: every figure renders


@pytest.mark.parametrize("model", MODELS)
def test_all_figures_render(fixtures, model, tmp_path):
    out = plot_probe.render(fixtures, model, out_dir=tmp_path)
    assert set(out) == set(plot_probe.FIGURES)
    for name, path in out.items():
        assert path is not None and path.exists() and path.stat().st_size > 10_000, name
        assert path.name == f"probe_{model}_{name}.png"


def test_figures_skip_missing_probe_files(tmp_path, capsys):
    """Only two probe files and no ablation: those two render, the rest return None, nothing raises."""
    fx.write_fixtures(tmp_path, "qwen3.5-0.8b", probes=["marginal", "acts"], with_ablation=False, with_runs=False)
    out = plot_probe.render(tmp_path, "qwen3.5-0.8b", out_dir=tmp_path)
    assert out["marginal"] is not None and out["acts"] is not None
    assert all(out[k] is None for k in out if k not in ("marginal", "acts"))
    err = capsys.readouterr().err
    assert "lowrank.json (missing)" in err and "ablation" in err


def test_figures_tolerate_missing_stats(tmp_path, capsys):
    """Stats renamed away: the figures still render with 'no data' panels and the skips are reported."""
    paths = fx.write_fixtures(tmp_path, "qwen3.5-0.8b", probes=["lowrank", "outliers", "sensitivity"], with_ablation=False)
    for probe in ("lowrank", "outliers", "sensitivity"):
        doc = json.loads(paths[probe].read_text(encoding="utf-8"))
        for r in doc["records"]:
            r["stat"] = {f"renamed_{k}": v for k, v in r["stat"].items()}
            r["control"] = {}
        doc["aggregate"] = {}
        doc.pop("global", None)  # the model-level allocation lives here
        paths[probe].write_text(json.dumps(doc), encoding="utf-8")
    out = plot_probe.render(tmp_path, "qwen3.5-0.8b", ["lowrank", "outliers", "sensitivity"], out_dir=tmp_path)
    assert all(p is not None and p.exists() for p in out.values())
    err = capsys.readouterr().err
    assert "skipped:" in err and "energy@" in err and "alloc_" in err


def test_acts_lorenz_with_explicit_curve_and_without_points(tmp_path):
    paths = fx.write_fixtures(tmp_path, "minicpm5-2b", probes=["acts"], with_ablation=False, with_runs=False)
    doc = json.loads(paths["acts"].read_text(encoding="utf-8"))
    for r in doc["records"]:
        if r["layer"] % 2:  # an explicit Lorenz curve is used when a probe stores one
            r["stat"]["lorenz_x"], r["stat"]["lorenz_y"] = [0, 0.1, 0.5, 1], [0, 0.4, 0.8, 1]
        else:  # otherwise the three stored points; with none of them the layer is skipped, not fatal
            r["stat"].pop("frac_channels_50")
            r["stat"].pop("energy_share_top20pct")
    paths["acts"].write_text(json.dumps(doc), encoding="utf-8")
    out = plot_probe.render(tmp_path, "minicpm5-2b", ["acts"], out_dir=tmp_path)
    assert out["acts"].exists()


def test_val_aliases_and_control_seeds():
    rec = {"stat": {"kurtosis_excess": 2.0, "energy_f0.1": 0.5, "H_idx_b4": 3.5}, "control": {"seed0_energy_r10pct": 0.2, "seed1_energy_r10pct": 0.25,
           "gain_bits_best_seed0": -0.1, "gain_bits_best_seed1": -0.12}, "gain_bits_control": -0.11, "numel": 10}
    assert io.val(rec, "stat", "kurtosis") == 2.0 and io.val(rec, "stat", "H_4") == 3.5
    assert io.val(rec, "stat", "energy@0.1") == 0.5
    assert io.val(rec, "control", "energy@0.1") == 0.2 and io.val(rec, "control_seed1", "energy@0.1") == 0.25
    assert io.val(rec, "stat", "nothing") is None and io.val({}, "stat", "kurtosis") is None
    assert io.control_seed_gains([rec]) == (-0.1, -0.12) and io.control_spread([rec]) == pytest.approx(0.02)
    rec2 = {"control": {"seed0_best_gain": 0.3, "seed1_best_gain": 0.1}, "numel": 1}
    assert io.control_spread([rec2]) == pytest.approx(0.2)
    assert io.control_spread([{"gain_bits_control": 0.1, "numel": 1}]) is None


def test_plot_probe_cli(fixtures, tmp_path):
    cmd = [sys.executable, "-I", str(ROOT / "scripts" / "plot_probe.py"), "--model", "qwen3.5-2b", "--results-dir", str(fixtures),
           "--probes", "ablation,marginal", "--out-dir", str(tmp_path)]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "probe_qwen3.5-2b_ablation.png").exists() and (tmp_path / "probe_qwen3.5-2b_marginal.png").exists()
    r = subprocess.run(cmd[:-4] + ["--probes", "nonsense"], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode != 0 and "unknown figure" in r.stderr


# ---------------------------------------------------------------- structure_table


def test_structure_table_rows(fixtures, tmp_path):
    text = st.build(fixtures, MODELS)
    out = tmp_path / "structure_table.md"
    out.write_text(text, encoding="utf-8")
    lines = text.splitlines()
    for m in MODELS:
        assert f"## {m}" in lines
        n_types = len({t for _, _, t, _, _ in fx.all_linears(m)})
        rows = [l for l in lines if l.startswith(f"| {m} | ")]
        # one row per (probe, type) for the eight tabulated probes (+ "embed" rows in marginal and lowrank)
        for probe in st.KEY_STATS:
            prow = [l for l in rows if l.startswith(f"| {m} | {probe} | ")]
            if probe in ("marginal", "lowrank"):
                assert len(prow) == n_types + 1, (m, probe, len(prow))
            elif probe == "acts":
                assert len(prow) == (3 if m.startswith("qwen") else 2), (m, probe)
            else:
                assert len(prow) == n_types, (m, probe, len(prow))
        # P0 summary rows for 4 and 3 bits
        assert any(l.startswith("| 4 | ") for l in lines) and any(l.startswith("| 3 | ") for l in lines)
        # P7 candidates: two on Qwen, one on MiniCPM5
        sw = [l for l in lines if l.startswith(f"| {m} | model.layers.") and "down_proj" in l]
        assert len(sw) == (2 if m.startswith("qwen") else 1)
        assert any("super weight" in l for l in sw)
    # verdict column carries the contract's rule words
    assert "pass" in text and "fail" in text and "stable" in text or "noise" in text
    assert "Spearman(s_t per type, rtn-sym-g128-4@only KL)" in text
    # the P6 allocation saving is printed only next to the Spearman, with its floored fraction and guarded variants
    assert "allocation saving, tensor granularity: 0.420 bits/param with 0.0% of the weights floored" in text
    assert "s_t is in absolute output units" in text
    # the pre-registered H-P3 / H-P0 group-size figures are computed, not read off a figure
    assert "+keepcols=1 at 4 bit recovers 35% of the RTN KL" in text and "H-P3 met (> 25 %)" in text
    assert "KL(g128) / KL(g32) at 4 bit: 2.22" in text and "H-P0 group size met (> 2)" in text
    assert "best aggregated r = " in text and "gain_bits = mean of per-record maxima" in text
    assert "## Skipped" not in text


def test_structure_table_verdict_rule():
    assert st.shuffle_verdict(0.10, 0.05, 0.01) == "pass"
    assert st.shuffle_verdict(0.10, 0.09, None) == "fail (1 seed)"  # 0.01 < 0.02 floor
    assert st.shuffle_verdict(0.10, 0.05, 0.08) == "fail"  # 0.05 > 0.02 but below the seed spread
    assert st.shuffle_verdict(0.05, 0.10, 0.01) == "fail (below control)"
    assert st.shuffle_verdict(None, 0.1, None) == "n/a"
    assert st.shuffle_verdict(0.03, 0.0, None) == "pass (1 seed)"


def test_structure_table_missing_files(tmp_path):
    fx.write_fixtures(tmp_path, "qwen3.5-0.8b", probes=["marginal"], with_ablation=False, with_runs=False)
    text = st.build(tmp_path, ["qwen3.5-0.8b", "qwen3.5-2b"])
    assert "## Skipped" in text and "qwen3.5-0.8b: lowrank.json missing" in text and "qwen3.5-2b: marginal.json missing" in text
    assert "_no @only rows in ablation.jsonl_" in text
    assert "| qwen3.5-0.8b | marginal | q |" in text


def test_structure_table_cli(fixtures, tmp_path):
    cmd = [sys.executable, "-I", str(ROOT / "scripts" / "structure_table.py"), "--models", "minicpm5-2b", "--results-dir", str(fixtures),
           "--out", str(tmp_path / "t.md")]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "t.md").exists() and "minicpm5-2b" in (tmp_path / "t.md").read_text(encoding="utf-8")


# ---------------------------------------------------------------- plot_rd --extra


def test_run_family_and_variants():
    assert plot_rd.run_family("rtn-sym-g128-4") == ("rtn-sym-g128", 4)  # unchanged Phase 1 behaviour
    assert plot_rd.run_family("fp16") == ("fp16", None)
    assert plot_rd.run_family("rtn-sym-g128-4+keepcols=1") == ("rtn-sym-g128+keepcols", 4)
    assert plot_rd.run_family("rtn-sym-g128-3+keepsuper=results/x.json") == ("rtn-sym-g128+keepsuper", 3)
    assert plot_rd.run_family("rtn-sym-g32-4") == ("rtn-sym-g32", 4)
    assert plot_rd.variant_of("rtn-sym-g32") == ("rtn-sym-g128", "g32")
    assert plot_rd.variant_of("rtn-sym-g128+keepcols") == ("rtn-sym-g128", "+keepcols")
    assert plot_rd.variant_of("rtn-sym-g128") is None and plot_rd.variant_of("gptq-sym-g128") is None
    assert plot_rd.variant_of("new-family-g128") is None
    assert plot_rd.is_full_model("rtn-sym-g128-4+keepcols=1") and not plot_rd.is_full_model("rtn-sym-g128-4@only=q")
    assert plot_rd.point_label({"method": "rtn-sym-g128-4+keepcols=0.5", "seed": 0}) == "4 p0.5"
    assert plot_rd.point_label({"method": "rtn-sym-g128-4", "seed": 0}) == "4"


def _run_rd(rdir: Path, *extra: str) -> subprocess.CompletedProcess:
    cmd = [sys.executable, "-I", str(ROOT / "scripts" / "plot_rd.py"), "--model", "qwen3.5-0.8b", "--results-dir", str(rdir), *extra]
    return subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)


def test_plot_rd_extra_draws_only_full_model_rows(fixtures, tmp_path):
    abl = fixtures / "probe" / "qwen3.5-0.8b" / "ablation.jsonl"
    r = _run_rd(fixtures, "--extra", str(abl), "--out", str(tmp_path / "rd.png"))
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "rd.png").exists()
    assert "@only / @except / @layer rows are not full-model points, skipped" in r.stderr
    # the merged run list contains the variant families and no partial rows
    runs = plot_rd.merge_runs(plot_rd.load_runs(fixtures / "qwen3.5-0.8b" / "runs.jsonl"),
                              [x for x in plot_rd.load_runs(abl) if plot_rd.is_full_model(x["method"])])
    fams = {plot_rd.run_family(x["method"])[0] for x in runs}
    assert {"rtn-sym-g32", "rtn-sym-g64", "rtn-sym-g128+keepcols", "rtn-sym-g128+keepsuper", "had-gptq-sym-g128"} <= fams
    assert not any("@" in x["method"] for x in runs)


def test_plot_rd_without_extra_is_unchanged(fixtures, tmp_path):
    """Same primary file, with and without an --extra that holds only partial-model rows: identical bytes."""
    abl = fixtures / "probe" / "qwen3.5-0.8b" / "ablation.jsonl"
    partial = tmp_path / "partial.jsonl"
    partial.write_text("".join(l for l in abl.read_text(encoding="utf-8").splitlines(keepends=True) if "@" in l), encoding="utf-8")
    a = _run_rd(fixtures, "--out", str(tmp_path / "a.png"))
    b = _run_rd(fixtures, "--extra", str(partial), "--out", str(tmp_path / "b.png"))
    assert a.returncode == 0 and b.returncode == 0, a.stderr + b.stderr
    # the title and footer mention --extra, so compare the plotted data instead of the bytes: no new family appears
    assert "no fp16 run" not in b.stderr
    runs_a = plot_rd.load_runs(fixtures / "qwen3.5-0.8b" / "runs.jsonl")
    runs_b = plot_rd.merge_runs(runs_a, [x for x in plot_rd.load_runs(partial) if plot_rd.is_full_model(x["method"])])
    assert {x["method"] for x in runs_a} == {x["method"] for x in runs_b}


# ---------------------------------------------------------------- probe_io: row identity, merge, Jaccard per type


def test_load_rows_keys_on_smoke_and_hessian_tags(tmp_path, capsys):
    """With include_smoke a later smoke row no longer shadows the full-protocol row; an f32-Hessian GPTQ row is a
    separate identity from the f64 one (same method / seed / calib_seed)."""
    full = fx._row("m", "rtn-sym-g128-4@only=down", 13.9, 24.4482, 0.03, note="rtn-sym-g128-4: f16 scale", ts="2026-11-08T10:00:00+00:00")
    smoke = fx._row("m", "rtn-sym-g128-4@only=down", 13.9, 20.5, 0.031, note="limit-windows 4; rtn-sym-g128-4: f16 scale", ts="2026-11-08T11:00:00+00:00")
    g64 = fx._row("m", "gptq-sym-g128-3", 3.125, 24.4482, 0.3131, calib_seed=0, note="calib 128x2048 seed 0; gptq 3b hessian-f64", ts="2026-11-08T03:00:00+00:00")
    g32 = fx._row("m", "gptq-sym-g128-3", 3.125, 23.9, 0.31, calib_seed=0, note="calib 128x2048 seed 0; gptq 3b hessian-f32", ts="2026-11-08T12:00:00+00:00")
    p = tmp_path / "ablation.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in (full, smoke, g64, g32)) + "\n", encoding="utf-8")
    rows = io.load_rows(p, include_smoke=True)
    assert sorted(r["ppl"] for r in rows) == [20.5, 23.9, 24.4482, 24.4482]
    assert io.hessian_tag(g32["note"]) == "hessian-f32" and io.hessian_tag(g64["note"]) == "" and io.hessian_tag("") == ""
    assert io.row_key(g32) != io.row_key(g64) and io.row_key(full) != io.row_key(smoke)
    rows = io.load_rows(p)
    assert sorted(r["ppl"] for r in rows) == [23.9, 24.4482, 24.4482]


def test_is_phase1_name_agrees_with_the_method_registry():
    from wcomp.methods import METHODS, PHASE1_METHODS

    for name in PHASE1_METHODS:
        assert io.is_phase1_name(name), name
    for name in METHODS:  # the Phase 3 registry entries (gptq32, rotation variants) are not Phase 1 names
        assert io.is_phase1_name(name) == (name in PHASE1_METHODS), name
    for name in ("rtn-sym-g32-4", "rtn-sym-g128-4@only=down", "rtn-sym-g128-4+keepcols=1", "rtn-sym-g128-3@layer=5", "gptq-sym-g64-3"):
        assert not io.is_phase1_name(name), name


def test_merge_rows_keeps_phase1_names_from_runs_and_reports_disagreement(capsys):
    """structure_table, plot_probe and plot_rd --extra share this merge: a Phase 1 method name comes from runs.jsonl
    whatever ablation.jsonl holds (reported when they differ); other keys take the latest timestamp."""
    runs = [fx._row("m", "fp16", 16.0, 14.9469, 0.0, note="bf16", ts="2026-11-08T01:00:00+00:00"),
            fx._row("m", "rtn-sym-g128-4", 4.125, 19.0075, 0.2, note="rtn", ts="2026-11-08T01:00:00+00:00"),
            fx._row("m", "gptq-sym-g128-3", 3.125, 24.4482, 0.3131, calib_seed=0, note="calib 128x2048 seed 0; gptq", ts="2026-11-08T03:00:00+00:00")]
    abl = [fx._row("m", "fp16", 16.0, 14.95, 0.0, note="bf16", ts="2026-11-08T07:00:00+00:00"),  # later, differs
           fx._row("m", "rtn-sym-g128-4", 4.125, 19.0075, 0.2, note="rtn", ts="2026-11-08T07:00:00+00:00"),  # later, identical
           fx._row("m", "rtn-sym-g128-4@only=down", 13.9, 15.5, 0.03, note="rtn", ts="2026-11-08T07:00:00+00:00"),
           fx._row("m", "rtn-sym-g128-4@only=down", 13.9, 15.6, 0.03, note="rtn", ts="2026-11-08T08:00:00+00:00"),
           fx._row("m", "gptq-sym-g128-3", 3.125, 23.9, 0.31, calib_seed=0, note="calib 128x2048 seed 0; gptq hessian-f32", ts="2026-11-08T09:00:00+00:00")]
    merged = io.merge_rows(runs, abl)
    by = {(r["method"], io.hessian_tag(r["note"])): r for r in merged}
    assert len(merged) == 5
    assert by[("fp16", "")]["ppl"] == 14.9469  # runs.jsonl wins for a Phase 1 name
    assert by[("gptq-sym-g128-3", "")]["ppl"] == 24.4482 and by[("gptq-sym-g128-3", "hessian-f32")]["ppl"] == 23.9
    assert by[("rtn-sym-g128-4@only=down", "")]["ppl"] == 15.6  # latest for a Phase 2 name
    err = capsys.readouterr().err
    assert "fp16" in err and "disagree" in err and "keeping runs.jsonl" in err
    assert io.find_row(merged, "fp16")["ppl"] == 14.9469


def test_jaccard_by_type_averages_every_series_of_a_type():
    doc = {"records": [
        {"tensor": "in_proj_ab:in_proj_a:cross-layer", "layer": -1, "type": "in_proj_ab", "stat": {"jaccard_mean": 0.0}, "control": {"jaccard_mean": 0.03}, "numel": 0},
        {"tensor": "in_proj_ab:in_proj_b:cross-layer", "layer": -1, "type": "in_proj_ab", "stat": {"jaccard_mean": 0.0667}, "control": {"jaccard_mean": 0.03}, "numel": 0},
        {"tensor": "q:q_proj:cross-layer", "layer": -1, "type": "q", "stat": {"jaccard_mean": 0.4}, "control": {"jaccard_mean": 0.02}, "numel": 0},
        {"tensor": "all-hidden-input:cross-layer", "layer": -1, "type": "all_hidden_input", "stat": {"jaccard_mean": 0.2}, "control": {"jaccard_mean": 0.01}, "numel": 0},
        {"tensor": "model.layers.0.self_attn.q_proj", "layer": 0, "type": "q", "stat": {"jaccard_mean": 9.0}, "control": {}, "numel": 100},
    ]}
    per = io.jaccard_by_type(doc)
    assert per["in_proj_ab"] == (pytest.approx(0.03335), pytest.approx(0.03)) and per["q"] == (0.4, 0.02)
    assert "all_hidden_input" not in per
    assert io.jaccard(doc, "in_proj_ab")[0] == pytest.approx(0.03335)
    assert io.jaccard(doc, "gate") == (0.2, 0.01) and io.jaccard(doc) == (0.2, 0.01)  # fallback to the pooled record
