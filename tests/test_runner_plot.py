"""Tests for scripts/run_baselines.py and scripts/plot_rd.py on synthetic runs.jsonl rows (no model, CPU only)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import plot_rd  # noqa: E402
import run_baselines as rb  # noqa: E402
from wcomp.bits import BitAccount  # noqa: E402

DESIGN_FIELDS = {"model", "method", "seed", "bits_blocks", "bits_all", "ppl", "kl", "top1_agree", "n_tokens",
                 "kl_tokens", "time_s", "note", "git_or_code_version", "timestamp"}


def _row(method: str, ppl: float, note: str, seed: int = 0, calib_seed=None, version: str = "v1",
         ts: str = "2026-11-08T00:00:00+00:00", kl: float = 0.01, bits: float = 4.125) -> dict:
    return {"model": "m", "method": method, "seed": seed, "calib_seed": calib_seed, "bits_blocks": bits,
            "bits_all": 8.0, "ppl": ppl, "kl": kl, "top1_agree": 0.9, "n_tokens": 1000, "kl_tokens": 100,
            "time_s": 1.0, "note": note, "git_or_code_version": version, "timestamp": ts}


def _write(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return path


# ---------------------------------------------------------------- run_baselines: smoke marking / results dir


def test_smoke_markers_and_results_dir():
    assert rb.smoke_markers(None, 32, 128) == []
    assert rb.default_results_dir([]) == "results"
    assert rb.smoke_markers(8, 4, 16) == ["limit-windows 8", "kl-windows 4", "calib-n 16"]
    # a reduced calibration set alone is off protocol (DESIGN.md 4: 128 windows) and must land in results/smoke
    m = rb.smoke_markers(None, 32, 16)
    assert m == ["calib-n 16"]
    assert rb.default_results_dir(m) == "results/smoke"
    assert rb.default_results_dir(rb.smoke_markers(None, 4, 128)) == "results/smoke"
    assert rb.default_results_dir(rb.smoke_markers(8, 32, 128)) == "results/smoke"
    # the calib-n marker belongs to calibrated methods only: an RTN-only invocation with --calib-n 16 is on protocol
    assert rb.smoke_markers(None, 32, 16, calib_used=False) == []
    assert rb.smoke_markers(8, 4, 16, calib_used=False) == ["limit-windows 8", "kl-windows 4"]
    assert rb.default_results_dir(rb.smoke_markers(None, 32, 16, calib_used=False)) == "results"


def test_smoke_results_dir_redirects_explicit_dirs():
    """An explicit --results-dir never receives smoke rows: results/probe -> results/probe/smoke; a dir that already
    says smoke, or a full-protocol run, is left alone."""
    assert rb.smoke_results_dir("results/probe", ["limit-windows 8"]) == str(Path("results/probe") / "smoke")
    assert rb.smoke_results_dir("results/probe", []) == "results/probe"
    assert rb.smoke_results_dir("results/smoke/probe", ["limit-windows 8"]) == "results/smoke/probe"
    assert rb.smoke_results_dir("results/smoke", ["kl-windows 4"]) == "results/smoke"
    assert rb.smoke_results_dir(str(Path("D:/x/smoke/probe")), ["calib-n 16"]) == str(Path("D:/x/smoke/probe"))


def test_make_row_has_design_fields():
    lin = nn.Linear(256, 8, bias=False)
    acct = BitAccount()
    acct.add("layers.0.q", lin.weight.numel(), 4 * lin.weight.numel(), 16 * 8 * 2)
    metrics = {"ppl": 15.0, "nll": 2.7, "n_tokens": 100, "kl": 0.1, "top1_agree": 0.9, "kl_tokens": 50}
    row = rb.make_row("m", "rtn-sym-g128-4", 0, None, acct, lin, metrics, 1.234, 0.5,
                      ["calib-n 16", "rtn-sym-g128-4: f16 scale"], "abc/def", "torch x")
    assert DESIGN_FIELDS <= set(row)
    assert tuple(row) == rb.ROW_FIELDS
    assert row["bits_blocks"] == 4.125 and row["note"] == "calib-n 16; rtn-sym-g128-4: f16 scale"
    json.dumps(row, allow_nan=False)
    assert plot_rd.is_smoke(row) and plot_rd.smoke_tag(row["note"]) == "calib-n 16"


def test_env_and_code_version_strings():
    env = rb.env_version()
    assert env.startswith("torch ") and "transformers " in env and "kernels " in env
    v = rb.code_version()
    assert "/" in v and len(v.split("/")[1]) == 10


def test_assert_disjoint_by_device_and_pointer():
    a = nn.Linear(8, 4)
    b = nn.Linear(8, 4)
    rb.assert_disjoint(a, b)
    with pytest.raises(AssertionError):
        rb.assert_disjoint(a, a)
    c = nn.Linear(8, 4)
    c.weight = a.weight  # shared storage on the same device
    with pytest.raises(AssertionError):
        rb.assert_disjoint(c, a)
    # Pointers are compared per device: a meta module (every data_ptr is 0) never collides with a CPU module,
    # just as the reference's CPU layers never collide with the quantized copy's CUDA layers.
    with torch.device("meta"):
        m = nn.Linear(8, 4)
    assert m.weight.data_ptr() == 0
    rb.assert_disjoint(m, a)
    rb.assert_disjoint(a, m)


# ---------------------------------------------------------------- plot_rd: smoke detection


@pytest.mark.parametrize("note,smoke", [
    ("bf16, not quantized", False),
    ("calib 128x2048 seed 0; gptq 3b sym g128 block128 damp0.01", False),
    ("calib 128x2048 seed 1; gptq 3b sym g128 block128 damp0.01", False),
    ("limit-windows 8; kl-windows 4; bf16, not quantized", True),
    ("kl-windows 4; rtn-sym-g128-4: f16 scale", True),
    ("calib-n 16; calib 16x2048 seed 0; gptq 4b sym g128 block128 damp0.01", True),
    ("calib 16x2048 seed 0; gptq 4b sym g128 block128 damp0.01", True),  # old row without the calib-n marker
])
def test_is_smoke(note, smoke):
    assert plot_rd.is_smoke({"note": note}) is smoke


def test_smoke_tag_normalizes_old_and_new_rows():
    new = "limit-windows 8; kl-windows 4; calib-n 16; calib 16x2048 seed 0; gptq 4b sym g128 block128 damp0.01"
    old = "limit-windows 8; kl-windows 4; calib 16x2048 seed 0; gptq 4b sym g128 block128 damp0.01"
    assert plot_rd.smoke_tag(new) == plot_rd.smoke_tag(old) == "calib-n 16; kl-windows 4; limit-windows 8"
    assert plot_rd.smoke_tag("calib 128x2048 seed 0; gptq 3b sym g128 block128 damp0.01") == ""


# ---------------------------------------------------------------- plot_rd: load_runs dedup / conflicts


def test_load_runs_ignores_smoke_unless_asked(tmp_path, capsys):
    p = _write(tmp_path / "runs.jsonl", [
        _row("fp16", 14.0, "bf16, not quantized", bits=16.0),
        _row("gptq-sym-g128-3", 15.0, "calib 128x2048 seed 0; gptq 3b", calib_seed=0),
        _row("gptq-sym-g128-3", 15.3, "calib 16x2048 seed 0; gptq 3b", calib_seed=0),  # reduced calibration
        _row("rtn-sym-g128-4", 15.5, "limit-windows 8; rtn"),
    ])
    runs = plot_rd.load_runs(p)
    assert {(r["method"], r["ppl"]) for r in runs} == {("fp16", 14.0), ("gptq-sym-g128-3", 15.0)}
    assert "ignored 2 smoke rows" in capsys.readouterr().err
    runs = plot_rd.load_runs(p, include_smoke=True)
    assert len(runs) == 4


def test_load_runs_conflict_keeps_latest_and_warns(tmp_path, capsys):
    """Same method from two code versions: the calib_seed field appeared in the second version and the note
    text changed, the keys still collide, the latest row wins and stderr lists both."""
    p = _write(tmp_path / "runs.jsonl", [
        _row("fp16", 14.0, "bf16, not quantized", bits=16.0, version="v1", ts="2026-11-08T01:00:00+00:00"),
        _row("fp16", 14.0, "bf16, not quantized", bits=16.0, version="v2", ts="2026-11-08T02:00:00+00:00"),
        _row("gptq-sym-g128-4", 15.948, "calib 128x2048 seed 0; gptq 4b sym g128 block128 damp0.01",
             version="v1", ts="2026-11-08T01:36:00+00:00"),
        _row("gptq-sym-g128-4", 16.248, "calib 128x2048 seed 0; gptq 4b sym g128 block128 damp0.01; rotation note",
             calib_seed=0, version="v2", ts="2026-11-08T02:02:00+00:00"),
        # an older timestamp appended later must not win
        _row("gptq-sym-g128-4", 15.0, "calib 128x2048 seed 0; gptq 4b sym g128 block128 damp0.01",
             calib_seed=0, version="v0", ts="2026-11-08T00:00:00+00:00"),
    ])
    runs = plot_rd.load_runs(p)
    by = {r["method"]: r for r in runs}
    assert len(runs) == 2
    assert by["gptq-sym-g128-4"]["ppl"] == 16.248 and by["gptq-sym-g128-4"]["git_or_code_version"] == "v2"
    err = capsys.readouterr().err
    assert "gptq-sym-g128-4" in err and "v1" in err and "v2" in err and "v0" in err and "keeping the latest" in err
    assert "fp16" in err  # the version change of fp16 is reported too


def test_load_runs_distinct_seeds_are_separate_points(tmp_path):
    p = _write(tmp_path / "runs.jsonl", [
        _row("gptq-sym-g128-3", 15.0, "calib 128x2048 seed 0; gptq", calib_seed=0),
        _row("gptq-sym-g128-3", 15.1, "calib 128x2048 seed 1; gptq", seed=1, calib_seed=1),
    ])
    assert len(plot_rd.load_runs(p)) == 2


# ---------------------------------------------------------------- plot_rd: baselines


def test_baselines_collapse_equal_fp16_rows_across_seeds():
    runs = [_row("fp16", 14.0, "bf16, not quantized", seed=0, bits=16.0),
            _row("fp16", 14.0, "bf16, not quantized", seed=1, bits=16.0),
            _row("gptq-sym-g128-3", 15.0, "calib 128x2048 seed 1; gptq", seed=1, calib_seed=1)]
    assert plot_rd.baselines(runs) == {"": 14.0}
    assert plot_rd.baseline_ppl(runs) == 14.0


def test_baselines_reject_disagreeing_fp16_rows():
    runs = [_row("fp16", 14.0, "bf16, not quantized", bits=16.0),
            _row("fp16", 14.5, "bf16, not quantized", seed=1, bits=16.0)]
    with pytest.raises(SystemExit):
        plot_rd.baselines(runs)
    with pytest.raises(SystemExit):
        plot_rd.baselines([_row("rtn-sym-g128-4", 15.0, "rtn")])


def test_eval_tag_only_tracks_the_evaluated_windows():
    assert plot_rd.eval_tag("limit-windows 8; kl-windows 4; calib-n 16; calib 16x2048 seed 0; gptq") == "limit-windows 8"
    assert plot_rd.eval_tag("limit-windows 8; kl-windows 4; bf16, not quantized") == "limit-windows 8"
    assert plot_rd.eval_tag("calib-n 16; calib 16x2048 seed 0; gptq") == ""  # same windows as a full run
    assert plot_rd.eval_tag("kl-windows 4; bf16, not quantized") == ""


def test_baselines_match_calib_n_rows_to_the_fp16_of_their_windows():
    """The real smoke file: fp16 'limit-windows 8; kl-windows 4' and gptq '...; calib 16x2048 seed 0' share the
    eval windows, so the gptq row must be differenced against that fp16 row, not skipped."""
    runs = [_row("fp16", 14.947, "limit-windows 8; kl-windows 4; bf16, not quantized", bits=16.0),
            _row("gptq-sym-g128-4", 16.248, "limit-windows 8; kl-windows 4; calib 16x2048 seed 0; gptq 4b", calib_seed=0)]
    b = plot_rd.baselines(runs)
    assert b == {"limit-windows 8": 14.947}
    assert plot_rd.eval_tag(runs[1]["note"]) in b


def test_baselines_are_per_window_subset():
    runs = [_row("fp16", 14.0, "bf16, not quantized", bits=16.0),
            _row("fp16", 12.593, "limit-windows 1; bf16, not quantized", bits=16.0),
            _row("rtn-sym-g128-4", 13.2, "limit-windows 1; rtn")]
    b = plot_rd.baselines(runs)
    assert b == {"": 14.0, "limit-windows 1": 12.593}
    with pytest.raises(SystemExit):  # smoke-only file has no full baseline
        plot_rd.baseline_ppl(runs[1:])
    assert plot_rd.baselines(runs[1:]) == {"limit-windows 1": 12.593}


# ---------------------------------------------------------------- plot_rd: the script end to end


def _run_plot(rdir: Path, *extra: str) -> subprocess.CompletedProcess:
    cmd = [sys.executable, "-I", str(ROOT / "scripts" / "plot_rd.py"), "--model", "m", "--results-dir", str(rdir), *extra]
    return subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)


def test_plot_script_smoke_row_uses_its_own_fp16(tmp_path):
    rdir = tmp_path / "results"
    _write(rdir / "m" / "runs.jsonl", [
        _row("fp16", 14.0, "bf16, not quantized", bits=16.0, kl=0.0),
        _row("rtn-sym-g128-4", 15.0, "rtn-sym-g128-4: f16 scale"),
        _row("fp16", 12.593, "limit-windows 1; bf16, not quantized", bits=16.0, kl=0.0),
        _row("rtn-sym-g128-4", 13.2, "limit-windows 1; rtn-sym-g128-4: f16 scale"),  # +0.6 on its own windows
        _row("gptq-sym-g128-3", 15.3, "limit-windows 2; calib 128x2048 seed 0; gptq", calib_seed=0),  # no fp16 row
        _row("had-fp16", 14.01, "bf16, not quantized; input rotation hadamard(1024)", bits=16.0, kl=0.001),
    ])
    r = _run_plot(rdir)
    assert r.returncode == 0, r.stderr
    assert (rdir / "rd_m.png").exists()
    assert "ignored 3 smoke rows" in r.stderr and "hollow" not in r.stderr
    r = _run_plot(rdir, "--include-smoke", "--out", str(tmp_path / "inc.png"))
    assert r.returncode == 0, r.stderr
    assert "hollow" not in r.stderr  # 13.2 - 12.593 > 0: differenced against its own fp16 row
    assert "no fp16 row for these windows, skipped" in r.stderr and "gptq-sym-g128-3" in r.stderr
    assert "fp16 baselines per window subset" in r.stderr


def test_plot_script_seed1_fp16_does_not_break(tmp_path):
    rdir = tmp_path / "results"
    _write(rdir / "m" / "runs.jsonl", [
        _row("fp16", 14.0, "bf16, not quantized", bits=16.0, kl=0.0),
        _row("gptq-sym-g128-3", 15.0, "calib 128x2048 seed 0; gptq", calib_seed=0),
        _row("fp16", 14.0, "bf16, not quantized", seed=1, bits=16.0, kl=0.0),  # DESIGN.md seed-1 GPTQ run
        _row("gptq-sym-g128-3", 15.1, "calib 128x2048 seed 1; gptq", seed=1, calib_seed=1),
    ])
    r = _run_plot(rdir)
    assert r.returncode == 0, r.stderr


# ---------------------------------------------------------------- plot_rd: same settings, different result


def test_load_runs_same_settings_different_ppl_is_a_conflict(tmp_path, capsys):
    """Two rows with the same key, note and code version but another ppl (a non-reproducible rerun or a GPU
    fault) must be reported, not silently collapsed to the latest; byte-identical duplicates stay silent."""
    p = _write(tmp_path / "runs.jsonl", [
        _row("rtn-sym-g128-4", 19.0, "rtn-sym-g128-4: f16 scale", ts="2026-11-08T01:00:00+00:00"),
        _row("rtn-sym-g128-4", 25.0, "rtn-sym-g128-4: f16 scale", ts="2026-11-08T02:00:00+00:00"),
    ])
    runs = plot_rd.load_runs(p)
    assert [r["ppl"] for r in runs] == [25.0]
    err = capsys.readouterr().err
    assert "warning" in err and "rtn-sym-g128-4" in err and "ppl 19" in err and "ppl 25" in err
    # same ppl, different kl: also a conflict
    p = _write(tmp_path / "runs2.jsonl", [
        _row("rtn-sym-g128-4", 19.0, "rtn-sym-g128-4: f16 scale", kl=0.01),
        _row("rtn-sym-g128-4", 19.0, "rtn-sym-g128-4: f16 scale", kl=0.02),
    ])
    plot_rd.load_runs(p)
    assert "warning" in capsys.readouterr().err
    # identical duplicate: silent
    p = _write(tmp_path / "runs3.jsonl", [
        _row("rtn-sym-g128-4", 19.0, "rtn-sym-g128-4: f16 scale"),
        _row("rtn-sym-g128-4", 19.0, "rtn-sym-g128-4: f16 scale"),
    ])
    assert len(plot_rd.load_runs(p)) == 1
    assert "warning" not in capsys.readouterr().err
    assert plot_rd.rows_disagree(_row("a", 1.0, "n"), _row("a", 1.0, "n", version="v2"))
    assert not plot_rd.rows_disagree(_row("a", 1.0, "n"), _row("a", 1.0, "n", ts="2026-01-01T00:00:00+00:00"))


# ---------------------------------------------------------------- plot_rd: family colors / hollow points


def test_family_style_is_fixed_by_family_identity():
    gptq = plot_rd.FAMILIES.index("gptq-sym-g128")
    expected = (plot_rd.COLORS[gptq], plot_rd.MARKERS[gptq])
    assert plot_rd.family_style("gptq-sym-g128", ["rtn-sym-g128", "gptq-sym-g128"]) == expected
    assert plot_rd.family_style("gptq-sym-g128", plot_rd.FAMILIES) == expected
    assert plot_rd.family_style("gptq-sym-g128", ["gptq-sym-g128"]) == expected
    # families outside FAMILIES take the slots after it, in order, without colliding with a known family
    order = ["rtn-sym-g128", "new-a", "new-b"]
    a, b = plot_rd.family_style("new-a", order), plot_rd.family_style("new-b", order)
    assert a != b and a[0] not in plot_rd.COLORS[: len(plot_rd.FAMILIES)]


def test_line_ys_breaks_the_line_at_hollow_points():
    import math
    ys = plot_rd.line_ys([4.0, 0.65, 0.1], [False, True, False])
    assert ys[0] == 4.0 and math.isnan(ys[1]) and ys[2] == 0.1
    assert plot_rd.line_ys([1.0, 2.0], [False, False]) == [1.0, 2.0]


# ---------------------------------------------------------------- run_baselines: effective kl-windows, timing


def test_smoke_markers_name_the_effective_kl_windows():
    # --limit-windows 4 with the default --kl-windows 32: the KL runs over 4 windows and the note must say so
    assert rb.smoke_markers(4, 32, 128, n_windows=4) == ["limit-windows 4", "kl-windows 4"]
    assert rb.smoke_markers(8, 4, 128, n_windows=8) == ["limit-windows 8", "kl-windows 4"]
    assert rb.smoke_markers(None, 32, 128, n_windows=145) == []  # full protocol: no clamp, no marker
    assert rb.smoke_markers(None, 32, 128) == []


def test_timed_measures_only_the_call():
    import time

    def slow(x):
        time.sleep(0.05)
        return x * 2

    out, s = rb.timed(slow, 21)
    assert out == 42 and 0.04 <= s < 2.0
    _, s0 = rb.timed(lambda: None)
    assert s0 < 0.5  # a no-op method (fp16) must not be charged with a model load


# ---------------------------------------------------------------- Hessian path in the row identity


def test_run_key_and_merge_keep_the_f32_acceptance_row_apart(capsys):
    p1 = _row("gptq-sym-g128-3", 24.4482, "calib 128x2048 seed 0; gptq 3b sym g128 block128 damp0.01", calib_seed=0,
              ts="2026-11-08T03:42:30+00:00")
    acc = _row("gptq-sym-g128-3", 23.9, "calib 128x2048 seed 0; gptq 3b sym g128 block128 damp0.01 hessian-f32", calib_seed=0,
               ts="2026-11-08T12:00:00+00:00")
    f64 = _row("gptq-sym-g128-3", 24.4482, "calib 128x2048 seed 0; gptq 3b sym g128 block128 damp0.01 hessian-f64", calib_seed=0,
               ts="2026-11-08T12:30:00+00:00")
    assert plot_rd.run_key(p1) != plot_rd.run_key(acc) and plot_rd.run_key(p1) == plot_rd.run_key(f64)
    assert plot_rd.hessian_tag(acc["note"]) == "hessian-f32" and plot_rd.hessian_tag(p1["note"]) == ""
    merged = plot_rd.merge_runs([p1], [acc])
    assert sorted(r["ppl"] for r in merged) == [23.9, 24.4482]
    assert not plot_rd.default_seed(acc) and plot_rd.default_seed(p1)
    assert plot_rd.point_label(acc) == "3 H32" and plot_rd.point_label(p1) == "3"
    # a Phase 1 name with the same key is never replaced by an --extra row, and a disagreement is reported
    other = _row("gptq-sym-g128-3", 25.0, "calib 128x2048 seed 0; gptq 3b sym g128 block128 damp0.01 hessian-f64", calib_seed=0,
                 ts="2026-11-08T13:00:00+00:00")
    merged = plot_rd.merge_runs([p1], [other])
    assert len(merged) == 1 and merged[0]["ppl"] == 24.4482
    assert "disagree" in capsys.readouterr().err
    # a Phase 2 name takes the latest row, silently when identical
    a = _row("rtn-sym-g32-4", 18.0, "rtn", ts="2026-11-08T01:00:00+00:00")
    b = _row("rtn-sym-g32-4", 18.1, "rtn", ts="2026-11-08T02:00:00+00:00")
    assert plot_rd.merge_runs([a], [b])[0]["ppl"] == 18.1 and "disagree" in capsys.readouterr().err
    assert plot_rd.merge_runs([], [a, a]) == [a] and capsys.readouterr().err == ""


def test_hessian_choice_required_only_for_gptq_rows_into_the_probe_tree():
    assert rb.hessian_choice_required(["gptq-sym-g128-3"], "results/probe")
    assert rb.hessian_choice_required(["rtn-sym-g32-4", "had-gptq-sym-g128-3"], "results/probe")
    assert not rb.hessian_choice_required(["gptq-sym-g128-3"], None)  # Phase 1 runs.jsonl: the Phase 1 default path
    assert not rb.hessian_choice_required(["gptq-sym-g128-3"], "results")
    assert not rb.hessian_choice_required(["gptq-sym-g128-3"], "results/smoke/probe")
    assert not rb.hessian_choice_required(["rtn-sym-g128-4@only=down", "fp16"], "results/probe")


def test_planned_results_dir_decides_the_hessian_gate_before_loading():
    """A GPTQ smoke run aimed at results/probe lands in results/probe/smoke, where no Hessian flag is required; the
    gate must see that redirected path, not the raw --results-dir."""
    m = ["gptq-sym-g128-3"]
    assert rb.planned_results_dir(None, m, None, 32, 128) is None
    assert rb.planned_results_dir("results/probe", m, None, 32, 128) == "results/probe"
    assert rb.hessian_choice_required(m, rb.planned_results_dir("results/probe", m, None, 32, 128))
    smoke = rb.planned_results_dir("results/probe", m, 4, 4, 4)
    assert smoke == str(Path("results/probe") / "smoke") and not rb.hessian_choice_required(m, smoke)
    assert rb.planned_results_dir("results/probe", m, None, 32, 16) == str(Path("results/probe") / "smoke")  # calib-n alone
    assert rb.planned_results_dir("results/probe", ["rtn-sym-g32-4"], None, 32, 16) == "results/probe"  # calib-n is no RTN marker
    assert rb.planned_results_dir("results/smoke/probe", m, 4, 4, 4) == "results/smoke/probe"


def test_skip_keepsuper_refuses_a_missing_or_malformed_p7_json(tmp_path):
    with pytest.raises(SystemExit, match="cannot read the P7 JSON"):
        rb.skip_keepsuper(f"rtn-sym-g128-4+keepsuper={tmp_path / 'missing.json'}")
    bad = tmp_path / "bad.json"
    bad.write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit, match="cannot read the P7 JSON"):
        rb.skip_keepsuper(f"rtn-sym-g128-4+keepsuper={bad}")


def test_skip_keepsuper_without_spike_candidates(tmp_path, caplog):
    no_spike = tmp_path / "sw_top.json"
    no_spike.write_text(json.dumps({"records": [{"tensor": "model.layers.0.mlp.down_proj", "stat": {"row": 1, "col": 2}, "note": "top16; z"}]}), encoding="utf-8")
    spike = tmp_path / "sw_spike.json"
    spike.write_text(json.dumps({"records": [{"tensor": "model.layers.0.mlp.down_proj", "stat": {"row": 1, "col": 2}, "note": "spike+top16; z"}]}), encoding="utf-8")
    with caplog.at_level("WARNING"):
        assert rb.skip_keepsuper(f"rtn-sym-g128-4+keepsuper={no_spike}")
        assert not rb.skip_keepsuper(f"rtn-sym-g128-4+keepsuper={no_spike}", any_kind=True)
    assert "skipped" in caplog.text and "positive-amax clip" in caplog.text and "running anyway" in caplog.text
    assert not rb.skip_keepsuper(f"rtn-sym-g128-4+keepsuper={spike}")
    assert not rb.skip_keepsuper("rtn-sym-g128-4+keepcols=1") and not rb.skip_keepsuper("gptq-sym-g128-3")
