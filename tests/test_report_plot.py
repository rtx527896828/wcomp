"""scripts/plot_report.py on a tiny synthetic results tree (no model, no GPU): the method -> marker kind mapping, the seed
means and seed bars, the Phase 1 RTN context with the dPPL cap, and that the CLI renders both figures for two models."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

import plot_report as pr  # noqa: E402
import probe_fixtures as fx  # noqa: E402

FP16 = 17.5


def row(model: str, method: str, bits: float, ppl: float, kl: float, seed: int = 0, huff: float | None = None) -> dict:
    r = fx._row(model, method, bits, ppl, kl, seed=seed, calib_seed=seed if "gptq" in method else None)
    r["bits_blocks_huff"] = round(bits - 0.45, 4) if huff is None else huff
    return r


def phase3_rows(model: str) -> list[dict]:
    rows = [fx._row(model, "fp16", 16.0, FP16, 0.0, note="bf16, not quantized")]
    for b, kl, ppl in ((4, 0.07, 18.0), (3, 0.30, 23.4), (2, 1.85, 120.0)):
        for s, f in ((0, 1.0), (1, 1.04)):
            rows.append(row(model, f"gptq32-sym-g128-{b}", b + 0.125, ppl * f, kl * f, seed=s))
            rows.append(row(model, f"had-gptq32-sym-g128-{b}", b + 0.125, ppl * 0.9 * f, kl * 0.7 * f, seed=s, huff=b + 0.125 - 0.41))
    rows += [row(model, "rnd-gptq32-sym-g128-3", 3.125, 21.0, 0.21),
             row(model, "eig-gptq32-sym-g128-3", 17.8923, 29.7, 0.49),
             row(model, "lloyd-gptq32-g128-3", 3.125, 24.3, 0.34, seed=0),
             row(model, "lloyd-gptq32-g128-3", 3.125, 24.0, 0.36, seed=1),
             row(model, "alloc-type-gptq32-avg3", 3.1246, 24.6, 0.32),
             row(model, "alloc-tensor-gptq32-avg3", 3.125, 22.6, 0.24),
             row(model, "alloc-type-rtn-avg3", 3.1246, 48.0, 0.95),
             row(model, "alloc-type-shuf1-gptq32-avg3", 3.11, 30.0, 0.54),
             row(model, "rowalloc-fisher-gptq32-avg3", 3.1266, 23.9, 0.27),
             row(model, "rowalloc-fisher-rtn-avg4", 4.1266, 21.1, 0.175),
             row(model, "rowalloc-shuf1-rtn-avg4", 4.1266, 24.2, 0.266),
             row(model, "combo-gptq32-avg3", 3.1258, 22.6, 0.266, seed=0, huff=2.712),
             row(model, "combo-gptq32-avg3", 3.1258, 23.2, 0.274, seed=1, huff=2.712),
             row(model, "rtn-sym-g128-4", 4.125, 22.67, 0.205)]
    return rows


def phase1_rows(model: str) -> list[dict]:
    return [fx._row(model, "fp16", 16.0, FP16, 0.0, note="bf16, not quantized"),
            fx._row(model, "rtn-sym-g128-8", 8.125, 17.49, 0.0015),
            fx._row(model, "rtn-sym-g128-4", 4.125, 22.67, 0.205),
            fx._row(model, "rtn-sym-g128-3", 3.125, 84.8, 1.5),
            fx._row(model, "rtn-sym-g128-2", 2.125, 1.0e6, 11.06)]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


@pytest.fixture()
def tree(tmp_path: Path) -> dict[str, Path]:
    for m in ("a", "b"):
        write_jsonl(tmp_path / "phase3" / m / "runs.jsonl", phase3_rows(m))
        write_jsonl(tmp_path / "p1" / m / "runs.jsonl", phase1_rows(m))
    return {"rdir": tmp_path / "phase3", "p1dir": tmp_path / "p1", "root": tmp_path}


def test_kind_of_maps_every_phase3_family():
    assert pr.kind_of("gptq32-sym-g128-3") is None and pr.kind_of("had-gptq32-sym-g128-3") is None  # drawn as lines
    assert pr.kind_of("rnd-gptq32-sym-g128-3") == "rnd" and pr.kind_of("eig-gptq32-sym-g128-3") == "eig"
    assert pr.kind_of("lloyd-gptq32-g128-2") == "lloyd" and pr.kind_of("nfq-gptq32-g128-4") == "nfq"
    assert pr.kind_of("alloc-type-gptq32-avg2.5") == "type" and pr.kind_of("alloc-tensor-gptq32-avg3") == "tensor"
    assert pr.kind_of("alloc-type-rtn-avg3") == "type-rtn" and pr.kind_of("alloc-type-shuf2-gptq32-avg3") == "shuf"
    assert pr.kind_of("rowalloc-fisher-gptq32-avg4") == "fisher" and pr.kind_of("rowalloc-fisher-rtn-avg4") == "fisher-rtn"
    assert pr.kind_of("rowalloc-shuf1-rtn-avg4") == "shuf-rtn" and pr.kind_of("combo-gptq32-avg3") == "combo"
    assert pr.kind_of("rtn-sym-g128-4") is None and pr.kind_of("fp16") is None
    assert set(pr.KIND_STYLE) >= {pr.kind_of(r["method"]) for r in phase3_rows("a")} - {None}


def test_seed_means_and_bars():
    rows = phase3_rows("a")
    ref = pr.seed_curve(rows, "gptq32-sym-g128-", FP16, "kl")
    assert [p["bits"] for p in ref] == [2.125, 3.125, 4.125] and all(p["n"] == 2 for p in ref)
    p3b = ref[1]
    assert p3b["y"] == pytest.approx(0.30 * 1.02) and p3b["lo"] == pytest.approx(0.30) and p3b["hi"] == pytest.approx(0.312)
    assert p3b["huff"] == pytest.approx(3.125 - 0.45)
    pts, notes = pr.method_points(rows, FP16, "kl")
    by = {p["method"]: p for p in pts}
    assert by["lloyd-gptq32-g128-3"]["y"] == pytest.approx(0.35) and by["lloyd-gptq32-g128-3"]["n"] == 2
    assert by["combo-gptq32-avg3"]["huff"] == pytest.approx(2.712)
    assert "rtn-sym-g128-4" not in by and not notes  # the Phase 3 copy of the RTN row is context, not a method
    d = pr.method_points(rows, FP16, "dppl")[0]
    assert {p["method"]: p for p in d}["rnd-gptq32-sym-g128-3"]["y"] == pytest.approx(100 * (21.0 / FP16 - 1))


def test_rtn_context_skips_8_bit_and_caps_dppl():
    rtn = pr.rtn_curve(phase1_rows("a"), FP16, "kl")
    assert [p["bits"] for p in rtn] == [2.125, 3.125, 4.125] and not any(p["off"] for p in rtn)
    rtn_d = pr.rtn_curve(phase1_rows("a"), FP16, "dppl")
    assert [p["off"] for p in rtn_d] == [True, False, False]


def test_nonpositive_metric_is_reported_not_drawn():
    rows = [row("a", "alloc-type-gptq32-avg4", 4.12, FP16 - 0.1, 0.05)]
    pts, notes = pr.method_points(rows, FP16, "dppl")
    assert pts == [] and "not on the log axis" in notes[0]


def test_single_seed_methods_lists_only_drawn_one_seed_rows():
    rows = phase3_rows("a")
    one = pr.single_seed_methods(rows)
    assert "rnd-3" in one and "eig-3" in one and "alloc-type-shuf1-avg3" in one and "rowalloc-fisher-rtn-avg4" in one
    assert "lloyd-3" not in one and "combo-avg3" not in one and "gptq32-3" not in one and "had-3" not in one  # two seeds
    assert not any(m.startswith("rtn-sym") or m == "fp16" for m in one)  # context / unquantized rows are not drawn markers
    rows_1 = [r for r in rows if not (r["method"] == "had-gptq32-sym-g128-3" and r["seed"] == 1)]
    assert "had-3" in pr.single_seed_methods(rows_1)  # a one-seed curve point is listed too


def test_codebook_rows_get_a_huffman_twin():
    assert {"lloyd", "nf", "nfq", "combo"} <= set(pr.HUFF_KINDS)
    pts, _ = pr.method_points(phase3_rows("a"), FP16, "kl")
    lloyd = {p["method"]: p for p in pts}["lloyd-gptq32-g128-3"]
    assert lloyd["huff"] == pytest.approx(3.125 - 0.45)


def test_dppl_ylim_covers_every_panel(tree, monkeypatch):
    """sharey + a limit fixed while only the first panel was drawn clipped a lower point of a later panel."""
    rows_b = phase3_rows("b")
    for r in rows_b:
        if r["method"] == "gptq32-sym-g128-4" and r["seed"] == 1:
            r["ppl"] = FP16 * 1.003  # dPPL 0.3%, far below anything in panel "a" (min ~2.9%)
    write_jsonl(tree["rdir"] / "b" / "runs.jsonl", rows_b)
    figs = []
    monkeypatch.setattr(pr.plt, "close", lambda fig: figs.append(fig))
    pr.render(["a", "b"], tree["rdir"], tree["p1dir"], "dppl", tree["root"] / "x.png")
    for ax in figs[0].axes[:2]:
        lo, hi = ax.get_ylim()
        assert lo <= 0.3 and hi == pytest.approx(pr.DPPL_CAP * 1.6)


def test_cli_renders_both_figures(tree):
    out, out_ppl = tree["root"] / "rd.png", tree["root"] / "rd_ppl.png"
    cmd = [sys.executable, "-I", str(ROOT / "scripts" / "plot_report.py"), "--models", "a,b", "--results-dir", str(tree["rdir"]),
           "--phase1-dir", str(tree["p1dir"]), "--out", str(out), "--out-ppl", str(out_ppl)]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    assert out.exists() and out.stat().st_size > 50_000 and out_ppl.exists() and out_ppl.stat().st_size > 50_000
    assert "eig-gptq32-sym-g128-3 at 17.89 bits drawn at the right edge" in r.stderr
    assert "above the 2000% cap" in r.stderr


def test_cli_without_fp16_fails_clearly(tmp_path):
    write_jsonl(tmp_path / "phase3" / "a" / "runs.jsonl", [row("a", "gptq32-sym-g128-3", 3.125, 23.0, 0.3)])
    cmd = [sys.executable, "-I", str(ROOT / "scripts" / "plot_report.py"), "--models", "a", "--results-dir", str(tmp_path / "phase3"),
           "--phase1-dir", str(tmp_path / "p1"), "--out", str(tmp_path / "x.png"), "--no-ppl"]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    assert r.returncode != 0 and "fp16" in r.stderr
