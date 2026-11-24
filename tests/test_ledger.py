from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

import pytest

from weightbank import configs
from weightbank.configs import ROUNDS_ONE_AND_TWO, model_config, plain_words
from weightbank.ledger import OPEN, PATH, REFUTED, SEEDS, STEPS, SUPPORTED, Cell, Ledger, Row, RunRow, verdict
from weightbank.model import Drawn, Fixed, Layout, Own, Routed, Settled, Transformer

RUNS = Path("runs")


def fields() -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads(PATH.read_text())
    return loaded


def every_cell(ledger: Ledger) -> list[Cell]:
    values = [list(axis.tags) for axis in ledger.axes]
    return [dict(zip((axis.name for axis in ledger.axes), combination)) for combination in itertools.product(*values)]


def test_the_ledger_holds_what_the_run_files_hold(tmp_path: Path) -> None:
    ledger = Ledger.read()
    files = sorted(path.as_posix() for steps in ("8000", "32000", "128000") for path in (RUNS / steps).glob("*.json"))
    assert sorted(run.file for row in ledger.rows for run in row.runs) == files
    ledger.refresh(RUNS)
    ledger.write(written := tmp_path / "ledger.json")
    assert written.read_text() == PATH.read_text()


def test_every_name_spells_one_cell_and_reads_back() -> None:
    ledger = Ledger.read()
    cells = every_cell(ledger)
    names = {ledger.key(cell) for cell in cells}
    assert len(names) == len(cells)
    for cell in cells:
        steps = cell[STEPS]
        assert isinstance(steps, int)
        assert ledger.parse(ledger.name(cell), steps) == cell


def test_names_are_the_tags_of_what_differs_from_the_default() -> None:
    ledger = Ledger.read()
    assert ledger.name(ledger.cell({})) == "full"
    assert ledger.name(ledger.cell({"shared": "both", "pick": "keys"})) == "both-routed-keys"
    assert ledger.name(ledger.cell({"shared": "both", "chunks": 1, "rank": 24})) == "both-c1-r24"
    assert ledger.key(ledger.cell({"width": 128, STEPS: 32000})) == "32000/plain-w128"
    assert ledger.run_path(RUNS, ledger.cell({"modules": 2}), 5e-4, 1) == RUNS / "8000" / "plain-l2-lr0.0005-s1.json"
    assert ledger.sparse(ledger.cell({"shared": "ff"})) == {"shared": "ff", STEPS: 8000}
    for wrong in ("both-both", "both-routed-both", "full-c1", "c1-both", ""):
        with pytest.raises(ValueError):
            ledger.parse(wrong, 8000)
    with pytest.raises(ValueError):
        ledger.cell({"chunks": 3})
    with pytest.raises(ValueError):
        ledger.cell({"depth": 3})


def test_rounds_one_and_two_are_complete_at_both_budgets() -> None:
    ledger = Ledger.read()
    for steps in (8000, 32000):
        for name in ROUNDS_ONE_AND_TWO:
            row = ledger.row(ledger.parse(name, steps))
            assert row is not None and row.losses is not None and len(row.losses) == len(SEEDS), (name, steps)


def test_an_axis_added_to_the_ledger_reaches_names_cells_and_the_model(monkeypatch: pytest.MonkeyPatch) -> None:
    before = Ledger.read()
    added = fields()
    added["axes"] = [axis for axis in added["axes"] if axis["name"] != "architecture"]
    added["hypotheses"] = []
    added["cells"] = [row for row in added["cells"] if "architecture" not in row["cell"]]
    without = Ledger(json.loads(json.dumps(added)))
    added["axes"].append({"name": "architecture", "meaning": "", "default": "classic", "values": [["classic", "classic", "classic"], ["modern", "modern", "modern"]]})
    ledger = Ledger(added)
    classic = [before.key(row.cell) for row in before.rows if row.cell["architecture"] == "classic"]
    assert [ledger.key(row.cell) for row in ledger.rows] == classic
    assert all(row.cell["architecture"] == "classic" for row in ledger.rows)
    assert all("architecture" not in row.cell for row in without.rows)
    cell = ledger.cell({"shared": "both", "architecture": "modern", STEPS: 32000})
    assert ledger.key(cell) == "32000/both-modern"
    assert ledger.parse("both-modern", 32000) == cell
    monkeypatch.setattr(configs, "LEDGER", ledger)
    assert Transformer(model_config(cell)).position is None
    monkeypatch.setattr(configs, "LEDGER", without)
    assert Transformer(model_config(without.cell({"shared": "both"}))).position is not None


def test_an_axis_the_model_has_no_setting_for_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    added = fields()
    added["axes"].append({"name": "colour", "meaning": "", "default": "red", "values": [["red", "red", "red"]]})
    ledger = Ledger(added)
    monkeypatch.setattr(configs, "LEDGER", ledger)
    with pytest.raises(ValueError):
        model_config(ledger.cell({}))


def test_an_axis_needs_its_default_among_its_values_and_a_tag_and_plain_words_for_each() -> None:
    for axis in (
        {"name": "a", "meaning": "", "default": 1, "values": [[2, "two", "two"]]},
        {"name": "a", "meaning": "", "default": 1, "values": [[1, "one", "one"], [2, "one", "two"]]},
        {"name": "a", "meaning": "", "default": 1, "values": [[1, "", "one"]]},
        {"name": "a", "meaning": "", "default": 1, "values": [[1, "one", "one"], [2, "two", "one"]]},
        {"name": "a", "meaning": "", "default": 1, "values": [[1, "one", ""]]},
        {"name": "rank", "meaning": "", "default": 1, "values": [[1, "one", "one"]]},
        {"name": "a", "meaning": "", "default": 1, "values": [[1, "one", "one"], [2, "routed", "two"]]},
    ):
        added = fields()
        added["axes"].append(axis)
        with pytest.raises(ValueError):
            Ledger(added)


def test_with_nothing_shared_the_sharing_axes_stay_at_their_defaults() -> None:
    ledger = Ledger.read()
    changes: list[Cell] = [{"pick": "routed"}, {"chunks": 2}, {"rank": 24}]
    for changed in changes:
        with pytest.raises(ValueError):
            model_config(ledger.cell(changed))


def test_a_value_the_model_cannot_build_is_refused_when_a_cell_names_it(monkeypatch: pytest.MonkeyPatch) -> None:
    added = fields()
    next(axis for axis in added["axes"] if axis["name"] == "pick")["values"].append(["dice", "dice", "dice"])
    ledger = Ledger(added)
    monkeypatch.setattr(configs, "LEDGER", ledger)
    assert ledger.name(ledger.cell({"shared": "both", "pick": "dice"})) == "both-dice"
    with pytest.raises(ValueError):
        model_config(ledger.cell({"shared": "both", "pick": "dice"}))
    model_config(ledger.cell({"shared": "both"}))


def test_two_hypotheses_cannot_share_an_id() -> None:
    added = fields()
    added["hypotheses"].append(dict(added["hypotheses"][0]))
    with pytest.raises(ValueError):
        Ledger(added)


def test_reading_a_ledger_leaves_its_source_alone() -> None:
    source = fields()
    source["hypotheses"][0]["settle"][0]["lower"]["width"] = 256
    kept = json.dumps(source)
    ledger = Ledger(source)
    assert json.dumps(source) == kept
    assert "width" not in ledger.hypotheses[0]["settle"][0]["lower"]


def test_every_settling_cell_builds_a_model_and_every_hypothesis_has_a_priority() -> None:
    ledger = Ledger.read()
    assert len({hypothesis["id"] for hypothesis in ledger.hypotheses}) == len(ledger.hypotheses)
    for hypothesis in ledger.hypotheses:
        assert isinstance(hypothesis["priority"], int) and hypothesis["status"] in (OPEN, SUPPORTED, REFUTED)
        assert hypothesis["statement"]
        for comparison in hypothesis["settle"]:
            for side in ("lower", "higher"):
                model_config(ledger.cell(comparison[side]))


def test_a_verdict_needs_every_seed_on_one_side_and_a_mean_beyond_the_seed_spread() -> None:
    higher = [1.2, 1.21, 1.22]
    assert verdict([1.1, 1.11, 1.12], higher) == SUPPORTED
    assert verdict(higher, [1.1, 1.11, 1.12]) == REFUTED
    assert verdict([1.19, 1.22, 1.12], higher) == OPEN
    assert verdict([1.19, 1.21, 1.12], higher) == OPEN
    assert verdict([1.195, 1.205, 1.215], higher) == OPEN
    assert verdict([1.15, 1.2, 1.21], higher) == OPEN
    assert verdict([1.18, 1.19, 1.2], higher) == SUPPORTED
    assert verdict([1.10, 1.11, 1.12], [1.20, 1.21, 1.115]) == OPEN


def test_a_cell_with_two_complete_rates_reports_the_better_one() -> None:
    runs = [RunRow("", rate, seed, loss + seed / 100, 1.0) for rate, loss in ((1e-3, 1.2), (2e-3, 1.1)) for seed in SEEDS]
    row = Row({STEPS: 8000}, 1, [*runs, RunRow("", 4e-3, 0, 1.0, 1.0)])
    assert row.rate == 2e-3 and row.losses == [1.1, 1.11, 1.12]


def test_a_status_comes_from_finished_comparisons_that_agree() -> None:
    added = fields()
    both = {"shared": "both", STEPS: 32000}
    lower, higher = {"modules": 4, STEPS: 32000}, {"shared": "both", "pick": "routed", STEPS: 32000}
    added["hypotheses"] = [
        {"id": "unrun", "priority": 1, "status": SUPPORTED, "statement": "", "settle": [{"lower": {"shared": "both", "rank": 24, STEPS: 32000}, "higher": both, "differences": [0.0]}]},
        {"id": "split", "priority": 1, "status": SUPPORTED, "statement": "", "settle": [{"lower": both, "higher": lower}, {"lower": both, "higher": higher}, {"lower": higher, "higher": both}]},
        {"id": "waiting", "priority": 1, "status": SUPPORTED, "statement": "", "settle": [{"lower": both, "higher": lower}, {"lower": both, "higher": higher}, {"lower": {"shared": "both", STEPS: 128000}, "higher": both}]},
        {"id": "agreed", "priority": 1, "status": OPEN, "statement": "", "settle": [{"lower": both, "higher": lower}, {"lower": both, "higher": higher}]},
        {"id": "against", "priority": 1, "status": OPEN, "statement": "", "settle": [{"lower": higher, "higher": both}]},
        {"id": "nothing", "priority": 1, "status": SUPPORTED, "statement": "", "settle": []},
    ]
    ledger = Ledger(added)
    ledger.refresh(RUNS)
    assert [hypothesis["status"] for hypothesis in ledger.hypotheses] == [OPEN, OPEN, OPEN, SUPPORTED, REFUTED, OPEN]
    assert ["differences" in comparison for comparison in ledger.hypotheses[2]["settle"]] == [True, True, False]
    assert "differences" not in ledger.hypotheses[0]["settle"][0]
    assert all(difference < 0 for difference in ledger.hypotheses[3]["settle"][0]["differences"])


def test_a_run_file_under_another_cells_name_is_refused(tmp_path: Path) -> None:
    ledger = Ledger.read()
    (tmp_path / "8000").mkdir()
    run = json.loads((RUNS / "8000" / "full-lr0.001-s0.json").read_text())
    (tmp_path / "8000" / "plain-l4-lr0.001-s0.json").write_text(json.dumps(run))
    with pytest.raises(ValueError):
        ledger.refresh(tmp_path)


def test_runs_of_one_cell_that_store_different_weights_are_refused(tmp_path: Path) -> None:
    ledger = Ledger.read()
    (tmp_path / "8000").mkdir()
    for seed in (0, 1):
        run = json.loads((RUNS / "8000" / f"full-lr0.001-s{seed}.json").read_text())
        run["stored_weights"] += seed
        (tmp_path / "8000" / f"full-lr0.001-s{seed}.json").write_text(json.dumps(run))
    with pytest.raises(ValueError):
        ledger.refresh(tmp_path)


def test_each_pick_builds_its_own_kind_of_bank() -> None:
    ledger = Ledger.read()
    built = {
        pick: model_config(ledger.cell({"shared": "both", "pick": pick})).feed_forward
        for pick in ledger.axis("pick").tags
        if pick != "drawn"
    }
    assert built == {
        "fixed": Fixed(4),
        "ends": Fixed(4, Layout.ENDS),
        "routed": Routed(4),
        "argmax": Routed(4, noise=False),
        "keys": Routed(4, keys=True),
        "frozen": Routed(4, settled=Settled.PICKS),
        "stopped": Routed(4, settled=Settled.ROUTER),
    }
    drawn = {draw: model_config(ledger.cell({"shared": "ff", "pick": "drawn", "draw": draw})).feed_forward for draw in (1, 4)}
    assert drawn == {1: Fixed(4, Drawn(1)), 4: Fixed(4, Drawn(4))}
    partly = model_config(ledger.cell({"shared": "attn", "pick": "routed", "chunks": 2}))
    assert (partly.feed_forward, partly.attention) == (Own(), Routed(2))


def test_a_chart_label_is_the_plain_words_of_the_axes_and_never_a_tag() -> None:
    ledger = Ledger.read()
    assert plain_words(ledger.parse("both-c1-r24", 32000)) == (
        "attention and feed-forward shared, fixed map, 1 chunk, rank-24 correction per module"
    )
    assert plain_words(ledger.parse("ff-routed", 32000)) == "feed-forward shared, picked per token, 4 chunks"
    assert plain_words(ledger.parse("plain-l4-modern", 32000)) == "nothing shared, 4 modules, modern architecture"
    assert plain_words(ledger.cell({})) == "nothing shared"
    for cell in every_cell(ledger):
        words = plain_words(cell).split(", ")
        tags = {tag for axis in ledger.model_axes for value, tag in axis.tags.items() if axis.words[value] != tag}
        assert not tags & set(words)
        assert all(axis.words[cell[axis.name]] in words for axis in ledger.model_axes if cell[axis.name] != axis.default)


@pytest.mark.parametrize(
    ("cell", "refusal"),
    [
        ({"shared": "both", "pick": "drawn"}, "a draw goes with"),
        ({"shared": "both", "draw": 1}, "a draw goes with"),
        ({"shared": "both", "pick": "routed", "draw": 2}, "a draw goes with"),
        ({"draw": 1}, "with nothing shared"),
        ({"shared": "both", "pick": "ends", "chunks": 2}, "at least 3 chunks"),
        ({"shared": "attn", "pick": "drawn", "draw": 1, "modules": 2}, "cannot use all"),
    ],
)
def test_a_cell_whose_map_cannot_be_laid_out_is_refused_before_any_run(cell: Cell, refusal: str) -> None:
    with pytest.raises(ValueError, match=refusal):
        model_config(Ledger.read().cell(cell))


def test_a_map_other_than_the_mod_map_takes_its_rate_from_the_cell_that_has_a_sweep() -> None:
    assert configs.rate_source("both-drawn-d1-modern") == "both-modern" == configs.rate_source("both-ends-modern")
    assert configs.rate_source("both-drawn-d2-c8") == "full" == configs.rate_source("both-c8")
    assert configs.rate_source("attn-ends-c2") == "attn-c2"
    for name in ("full", "both", "both-routed", "plain-l4-modern", "both-c1-r24"):
        assert configs.rate_source(name) == name
