"""Structural invariants of the car park world model."""
import math

import pytest

from carpark import (
    build_demo_carpark, build_scaled_carpark, build_carpark,
    DEMO_SHORTCUT_CELL_IDX, RAMP_METRES_PER_FLOOR,
)


@pytest.fixture(scope="module")
def demo():
    return build_demo_carpark()


def test_demo_capacity(demo):
    assert demo.total_capacity == 60
    assert len(demo.floors) == 3
    assert all(f.capacity == 20 for f in demo.floors)


def test_scaled_matches_demo_at_10_bays(demo):
    scaled = build_scaled_carpark(bays_per_row=10)
    assert scaled.total_capacity == demo.total_capacity
    for bd, bs in zip(
        (b for f in demo.floors for b in f.bays),
        (b for f in scaled.floors for b in f.bays),
    ):
        assert bd.id == bs.id
        assert len(bd.entry_path_cells) == len(bs.entry_path_cells)
        assert len(bd.exit_path_cells) == len(bs.exit_path_cells)


def test_entry_paths_start_at_entrance_and_end_at_approach(demo):
    for f in demo.floors:
        for b in f.bays:
            path = b.entry_path_cells
            assert path, f"bay {b.id} has no entry path"
            assert path[0].lane_id == "TOP_F0" and path[0].index == 0, \
                f"bay {b.id} entry path does not start at the entrance cell"
            assert path[-1] is b.approach_cell, \
                f"bay {b.id} entry path does not end at its approach cell"


def test_exit_paths_start_at_approach_and_end_at_exit(demo):
    for f in demo.floors:
        for b in f.bays:
            path = b.exit_path_cells
            assert path, f"bay {b.id} has no exit path"
            assert path[0] is b.approach_cell
            last = path[-1]
            assert last.lane_id == "BOT_F0" and last.index == len(
                demo.lanes["BOT_F0"].cells) - 1, \
                f"bay {b.id} exit path does not end at USCITA"


def test_paths_never_go_backwards(demo):
    """One-way flow: within any lane, a path visits cells in strictly
    increasing index order (the physical no-reversing rule)."""
    for f in demo.floors:
        for b in f.bays:
            for path in (b.entry_path_cells, b.exit_path_cells):
                last_idx_per_lane = {}
                for cell in path:
                    prev = last_idx_per_lane.get(cell.lane_id)
                    if prev is not None:
                        assert cell.index == prev + 1, \
                            f"bay {b.id}: path jumps within {cell.lane_id}"
                    last_idx_per_lane[cell.lane_id] = cell.index


def test_upper_floors_cost_more_drive_distance_top_row(demo):
    """Same TOP-row bay position on a higher floor must have a strictly
    longer drive-in (more lane cells + ramp length).

    Deliberately restricted to the TOP row: for BOT-row bays west of the
    shortcut the relationship legitimately INVERTS — the one-way flow
    forces a Floor-0 car around the F2 U-turn and down through every
    BOT lane, so the same bay is a longer drive on a LOWER floor."""
    by_num = {}
    for f in demo.floors:
        for b in f.bays:
            if b.row == "TOP":
                by_num.setdefault(b.number, {})[b.floor] = b
    assert by_num, "no TOP-row bays found"
    for num, floors in by_num.items():
        assert floors[0].distance_to_entrance < floors[1].distance_to_entrance \
            < floors[2].distance_to_entrance


def test_demo_and_scaled_include_ramp_length(demo):
    """Regression: demo bays must include RAMP_METRES_PER_FLOOR per floor,
    exactly like the scaled builder (they once disagreed)."""
    scaled = build_scaled_carpark(bays_per_row=10)
    for cp in (demo, scaled):
        f0 = {b.number: b for b in cp.floors[0].bays}
        f1 = {b.number: b for b in cp.floors[1].bays}
        # Bay 1 (TOP row, west-most): F1 path adds one full TOP lane
        # traversal (11 cells) plus one ramp vs F0.
        extra_cells = len(f1[1].entry_path_cells) - len(f0[1].entry_path_cells)
        cell_metres = (f0[1].distance_to_entrance
                       / len(f0[1].entry_path_cells))
        expected = extra_cells * cell_metres + RAMP_METRES_PER_FLOOR
        actual = f1[1].distance_to_entrance - f0[1].distance_to_entrance
        assert math.isclose(actual, expected, rel_tol=1e-6)


def test_carpark_a_builds():
    cp = build_carpark()
    assert cp.total_capacity == 447
    assert len(cp.floors) == 3
