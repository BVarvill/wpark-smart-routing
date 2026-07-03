"""
WPark Car Park — Live pygame demo
=================================
Real-time animation of the 60-bay demo garage (20 bays × 3 floors,
--bays scales it up).  Cars drive the one-way lane network cell by
cell, queue behind each other, cross the ramps, and park on their
assigned floor.

Usage:
    python demo_pygame.py
    python demo_pygame.py --policy balanced_smart
    python demo_pygame.py --ga                  # use Grand Arcade arrival patterns
    python demo_pygame.py --peak-rate 120       # synthetic demand intensity

Keyboard:
    SPACE   pause / play
    UP/DN   speed up / slow down (1x – 500x realtime)
    1       switch to Nearest-to-Entrance (baseline)
    2       switch to Floor-Directed
    3       switch to RL Policy (PPO)
    R       reset clock to 06:00
    Q       quit
"""
from __future__ import annotations
import argparse
import math
import os
import sys
from typing import Optional, Tuple, List, Dict

import pygame

# Make sure we can import the simulation modules
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from carpark import (
    build_demo_carpark, build_scaled_carpark,
    DEMO_ENTRANCE_XY, DEMO_EXIT_XY, DEMO_SHOP_EXIT_XY,
    DEMO_TOP_PATH_Y, DEMO_BOTTOM_PATH_Y,
    DEMO_SHORTCUT_X, DEMO_UTURN_X,
    DEMO_WEST_X, DEMO_EAST_X,
    DEMO_CELL_XS, DEMO_CELL_STEP,
)

from engine import SimulationEngine, POLICIES
from demand import build_synthetic_demand, load_single_carpark, build_demand_profile


# ═══════════════════════════════════════════════════════════════════════════
# Layout constants
# ═══════════════════════════════════════════════════════════════════════════
WINDOW_W = 1700
WINDOW_H = 920
HUD_W = 340
FPS = 60

FLOOR_AREA_W = WINDOW_W - HUD_W - 40
EACH_FLOOR_W = FLOOR_AREA_W // 3
FLOOR_AREA_Y = 70
FLOOR_AREA_H = WINDOW_H - 130


# ═══════════════════════════════════════════════════════════════════════════
# Colours — modern restrained palette (flat + layered)
# ═══════════════════════════════════════════════════════════════════════════
# Background hierarchy
COL_BG            = (244, 246, 250)    # off-white app background
COL_PANEL         = (255, 255, 255)    # card background
COL_PANEL_BORDER  = (226, 230, 238)    # subtle card border

# Text
COL_TEXT          = (24, 28, 38)
COL_TEXT_MUTED    = (108, 115, 130)
COL_TEXT_FAINT    = (165, 172, 188)

# Floor chrome (neutral tints, not pastels)
COL_FLOOR_BG = [
    (244, 246, 250),
    (240, 243, 248),
    (236, 240, 246),
]
COL_FLOOR_ACCENT = [
    (72, 122, 194),      # Ground — blue
    (100, 138, 198),     # First  — lighter blue
    (56, 104, 176),      # Second — deeper blue
]

# Roads (dark asphalt)
COL_ROAD          = (60, 66, 76)
COL_ROAD_DARK     = (40, 44, 52)
COL_ROAD_EDGE     = (92, 99, 112)
COL_ROAD_LINE     = (240, 240, 244)    # white lane markings
COL_ROAD_ARROW    = (225, 227, 232)

# Parking bays
COL_BAY_EMPTY     = (222, 226, 234)    # light neutral
COL_BAY_EMPTY_BORDER = (250, 252, 255) # white-ish painted border
COL_BAY_OCC       = (196, 172, 172)    # muted dusty rose (occupied)
COL_BAY_NUMBER    = (134, 140, 152)

# Cars (rounded-rect states)
COL_CAR_MOVING    = (72, 130, 204)     # blue = in transit
COL_CAR_MOVING_BORDER = (40, 88, 156)
COL_CAR_CORRECT   = (72, 168, 108)     # green = parked, right floor
COL_CAR_CORRECT_BORDER = (42, 118, 70)
COL_CAR_WRONG     = (214, 96, 88)      # red = parked, wrong floor
COL_CAR_WRONG_BORDER = (160, 56, 48)
COL_CAR_TEXT      = (255, 255, 255)

# Ramps / bridges
COL_RAMP          = (82, 88, 102)      # dark gray
COL_RAMP_EDGE     = (56, 62, 74)
COL_RAMP_LINE     = (220, 222, 228)

# Entrance / exit markers
COL_ENTRANCE      = (72, 158, 104)
COL_EXIT          = (192, 78, 70)
COL_MALL          = (116, 74, 176)

# Accents
COL_ACCENT        = (72, 130, 204)
COL_ACCENT_SOFT   = (204, 218, 238)
COL_SUCCESS       = (72, 168, 108)
COL_WARNING       = (218, 162, 54)
COL_DANGER        = (214, 96, 88)

# Back-compat aliases — used by legacy functions still in the file
BG                = COL_BG
DARK              = COL_TEXT
CREAM             = COL_PANEL
MID_GREY          = COL_TEXT_MUTED
LIGHT_GREY        = COL_TEXT_FAINT
NAVY              = COL_ACCENT
GOLD              = COL_ACCENT
GOLD_DARK         = (52, 98, 168)
GOLD_LIGHT        = COL_ACCENT_SOFT
PANEL_BORDER      = COL_PANEL_BORDER
FLOOR_TINTS       = COL_FLOOR_BG
FLOOR_EDGES       = COL_FLOOR_ACCENT
BAY_AVAIL         = COL_BAY_EMPTY
BAY_AVAIL_EDGE    = COL_BAY_EMPTY_BORDER
BAY_OCC           = COL_BAY_OCC
BAY_OCC_EDGE      = COL_BAY_OCC
CAR_MOVING        = COL_CAR_MOVING
CAR_MOVING_EDGE   = COL_CAR_MOVING_BORDER
CAR_CORRECT       = COL_CAR_CORRECT
CAR_CORRECT_EDGE  = COL_CAR_CORRECT_BORDER
CAR_WRONG         = COL_CAR_WRONG
CAR_WRONG_EDGE    = COL_CAR_WRONG_BORDER
ENTRANCE_COL      = COL_ENTRANCE
EXIT_COL          = COL_EXIT
MALL_COL          = COL_MALL
PATH_COL          = COL_ROAD
PATH_EDGE         = COL_ROAD_DARK
PATH_ARROW        = COL_ROAD_ARROW
SHORTCUT_COL      = COL_RAMP
SHORTCUT_ARROW    = COL_RAMP_LINE
UTURN_COL         = COL_RAMP
UTURN_ARROW       = COL_RAMP_LINE
BRIDGE_COL        = COL_RAMP
BRIDGE_EDGE       = COL_RAMP_EDGE


# ═══════════════════════════════════════════════════════════════════════════
# World → screen transform (per floor subplot)
# ═══════════════════════════════════════════════════════════════════════════
def subplot_rect(floor_level: int) -> pygame.Rect:
    x0 = 20 + floor_level * EACH_FLOOR_W
    return pygame.Rect(x0, FLOOR_AREA_Y, EACH_FLOOR_W - 15, FLOOR_AREA_H)


def world_to_screen(wx: float, wy: float, floor_level: int) -> Tuple[int, int]:
    """Map (wx, wy) in world [0,100]×[0,100] to screen pixels within the
    floor's subplot, leaving padding for title and shop name."""
    rect = subplot_rect(floor_level)
    pad_top = 56     # room for card-style header
    pad_bot = 36     # room for bottom label
    pad_side = 24
    area_w = rect.width - 2 * pad_side
    area_h = rect.height - pad_top - pad_bot
    sx = rect.x + pad_side + (wx / 100.0) * area_w
    sy = rect.y + pad_top + (1 - wy / 100.0) * area_h
    return int(sx), int(sy)


def cell_px(floor_level: int) -> Tuple[int, int]:
    """Return (cell_width_px, cell_height_px) for the given floor.
    Cells are DEMO_CELL_STEP = 9 world units wide, and the full floor
    spans 100 world units in both x and y.  Everything downstream is
    measured in these units so the layout scales with the window size."""
    from carpark import DEMO_CELL_STEP
    # Get pixel distance between two world points at a known cell stride
    x0, y0 = world_to_screen(0.0, 0.0, floor_level)
    x1, y1 = world_to_screen(DEMO_CELL_STEP, DEMO_CELL_STEP, floor_level)
    return abs(x1 - x0), abs(y1 - y0)


def rounded_rect(screen, rect, color, border_radius=4, border_color=None, border_width=0):
    """Pygame wrapper for a filled rounded rect with optional border."""
    pygame.draw.rect(screen, color, rect, border_radius=border_radius)
    if border_color is not None and border_width > 0:
        pygame.draw.rect(screen, border_color, rect,
                         width=border_width, border_radius=border_radius)


def draw_card(screen, rect, title: Optional[str] = None, font_title=None):
    """Draw a UI card: white rounded rectangle with subtle border and
    an optional muted title label in the top-left."""
    rounded_rect(screen, rect, COL_PANEL,
                 border_radius=10,
                 border_color=COL_PANEL_BORDER, border_width=1)
    if title and font_title is not None:
        lbl = font_title.render(title.upper(), True, COL_TEXT_MUTED)
        screen.blit(lbl, (rect.x + 14, rect.y + 10))


def draw_progress_bar(screen, rect, value: float, max_value: float,
                      color=None, track_color=None):
    """Horizontal progress bar — rounded ends.  value/max clamped to [0,1]."""
    if track_color is None:
        track_color = (233, 237, 244)
    if color is None:
        color = COL_ACCENT
    radius = rect.height // 2
    rounded_rect(screen, rect, track_color, border_radius=radius)
    frac = 0.0 if max_value <= 0 else max(0.0, min(1.0, value / max_value))
    fill_w = int(rect.width * frac)
    if fill_w > 0:
        fill_rect = pygame.Rect(rect.x, rect.y, fill_w, rect.height)
        rounded_rect(screen, fill_rect, color, border_radius=radius)


# ─── Scene-graph helpers (cell-sized primitives) ─────────────────────────

def draw_road_cell(screen, wx, wy, floor_level, direction=None,
                   highlight=False):
    """Draw a single ROAD cell as a dark rounded rectangle the same
    size as a parking bay.  If direction is given, draw a small arrow
    inside the cell."""
    sx, sy = world_to_screen(wx, wy, floor_level)
    cw, ch = cell_px(floor_level)
    w = int(cw * 0.92)
    h = int(ch * 0.92)
    rect = pygame.Rect(sx - w // 2, sy - h // 2, w, h)
    col = COL_ROAD if not highlight else (75, 82, 96)
    rounded_rect(screen, rect, col,
                 border_radius=max(2, w // 8),
                 border_color=COL_ROAD_DARK, border_width=1)
    # Direction arrow
    if direction is not None:
        a = max(3, int(min(w, h) * 0.22))
        cx, cy = rect.centerx, rect.centery
        if direction == "east":
            pts = [(cx - a, cy - a // 2), (cx + a // 2, cy), (cx - a, cy + a // 2)]
        elif direction == "west":
            pts = [(cx + a, cy - a // 2), (cx - a // 2, cy), (cx + a, cy + a // 2)]
        elif direction == "south":
            pts = [(cx - a // 2, cy - a), (cx, cy + a // 2), (cx + a // 2, cy - a)]
        else:
            pts = [(cx - a // 2, cy + a), (cx, cy - a // 2), (cx + a // 2, cy + a)]
        pygame.draw.polygon(screen, COL_ROAD_ARROW, pts)


def draw_road_strip(screen, cell_positions, floor_level, direction=None):
    """Draw a row of road cells at the given world positions."""
    for wx, wy in cell_positions:
        draw_road_cell(screen, wx, wy, floor_level, direction=direction)


def draw_parking_space(screen, bay, floor_level, occupied: bool,
                       font_number=None):
    """Draw a single parking bay as a cell-sized rounded rectangle."""
    sx, sy = world_to_screen(bay.x, bay.y, floor_level)
    cw, ch = cell_px(floor_level)
    w = int(cw * 0.92)
    h = int(ch * 0.92)
    rect = pygame.Rect(sx - w // 2, sy - h // 2, w, h)
    color = COL_BAY_OCC if occupied else COL_BAY_EMPTY
    border = (180, 160, 160) if occupied else COL_BAY_EMPTY_BORDER
    rounded_rect(screen, rect, color,
                 border_radius=max(2, w // 8),
                 border_color=border, border_width=2)
    if font_number is not None and w >= 16:
        num = font_number.render(f"{bay.number:02d}", True, COL_BAY_NUMBER)
        nr = num.get_rect(center=(sx, sy))
        screen.blit(num, nr)


# Value gradient for RL mode: vivid purple (low value) → bright gold (high value)
COL_VALUE_LOW    = (140, 80, 220)     # vivid purple
COL_VALUE_MID    = (220, 160, 60)     # warm amber
COL_VALUE_HIGH   = (255, 210, 50)     # bright gold
COL_VALUE_LOW_B  = (100, 50, 170)     # purple border
COL_VALUE_HIGH_B = (190, 150, 20)     # gold border


def _lerp_color(c1, c2, t):
    """Linear interpolate between two RGB tuples, t in [0, 1]."""
    t = max(0.0, min(1.0, t))
    return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))


def draw_car(screen, wx: float, wy: float, floor_level: int,
             mode: str, correct: bool, dest_floor: int,
             font_label=None, use_value_gradient: bool = False,
             value_score: float = 0.5):
    """Draw a car as a rounded rectangle that FILLS the cell (~85%).

    use_value_gradient=True (RL mode): colours by revenue value
      purple (low) → gold (high) instead of red/green correct/wrong.
    use_value_gradient=False (baseline/floor-match): red/green as before.
    """
    sx, sy = world_to_screen(wx, wy, floor_level)
    cw, ch = cell_px(floor_level)
    w = int(cw * 0.84)
    h = int(ch * 0.84)
    rect = pygame.Rect(sx - w // 2, sy - h // 2, w, h)

    if mode == "parked":
        if use_value_gradient:
            # Purple → gold gradient based on value_score (0 = low, 1 = high)
            if value_score < 0.5:
                col = _lerp_color(COL_VALUE_LOW, COL_VALUE_MID, value_score * 2)
                border = _lerp_color(COL_VALUE_LOW_B, COL_VALUE_HIGH_B, value_score)
            else:
                col = _lerp_color(COL_VALUE_MID, COL_VALUE_HIGH, (value_score - 0.5) * 2)
                border = _lerp_color(COL_VALUE_LOW_B, COL_VALUE_HIGH_B, value_score)
        else:
            col = COL_CAR_CORRECT if correct else COL_CAR_WRONG
            border = COL_CAR_CORRECT_BORDER if correct else COL_CAR_WRONG_BORDER
    elif mode == "parking":
        if use_value_gradient:
            # RL mode: parking cars glow bright purple (distinct from gold parked)
            col = (160, 100, 240)
            border = (120, 60, 200)
        else:
            col = COL_WARNING
            border = (168, 120, 30)
    else:
        if use_value_gradient:
            # RL mode: moving cars also show their value colour
            if value_score < 0.5:
                col = _lerp_color(COL_VALUE_LOW, COL_VALUE_MID, value_score * 2)
                border = _lerp_color(COL_VALUE_LOW_B, COL_VALUE_HIGH_B, value_score)
            else:
                col = _lerp_color(COL_VALUE_MID, COL_VALUE_HIGH, (value_score - 0.5) * 2)
                border = _lerp_color(COL_VALUE_LOW_B, COL_VALUE_HIGH_B, value_score)
        else:
            col = COL_CAR_MOVING
            border = COL_CAR_MOVING_BORDER

    rounded_rect(screen, rect, col,
                 border_radius=max(3, min(w, h) // 4),
                 border_color=border, border_width=2)

    if font_label is not None and min(w, h) >= 12:
        lbl_txt = ["G", "1", "2"][dest_floor]
        lbl = font_label.render(lbl_txt, True, COL_CAR_TEXT)
        lr = lbl.get_rect(center=(sx, sy))
        screen.blit(lbl, lr)


def compute_car_state(v: Dict, sim_time: float, carpark) -> Optional[Tuple]:
    """Return (floor, x, y, mode, on_correct) at sim_time, or None if
    vehicle is inactive.  mode ∈ {'parked','moving'}.

    CELLULAR MODEL: the car's position is smoothly interpolated along
    its pre-computed sequence of lane cells.  Because each cell holds
    at most one car at a time, there is no possibility of overlap and
    no need for a no-overtaking clamp.
    """
    arr  = v["arrival_second"]
    pf   = v.get("parked_from") or v.get("parked_from_second") or 0
    dep  = v.get("departure_second") or 0
    gone = v.get("gone_second") or 0

    if sim_time < arr or sim_time > gone:
        return None

    bay = carpark.get_bay(v["assigned_bay"])
    if bay is None:
        return None

    correct = (v["assigned_floor"] == v["dest_floor"])

    # The 45-second parking manoeuvre happens BEFORE parked_from.
    # During this time the car is ON THE ROAD CELL (the approach cell),
    # physically blocking any car behind it.  This is the key congestion
    # mechanism that the cellular model captures.
    PARK_SECS = 45.0
    parking_start = max(arr, pf - PARK_SECS)

    # ── Driving phase — car moving cell-by-cell toward the approach cell ──
    if sim_time < parking_start:
        path = bay.entry_path_cells
        if not path:
            return None
        drive_dur = max(1.0, parking_start - arr)
        progress = min(1.0, max(0.0, (sim_time - arr) / drive_dur))
        f, x, y = _cell_path_position(path, progress)
        return (f, x, y, "moving", correct)

    # ── Parking phase — car is ON THE ROAD CELL, blocking traffic ──
    # Shown at the approach cell (the last cell in the entry path,
    # which is on the lane, not in the bay).  This is where you see
    # the car stationary while it reverses into the bay.
    if sim_time < pf:
        ac = bay.approach_cell
        if ac is not None:
            return (ac.floor, ac.x, ac.y, "parking", correct)
        # Fallback: last cell in entry path
        if bay.entry_path_cells:
            c = bay.entry_path_cells[-1]
            return (c.floor, c.x, c.y, "parking", correct)
        return (bay.floor, bay.x, bay.y, "parking", correct)

    # ── Parked phase — car is IN the bay, off the road ──
    if sim_time < dep:
        return (bay.floor, bay.x, bay.y, "parked", correct)

    # ── Exit phase ──
    path = bay.exit_path_cells
    if not path:
        return None
    exit_dur = max(1.0, gone - dep)
    progress = min(1.0, max(0.0, (sim_time - dep) / exit_dur))
    f, x, y = _cell_path_position(path, progress)
    return (f, x, y, "moving", correct)


def _cell_path_position(path, progress: float) -> Tuple[int, float, float]:
    """Return (floor, x, y) at fractional progress along a list of LaneCells.

    SMOOTH within same-floor cells, SNAP at floor transitions.
    Cars glide from one cell to the next, but never visually enter a
    cell on a different floor mid-transition (that would look like
    teleporting).  The physics still enforces one-car-per-cell — this
    is purely a visual smoothing layer.
    """
    if not path:
        return (0, 50, 50)
    n = len(path)
    if n == 1:
        return (path[0].floor, path[0].x, path[0].y)

    frac = progress * (n - 1)
    lo = max(0, min(int(frac), n - 2))
    t = frac - lo                          # 0..1 within this cell pair

    c0 = path[lo]
    c1 = path[min(lo + 1, n - 1)]

    if c0.floor != c1.floor:
        # Floor transition — snap cleanly (no interpolation across floors)
        if t < 0.5:
            return (c0.floor, c0.x, c0.y)
        else:
            return (c1.floor, c1.x, c1.y)

    # Same floor — smooth linear interpolation
    x = c0.x + (c1.x - c0.x) * t
    y = c0.y + (c1.y - c0.y) * t
    return (c0.floor, x, y)


# ═══════════════════════════════════════════════════════════════════════════
# Simulation wrapper
# ═══════════════════════════════════════════════════════════════════════════
class Sim:
    def __init__(self, policy: str, use_ga: bool, peak_rate: int,
                 bays_per_row: int = 10):
        self.policy = policy
        self.use_ga = use_ga
        self.peak_rate = peak_rate
        self.bays_per_row = bays_per_row
        self._rebuild()

    def _rebuild(self):
        if self.bays_per_row == 10:
            self.carpark = build_demo_carpark()
        else:
            self.carpark = build_scaled_carpark(bays_per_row=self.bays_per_row)
        if self.use_ga:
            # Scale Grand Arcade data down to ~120-bay car park
            # GA has ~780 bays; scale factor 120/780 ≈ 0.154
            data_dir = os.path.dirname(HERE)
            df = load_single_carpark(data_dir, "GA")
            self.demand = build_demand_profile(df, name="Grand Arcade (scaled)")
            # Scale hourly arrival rates down to fit our 120-bay layout
            GA_SCALE = 0.15
            for h in range(24):
                if h in self.demand.hourly_arrival_rate:
                    self.demand.hourly_arrival_rate[h] *= GA_SCALE
        else:
            self.demand = build_synthetic_demand("Demo", peak_arrivals_per_hour=self.peak_rate)

        self.engine = SimulationEngine(
            self.carpark, self.demand, self.policy, "Saturday", random_seed=1,
        )
        self.engine.generate_arrivals()
        self.metrics = self.engine.run()

        self.sim_time = 10.0 * 3600    # start at 10:00
        self.speed    = 1.0            # 1× = realistic (10s to cross a floor)
        self.paused   = False

    def step(self, dt_real: float):
        if not self.paused:
            self.sim_time += dt_real * self.speed
            if self.sim_time >= 24 * 3600:
                self.sim_time = 6.0 * 3600

    def switch_policy(self, new_policy: str):
        # Keep time position when switching
        t = self.sim_time
        speed = self.speed
        self.policy = new_policy
        self._rebuild()
        self.sim_time = t
        self.speed = speed


# ═══════════════════════════════════════════════════════════════════════════
# Renderer
# ═══════════════════════════════════════════════════════════════════════════
class Renderer:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
        pygame.display.set_caption("WPark Car Park — Live Demo")
        self.clock = pygame.time.Clock()

        # Professional sans-serif stack — fall back gracefully.
        # SysFont picks the best match on the current OS.
        def _sans(size, bold=False):
            return pygame.font.SysFont(
                "helveticaneue,helvetica,arial,sans", size, bold=bold)
        def _mono(size, bold=False):
            return pygame.font.SysFont(
                "menlo,monaco,couriernew", size, bold=bold)
        self.font_xl    = _sans(26, bold=True)
        self.font_big   = _sans(18, bold=True)
        self.font_med   = _sans(14, bold=True)
        self.font       = _sans(13)
        self.font_small = _sans(11)
        self.font_tiny  = _sans(9)
        self.font_mono  = _mono(13, bold=True)
        self.font_mono_small = _mono(10)

    def draw_title(self, sim: Sim):
        bar = pygame.Rect(0, 0, WINDOW_W, 52)
        pygame.draw.rect(self.screen, NAVY, bar)
        pygame.draw.line(self.screen, GOLD_DARK, (0, 50), (WINDOW_W, 50), 2)
        t = self.font_xl.render("WPark", True, CREAM)
        self.screen.blit(t, (22, 11))
        subtitle = self.font_med.render(
            "Smart Car Park Routing · Live Simulation", True, CREAM)
        self.screen.blit(subtitle, (120, 18))
        cp_name = (f"{sim.carpark.total_capacity}-bay demo · "
                   f"{len(sim.carpark.floors)} floors · "
                   f"one-way flow")
        ct = self.font_small.render(cp_name, True, LIGHT_GREY)
        self.screen.blit(ct, (WINDOW_W - HUD_W - 220, 22))

    def draw_bridges(self, sim: Sim):
        """Draw the ramp connectors between adjacent floor subplots.
        Uses the new dark-asphalt road style with white lane markings."""
        for src in range(2):
            dst = src + 1
            r_src = subplot_rect(src)
            r_dst = subplot_rect(dst)

            cw, _ = cell_px(src)
            thickness = max(6, int(cw * 0.80))

            # TOP bridge (east-bound) — shows an UP RAMP
            top_y = world_to_screen(DEMO_EAST_X, DEMO_TOP_PATH_Y, src)[1]
            x1 = r_src.right
            x2 = r_dst.left
            bridge_rect = pygame.Rect(x1, top_y - thickness // 2,
                                      x2 - x1, thickness)
            rounded_rect(self.screen, bridge_rect, COL_RAMP,
                         border_radius=thickness // 3,
                         border_color=COL_RAMP_EDGE, border_width=1)
            # Dashed white centre line
            mid_y = bridge_rect.centery
            for dash_x in range(bridge_rect.x + 4, bridge_rect.right - 4, 8):
                pygame.draw.line(self.screen, COL_RAMP_LINE,
                                 (dash_x, mid_y), (dash_x + 4, mid_y), 1)
            # Arrow in the middle
            mx = bridge_rect.centerx
            pygame.draw.polygon(self.screen, COL_ROAD_ARROW, [
                (mx - 4, mid_y - 5),
                (mx + 5, mid_y),
                (mx - 4, mid_y + 5),
            ])
            lbl = self.font_tiny.render("UP RAMP", True, COL_TEXT_MUTED)
            self.screen.blit(lbl, (x1 + 2, top_y - thickness // 2 - 14))

            # BOT bridge (west-bound) — shows a DOWN RAMP
            bot_y = world_to_screen(DEMO_WEST_X, DEMO_BOTTOM_PATH_Y, src)[1]
            bbridge_rect = pygame.Rect(x1, bot_y - thickness // 2,
                                       x2 - x1, thickness)
            rounded_rect(self.screen, bbridge_rect, COL_RAMP,
                         border_radius=thickness // 3,
                         border_color=COL_RAMP_EDGE, border_width=1)
            bmid_y = bbridge_rect.centery
            for dash_x in range(bbridge_rect.x + 4, bbridge_rect.right - 4, 8):
                pygame.draw.line(self.screen, COL_RAMP_LINE,
                                 (dash_x, bmid_y), (dash_x + 4, bmid_y), 1)
            pygame.draw.polygon(self.screen, COL_ROAD_ARROW, [
                (mx + 4, bmid_y - 5),
                (mx - 5, bmid_y),
                (mx + 4, bmid_y + 5),
            ])
            lbl = self.font_tiny.render("DOWN RAMP", True, COL_TEXT_MUTED)
            self.screen.blit(lbl, (x1 + 2, bot_y + thickness // 2 + 2))

    def draw_floor(self, floor_level: int, sim: Sim, occupied_ids: set):
        """Layered render: background → roads → parking bays → labels.
        Cars are drawn separately in draw_cars() so they end up on top."""
        f = sim.carpark.get_floor(floor_level)
        if f is None:
            return

        rect = subplot_rect(floor_level)
        accent = COL_FLOOR_ACCENT[floor_level]

        # ── Layer 0: floor card (rounded background panel) ──
        rounded_rect(self.screen, rect, COL_PANEL,
                     border_radius=12,
                     border_color=COL_PANEL_BORDER, border_width=1)

        # ── Layer 1: floor tint band (subtle) ──
        tint_rect = pygame.Rect(rect.x + 4, rect.y + 52,
                                rect.width - 8, rect.height - 60)
        rounded_rect(self.screen, tint_rect,
                     COL_FLOOR_BG[floor_level],
                     border_radius=8)

        # ── Header bar ──
        header = pygame.Rect(rect.x, rect.y, rect.width, 48)
        header.inflate_ip(-8, 0)
        header.y += 4
        rounded_rect(self.screen, header, COL_PANEL, border_radius=8)
        # Accent bar on the left of the header
        acc_bar = pygame.Rect(header.x + 6, header.y + 10, 3, header.height - 20)
        rounded_rect(self.screen, acc_bar, accent, border_radius=2)
        title = self.font_big.render(f"FLOOR {f.level}", True, COL_TEXT)
        self.screen.blit(title, (header.x + 18, header.y + 7))
        shop_name = f.shops[0].name if f.shops else ""
        name_col = self.font_small.render(
            f.name + "  ·  " + shop_name, True, COL_TEXT_MUTED)
        self.screen.blit(name_col, (header.x + 18, header.y + 28))

        # Occupancy pill (top right)
        occ_count = sum(1 for b in f.bays if b.id in occupied_ids)
        pill_text = f"{occ_count}/{f.capacity}"
        pill = self.font_med.render(pill_text, True, accent)
        pr = pill.get_rect(topright=(header.right - 14, header.y + 12))
        self.screen.blit(pill, pr)

        # ── Layer 2: roads as discrete grid cells ──
        # TOP_PATH cells (east-bound)
        top_cells = [(x, DEMO_TOP_PATH_Y) for x in DEMO_CELL_XS]
        draw_road_strip(self.screen, top_cells, floor_level, direction="east")

        # BOTTOM_PATH cells (west-bound)
        bot_cells = [(x, DEMO_BOTTOM_PATH_Y) for x in reversed(DEMO_CELL_XS)]
        draw_road_strip(self.screen, bot_cells, floor_level, direction="west")

        # Vertical spur cells (shortcut on F0/F1, U-turn on F2)
        if floor_level < 2:
            spur_x = DEMO_SHORTCUT_X
            spur_label = "SHORTCUT"
        else:
            spur_x = DEMO_UTURN_X
            spur_label = "U-TURN"
        spur_ys = [64, 55, 46, 37, 28]  # intermediate cells between top and bottom paths
        spur_cells = [(spur_x, y) for y in spur_ys]
        draw_road_strip(self.screen, spur_cells, floor_level, direction="south")

        # Spur label
        spur_x_px, spur_top_y = world_to_screen(spur_x, DEMO_TOP_PATH_Y, floor_level)
        _, spur_bot_y = world_to_screen(spur_x, DEMO_BOTTOM_PATH_Y, floor_level)
        lbl = self.font_tiny.render(spur_label, True, COL_TEXT_MUTED)
        self.screen.blit(lbl,
                         (spur_x_px + 12, (spur_top_y + spur_bot_y) // 2 - 5))

        # ── Layer 4: parking bays (rounded rectangles with white borders) ──
        for b in f.bays:
            draw_parking_space(self.screen, b, floor_level,
                               occupied=(b.id in occupied_ids),
                               font_number=self.font_tiny)

        # ── Layer 5: mall entrance at south centre ──
        mx, my = world_to_screen(*DEMO_SHOP_EXIT_XY, floor_level)
        cw, _ = cell_px(floor_level)
        mw = int(cw * 5.0)
        mh = int(cw * 0.55)
        mall_rect = pygame.Rect(mx - mw // 2, my - mh // 2, mw, mh)
        rounded_rect(self.screen, mall_rect, COL_MALL,
                     border_radius=max(3, mh // 3))
        # Door hints (3 vertical lines)
        for dx in (-mw // 4, 0, mw // 4):
            pygame.draw.line(self.screen, COL_PANEL,
                             (mall_rect.centerx + dx, mall_rect.y + 4),
                             (mall_rect.centerx + dx, mall_rect.bottom - 4), 1)
        mall_lbl = self.font_tiny.render("MALL ENTRANCE", True, COL_PANEL)
        mlr = mall_lbl.get_rect(center=mall_rect.center)
        self.screen.blit(mall_lbl, mlr)

        # ── Layer 6: entrance/exit markers (ground floor only) ──
        if floor_level == 0:
            self._draw_entry_marker(DEMO_ENTRANCE_XY, "ENTRANCE",
                                    COL_ENTRANCE, floor_level)
            self._draw_exit_marker(DEMO_EXIT_XY, "EXIT",
                                   COL_EXIT, floor_level)

    def _draw_entry_marker(self, xy, label, color, floor_level):
        x, y = world_to_screen(*xy, floor_level)
        # Rounded badge with arrow going east into the car park
        w, h = 14, 22
        rect = pygame.Rect(x - w - 2, y - h // 2, w, h)
        rounded_rect(self.screen, rect, color, border_radius=3)
        # Small arrow tip pointing right
        tip = [(rect.right, y - 6), (rect.right + 8, y),
               (rect.right, y + 6)]
        pygame.draw.polygon(self.screen, color, tip)
        lbl = self.font_tiny.render(label, True, color)
        self.screen.blit(lbl, (x - 56, y - 24))

    def _draw_exit_marker(self, xy, label, color, floor_level):
        x, y = world_to_screen(*xy, floor_level)
        w, h = 14, 22
        rect = pygame.Rect(x - w - 2, y - h // 2, w, h)
        rounded_rect(self.screen, rect, color, border_radius=3)
        # Arrow tip pointing LEFT (out of the park)
        tip = [(rect.x, y - 6), (rect.x - 8, y), (rect.x, y + 6)]
        pygame.draw.polygon(self.screen, color, tip)
        lbl = self.font_tiny.render(label, True, color)
        self.screen.blit(lbl, (x - 44, y + 12))

    def draw_cars(self, sim: Sim):
        """Draw every active car using the rounded-rect helper.
        Cars are rendered on top of everything else so they're always
        the visual focus."""
        # ── Pass 1: collect raw state for every active car ──
        active = []
        for v in sim.metrics.vehicles_log:
            state = compute_car_state(v, sim.sim_time, sim.carpark)
            if state is None:
                continue
            floor, wx, wy, mode, correct = state
            active.append({
                "v": v,
                "floor": floor,
                "wx": wx,
                "wy": wy,
                "mode": mode,
                "correct": correct,
                "arrival": v["arrival_second"],
            })

        # ── Pass 2: collision clamp — no visual overtaking ──
        # Group moving cars by floor and lane-y (TOP_PATH y=70, BOT_PATH y=25).
        # Within each group, sort by arrival order and ensure no car's x
        # position passes the car ahead of it.  Cars that would overlap
        # are clamped to sit one cell-width behind the car in front.
        # Minimum gap between cars on the same lane, in world units
        MIN_GAP_WORLD = 4.0

        from collections import defaultdict
        lane_groups = defaultdict(list)
        for c in active:
            if c["mode"] in ("moving", "parking"):
                # Group by floor + approximate lane y
                lane_key = (c["floor"], round(c["wy"] / 10) * 10)
                lane_groups[lane_key].append(c)

        for key, cars in lane_groups.items():
            floor, lane_y = key
            if abs(lane_y - 70) < 15:
                # TOP_PATH — cars move east (increasing x). Earlier arrivals are AHEAD (higher x).
                cars.sort(key=lambda c: c["arrival"])
                # Walk from front to back, clamping
                for i in range(1, len(cars)):
                    ahead = cars[i - 1]
                    behind = cars[i]
                    if behind["wx"] > ahead["wx"] - MIN_GAP_WORLD:
                        behind["wx"] = ahead["wx"] - MIN_GAP_WORLD
            elif abs(lane_y - 25) < 15:
                # BOT_PATH — cars move west (decreasing x). Earlier arrivals are AHEAD (lower x).
                cars.sort(key=lambda c: c["arrival"])
                for i in range(1, len(cars)):
                    ahead = cars[i - 1]
                    behind = cars[i]
                    if behind["wx"] < ahead["wx"] + MIN_GAP_WORLD:
                        behind["wx"] = ahead["wx"] + MIN_GAP_WORLD

        # ── Pass 3: draw each car ──
        # If the current policy is RL (neural_smart), use the value
        # gradient (purple → gold) instead of correct/wrong colouring.
        is_rl = sim.policy == "neural_smart"

        for c in active:
            # Compute value score for RL gradient (0 = low value, 1 = high)
            val_score = 0.5
            if is_rl:
                v = c["v"]
                # Score based on total_wasted relative to the range
                # Lower wasted = higher value
                tw = v.get("total_wasted", 200)
                # Map: 100s (excellent) → 1.0, 300s (poor) → 0.0
                val_score = max(0.0, min(1.0, (300 - tw) / 200.0))

            draw_car(
                self.screen,
                c["wx"], c["wy"], c["floor"],
                mode=c["mode"], correct=c["correct"],
                dest_floor=c["v"]["dest_floor"],
                font_label=self.font_tiny,
                use_value_gradient=is_rl,
                value_score=val_score,
            )

    def draw_hud(self, sim: Sim, occupied_ids: set, moving_count: int):
        """Card-based HUD: every metric lives in a rounded white card
        with a clear label and (where appropriate) a progress bar."""
        hud_x = WINDOW_W - HUD_W
        col_x = hud_x + 14
        col_w = HUD_W - 28

        y = 66

        # ── Card 1: Clock + playback ──
        clock_card = pygame.Rect(col_x, y, col_w, 110)
        draw_card(self.screen, clock_card, "Time", self.font_tiny)

        h = int(sim.sim_time // 3600)
        m = int((sim.sim_time % 3600) // 60)
        s = int(sim.sim_time % 60)
        clk = self.font_xl.render(f"{h:02d}:{m:02d}", True, COL_TEXT)
        self.screen.blit(clk, (col_x + 14, y + 28))
        sec = self.font_small.render(f"{s:02d}s", True, COL_TEXT_MUTED)
        self.screen.blit(sec, (col_x + 14 + clk.get_width() + 4, y + 40))

        # Playback speed line
        spd_lbl = self.font_tiny.render("PLAYBACK", True, COL_TEXT_MUTED)
        self.screen.blit(spd_lbl, (col_x + 14, y + 66))
        spd_txt = "PAUSED" if sim.paused else f"{sim.speed:.0f}× realtime"
        spd_col = COL_DANGER if sim.paused else COL_TEXT
        sp = self.font_med.render(spd_txt, True, spd_col)
        self.screen.blit(sp, (col_x + 14, y + 80))
        # Speed bar
        bar = pygame.Rect(col_x + 14, y + 100, col_w - 28, 5)
        spd_frac = 0.0 if sim.paused else min(1.0, math.log(max(1, sim.speed)) / math.log(200))
        draw_progress_bar(self.screen, bar, spd_frac, 1.0,
                          color=COL_ACCENT)

        y = clock_card.bottom + 10

        # ── Card 2: Routing model ──
        model_card = pygame.Rect(col_x, y, col_w, 62)
        draw_card(self.screen, model_card, "Routing model", self.font_tiny)
        pol_label, _ = POLICIES[sim.policy]
        pol_txt = self.font_big.render(pol_label, True, COL_TEXT)
        self.screen.blit(pol_txt, (col_x + 14, y + 28))

        y = model_card.bottom + 10

        # ── Card 3: Live stats ──
        total_parked = len(occupied_ids)
        cp_total = sim.carpark.total_capacity
        live_card = pygame.Rect(col_x, y, col_w, 162)
        draw_card(self.screen, live_card, "Live stats", self.font_tiny)

        # Headline: parked total + progress bar
        parked_lbl = self.font_small.render("TOTAL PARKED", True, COL_TEXT_MUTED)
        self.screen.blit(parked_lbl, (col_x + 14, y + 28))
        parked_val = self.font_big.render(
            f"{total_parked} / {cp_total}", True, COL_TEXT)
        self.screen.blit(parked_val, (col_x + 14, y + 44))
        parked_bar = pygame.Rect(col_x + 14, y + 72, col_w - 28, 6)
        draw_progress_bar(self.screen, parked_bar,
                          total_parked, max(cp_total, 1),
                          color=COL_SUCCESS)

        # Per-floor mini bars
        row_y = y + 90
        for floor in sim.carpark.floors:
            occ_here = sum(1 for b in floor.bays if b.id in occupied_ids)
            pct = occ_here / max(floor.capacity, 1)
            lbl = self.font_small.render(f"F{floor.level}", True, COL_TEXT_MUTED)
            self.screen.blit(lbl, (col_x + 14, row_y))
            val = self.font_small.render(
                f"{occ_here}/{floor.capacity}", True, COL_TEXT)
            vr = val.get_rect(topright=(live_card.right - 14, row_y))
            self.screen.blit(val, vr)
            bar_rect = pygame.Rect(col_x + 36, row_y + 6,
                                   col_w - 100, 5)
            draw_progress_bar(self.screen, bar_rect, pct, 1.0,
                              color=COL_FLOOR_ACCENT[floor.level])
            row_y += 18
        # Moving line
        moving_lbl = self.font_small.render(
            f"Driving now: {moving_count}", True, COL_TEXT_MUTED)
        self.screen.blit(moving_lbl, (col_x + 14, row_y + 2))

        y = live_card.bottom + 10

        # ── Card 4: Full-day metrics ──
        m_obj = sim.metrics
        metrics_card = pygame.Rect(col_x, y, col_w, 192)
        draw_card(self.screen, metrics_card, "Full-day averages",
                  self.font_tiny)

        def _metric_row(label, value, bar_value, bar_max, bar_color, offset):
            lbl = self.font_small.render(label, True, COL_TEXT_MUTED)
            self.screen.blit(lbl, (col_x + 14, y + offset))
            val = self.font_small.render(value, True, COL_TEXT)
            vr = val.get_rect(topright=(metrics_card.right - 14, y + offset))
            self.screen.blit(val, vr)
            bar_rect = pygame.Rect(col_x + 14, y + offset + 17,
                                   col_w - 28, 4)
            draw_progress_bar(self.screen, bar_rect,
                              bar_value, bar_max,
                              color=bar_color)

        _metric_row("Avg cruise time",
                    f"{m_obj.avg_cruise_time:.0f}s",
                    m_obj.avg_cruise_time, 150,
                    COL_ACCENT, 28)
        _metric_row("Avg walk time",
                    f"{m_obj.avg_walk_time:.0f}s",
                    m_obj.avg_walk_time, 150,
                    COL_WARNING, 56)
        _metric_row("Avg queue wait",
                    f"{getattr(m_obj, 'avg_queue_wait_seconds', 0):.1f}s",
                    getattr(m_obj, 'avg_queue_wait_seconds', 0), 60,
                    COL_DANGER, 84)
        _metric_row("Correct floor",
                    f"{m_obj.correct_floor_pct:.0f}%",
                    m_obj.correct_floor_pct, 100,
                    COL_SUCCESS, 112)
        _metric_row("Served today",
                    f"{m_obj.vehicles_served}",
                    m_obj.vehicles_served, max(1, m_obj.total_vehicles),
                    COL_ACCENT, 140)

        y = metrics_card.bottom + 10

        # ── Card 5: Controls (bottom) ──
        ctrl_card = pygame.Rect(col_x, WINDOW_H - 146, col_w, 132)
        draw_card(self.screen, ctrl_card, "Controls", self.font_tiny)
        ctrl_y = ctrl_card.y + 28
        for key, desc in [
            ("SPACE", "pause / play"),
            ("↑ / ↓", "playback speed"),
            ("1",     "Baseline (no routing)"),
            ("2",     "Floor-Match"),
            ("3",     "RL Policy (PPO)"),
            ("R",     "reset clock"),
            ("Q",     "quit"),
        ]:
            k = self.font_mono_small.render(key, True, COL_ACCENT)
            self.screen.blit(k, (col_x + 14, ctrl_y))
            d = self.font_small.render(desc, True, COL_TEXT_MUTED)
            self.screen.blit(d, (col_x + 14 + 52, ctrl_y))
            ctrl_y += 16


# ═══════════════════════════════════════════════════════════════════════════
# Main loop
# ═══════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="WPark Car Park — pygame live demo")
    parser.add_argument("--policy", default="nearest_entrance",
                        choices=list(POLICIES.keys()),
                        help="initial routing policy (default: baseline)")
    parser.add_argument("--ga", action="store_true",
                        help="use Grand Arcade real demand data (scaled 0.15×)")
    parser.add_argument("--peak-rate", type=int, default=60,
                        help="synthetic peak arrivals per hour "
                             "(default 60 — park will saturate at peak hours)")
    parser.add_argument("--bays", type=int, default=10,
                        choices=[10, 15, 20, 25, 30],
                        help="bays per row per floor (10=60 bays, 20=120 bays, 30=180 bays)")
    args = parser.parse_args()

    total_bays = args.bays * 2 * 3
    print(f"→ Building simulation ({total_bays} bays = {args.bays}/row × 2 rows × 3 floors)...", flush=True)
    sim = Sim(args.policy, args.ga, args.peak_rate, bays_per_row=args.bays)
    print(f"→ Ready.  Served {sim.metrics.vehicles_served} cars over the day.", flush=True)

    renderer = Renderer()
    running = True

    while running:
        dt = renderer.clock.tick(FPS) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    sim.paused = not sim.paused
                elif event.key == pygame.K_UP:
                    sim.speed = min(sim.speed * 1.5, 500)
                elif event.key == pygame.K_DOWN:
                    sim.speed = max(sim.speed / 1.5, 1)
                elif event.key == pygame.K_1:
                    print("→ Switching to Baseline (no routing)...", flush=True)
                    sim.switch_policy("nearest_entrance")
                elif event.key == pygame.K_2:
                    print("→ Switching to Floor-Match...", flush=True)
                    sim.switch_policy("floor_directed")
                elif event.key == pygame.K_3:
                    print("→ Switching to RL Policy (PPO)...", flush=True)
                    sim.switch_policy("neural_smart")
                elif event.key == pygame.K_r:
                    sim.sim_time = 6.0 * 3600
                elif event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                    running = False

        sim.step(dt)

        # Compute which bays are occupied + how many cars are driving
        occupied_ids = set()
        moving_count = 0
        for v in sim.metrics.vehicles_log:
            state = compute_car_state(v, sim.sim_time, sim.carpark)
            if state is None:
                continue
            _, _, _, mode, _ = state
            if mode == "parked":
                occupied_ids.add(v["assigned_bay"])
            else:
                moving_count += 1

        # Draw everything
        renderer.screen.fill(BG)
        renderer.draw_title(sim)
        for fl in range(3):
            renderer.draw_floor(fl, sim, occupied_ids)
        renderer.draw_bridges(sim)
        renderer.draw_cars(sim)
        renderer.draw_hud(sim, occupied_ids, moving_count)
        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
