"""
Car Park Model — Faithful to parking plan -2.png (Car Park A)
=============================================================
Each floor has 149 numbered parking bays in the EXACT layout from
the architectural drawing.  The same layout is repeated on three
floors connected by a ramp located right next to the entrance.

Sections (matching the drawing):
   TOP        bays  1-11   horizontal row near the entrance
   RIGHT      bays 12-31   right-side column (20 bays)
   UL_L/UL_R  bays 80-87 / 79-72   upper-left facing pair
   UR_L/UR_R  bays 49-55 / 48-42   upper-right facing pair
   LL_L/LL_R  bays 88-96 / 71-65   lower-left facing pair
   LR_L/LR_R  bays 56-64 / 41-32   lower-right facing pair
   BOT_COL    bays 97-104  short column under the lower section
   LEFT       bays 109-129 (no 128) left side along the building
   SOUTH      bays 105-108 short south stub
   BOTTOM     bays 130-150 horizontal row across the bottom
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum
import math


# ── Enums ────────────────────────────────────────────────────────────────────
class BayStatus(Enum):
    AVAILABLE = "available"
    OCCUPIED = "occupied"


class BayType(Enum):
    STANDARD = "standard"
    DISABLED = "disabled"
    EV = "ev"


class ShopCategory(Enum):
    SUPERMARKET = "supermarket"
    CAFE = "cafe"
    PHARMACY = "pharmacy"
    FASHION = "fashion"
    ELECTRONICS = "electronics"
    RESTAURANT = "restaurant"
    CINEMA = "cinema"
    FOOD_COURT = "food_court"
    GYM = "gym"
    SERVICES = "services"


# ── Data classes ─────────────────────────────────────────────────────────────
@dataclass
class Shop:
    name: str
    category: ShopCategory
    floor: int
    avg_visit_minutes: float
    peak_hours: List[int]
    # Economic model: average spend per hour that a customer is in the shop.
    # Sourced from published UK retail spending averages per category (2024).
    spend_per_hour: float = 12.0


@dataclass
class ShopExit:
    name: str
    short_label: str
    x: float
    y: float


@dataclass
class ParkingBay:
    id: str
    floor: int
    number: int
    section: str
    x: float
    y: float
    bay_type: BayType = BayType.STANDARD
    status: BayStatus = BayStatus.AVAILABLE
    occupied_by: Optional[str] = None
    # Distances in metres
    distance_to_entrance: float = 0.0   # driving distance from entrance to bay
    distance_to_exit: float = 0.0       # driving distance from bay to exit
    distance_to_ramp: float = 0.0       # driving distance to the up-ramp
    distance_to_shops: float = 0.0      # walking distance from bay to nearest shop access
    # Base TIMES in seconds (congestion-free)
    cruise_in_seconds: float = 0.0      # time to drive in + park
    cruise_out_seconds: float = 0.0     # time to leave the bay + drive out
    walk_seconds_to_shops: float = 0.0  # walk time from bay to nearest shop access (no stair penalty)
    # Animation paths — (x, y) tuples on the DESTINATION floor only.
    entry_path: List[Tuple[float, float]] = field(default_factory=list)
    exit_path: List[Tuple[float, float]] = field(default_factory=list)
    # Legacy multi-floor journey for Streamlit compatibility.
    entry_journey: List[Tuple[int, float, float]] = field(default_factory=list)
    exit_journey:  List[Tuple[int, float, float]] = field(default_factory=list)
    # Cellular model — the bay's approach cell and full cell paths.
    approach_cell:    Optional["LaneCell"] = None
    entry_path_cells: List["LaneCell"]     = field(default_factory=list)
    exit_path_cells:  List["LaneCell"]     = field(default_factory=list)
    # Lane-segment sequences (for Option A queuing).  Each item is a segment
    # ID; the order is the order the car traverses them.  The last segment
    # in entry_segments is the "approach lane" that gets locked for the
    # park-manoeuvre duration when the car finishes arriving.
    entry_segments: List[str] = field(default_factory=list)
    exit_segments:  List[str] = field(default_factory=list)
    # backward-compat alias for old code that referenced bay.row
    row: str = ""

    def is_available(self) -> bool:
        return self.status == BayStatus.AVAILABLE


# ── Lane cells — cellular-automaton traffic model ───────────────────────
# Each lane is a list of discrete cells.  A cell holds at most one car at
# a time.  A car can only advance if the next cell in its path is empty.
# Parking cars hold their approach cell for PARK_MANEUVER_SECONDS ticks,
# physically blocking any car behind them.  This model is exact — there
# are no tuned congestion parameters and overtaking is impossible by
# construction.

@dataclass
class LaneCell:
    lane_id: str
    index: int              # position within the lane (0 = start)
    x: float                # world x (for rendering)
    y: float                # world y
    floor: int              # which floor this cell belongs to
    car_id: Optional[str] = None   # the car currently occupying the cell

    def __hash__(self) -> int:
        return hash((self.lane_id, self.index))

    def __eq__(self, other) -> bool:
        return (isinstance(other, LaneCell)
                and self.lane_id == other.lane_id
                and self.index == other.index)


@dataclass
class Lane:
    id: str
    floor: int
    direction: str          # "east" | "west" | "south"
    cells: List[LaneCell] = field(default_factory=list)


# ── Legacy: lane segments (kept for Streamlit backward compat) ──
@dataclass
class LaneSegment:
    id: str
    passage_seconds: float   # how long it takes to clear the segment when empty
    length_m: float          # physical length (for reference / rendering)


@dataclass
class Floor:
    level: int
    name: str
    bays: List[ParkingBay] = field(default_factory=list)
    shops: List[Shop] = field(default_factory=list)
    capacity: int = 0

    @property
    def occupied_count(self) -> int:
        return sum(1 for b in self.bays if b.status == BayStatus.OCCUPIED)

    @property
    def available_count(self) -> int:
        return sum(1 for b in self.bays if b.is_available())

    @property
    def occupancy_pct(self) -> float:
        if self.capacity == 0:
            return 0.0
        return self.occupied_count / self.capacity * 100


@dataclass
class CarPark:
    name: str
    floors: List[Floor] = field(default_factory=list)
    entry_side: str = "right"
    # Cellular lane network (populated by build_demo_carpark).  Empty for
    # the full-scale Car Park A model which still uses the old segment-FIFO.
    lanes: Dict[str, "Lane"] = field(default_factory=dict)

    @property
    def total_capacity(self) -> int:
        return sum(f.capacity for f in self.floors)

    @property
    def total_occupied(self) -> int:
        return sum(f.occupied_count for f in self.floors)

    @property
    def total_available(self) -> int:
        return sum(f.available_count for f in self.floors)

    @property
    def overall_occupancy_pct(self) -> float:
        if self.total_capacity == 0:
            return 0.0
        return self.total_occupied / self.total_capacity * 100

    def get_floor(self, level: int) -> Optional[Floor]:
        for f in self.floors:
            if f.level == level:
                return f
        return None

    def get_bay(self, bay_id: str) -> Optional[ParkingBay]:
        for f in self.floors:
            for b in f.bays:
                if b.id == bay_id:
                    return b
        return None

    def get_all_available_bays(self) -> List[ParkingBay]:
        return [b for f in self.floors for b in f.bays if b.is_available()]

    def reset(self):
        for f in self.floors:
            for b in f.bays:
                b.status = BayStatus.AVAILABLE
                b.occupied_by = None


# ── Layout constants — coordinates 0-100 ─────────────────────────────────────
ENTRANCE_XY = (82, 97)        # entrance arrow on the floor plan
EXIT_XY     = (20, 53)        # exit arrow on the floor plan

# Two ramps on the plan:
#   UP   — a bit inboard from the entrance, used when driving UP to floors 1-2
#   DOWN — near the exit, used when driving DOWN to leave from floors 1-2
UP_RAMP_XY   = (72, 92)       # up-ramp slightly away from the entrance
DOWN_RAMP_XY = (25, 48)       # down-ramp close to the exit
RAMP_XY      = UP_RAMP_XY     # backward-compat alias (old code referenced RAMP_XY)

# ── Physics constants ───────────────────────────────────────────────────────
# These are THE numbers the whole simulation is built on. Everything else
# (cruise time, walk time, fitness) is derived from them.
METRES_PER_UNIT          = 0.65    # coordinate units (0-100) → real metres
CAR_SPEED_KPH            = 12.0    # typical car park driving speed, 12 km/h
CAR_SPEED_MPS            = CAR_SPEED_KPH * 1000.0 / 3600.0   # = 3.33 m/s
WALK_SPEED_MPS           = 1.35    # average adult walking speed
PARK_MANEUVER_SECONDS    = 45.0    # seconds to reverse/pull into a bay
UNPARK_MANEUVER_SECONDS  = 40.0    # seconds to leave a bay (check mirrors, reverse out, straighten)
PARK_LANE_BLOCK_SECONDS  = 45.0    # how long the lane behind a parking car is blocked
RAMP_METRES_PER_FLOOR    = 7.5     # ramp length per floor (user spec)
STAIR_SECONDS_PER_FLOOR  = 45.0    # seconds per flight (walk to stairwell + climb + walk to shop floor)

# Back-compat: old engine.py imported this — keep as a synonym for the
# metre-equivalent of 35 seconds of stair walking, i.e. 35 s × 1.35 m/s ≈ 47 m.
# New code should use STAIR_SECONDS_PER_FLOOR directly.
STAIRS_METRES_PER_FLOOR  = STAIR_SECONDS_PER_FLOOR * WALK_SPEED_MPS

# Pedestrian access points to the shops (along the building wall on the left)
SHOP_EXITS: List[ShopExit] = [
    ShopExit("To Shops · Reception", "Reception", 11, 72),
    ShopExit("To Shops · Lift / G-Units", "Lift", 11, 55),
    ShopExit("To Shops · Café",      "Café",      11, 22),
]

# Driving lane Y/X coordinates used for entry/exit paths and animation
LANE_TOP_Y    = 91     # horizontal aisle behind the top row
LANE_MID_Y    = 53     # horizontal aisle by the exit
LANE_BOT_Y    = 4      # horizontal aisle along the bottom
LANE_LEFT_X   = 28     # vertical aisle on the left
LANE_RIGHT_X  = 86     # vertical aisle on the right
AISLE_UL_X    = 38     # aisle between bays 80-87 and 79-72
AISLE_C_X     = 50     # central aisle between the two upper pairs
AISLE_UR_X    = 64     # aisle between 49-55 and 48-42


def _dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def _path_length(path: List[Tuple[float, float]]) -> float:
    if len(path) < 2:
        return 0.0
    return sum(_dist(path[i], path[i + 1]) for i in range(len(path) - 1))


def interpolate_path(path: List[Tuple[float, float]], progress: float) -> Tuple[float, float]:
    """Return (x, y) at fractional position `progress` (0-1) along `path`."""
    if not path:
        return (0.0, 0.0)
    if progress <= 0:
        return path[0]
    if progress >= 1:
        return path[-1]
    total = _path_length(path)
    target = progress * total
    accumulated = 0.0
    for i in range(len(path) - 1):
        seg = _dist(path[i], path[i + 1])
        if accumulated + seg >= target and seg > 0:
            t = (target - accumulated) / seg
            x = path[i][0] + t * (path[i + 1][0] - path[i][0])
            y = path[i][1] + t * (path[i + 1][1] - path[i][1])
            return (x, y)
        accumulated += seg
    return path[-1]


# ── Driving-path computation per bay ─────────────────────────────────────────
def _entry_path(x: float, y: float, section: str, floor: int = 0) -> List[Tuple[float, float]]:
    """Path a car drives from the entrance (or from the up-ramp on upper
    floors) to its bay."""
    p: List[Tuple[float, float]] = []
    if floor == 0:
        # Ground floor: enter through the real entrance arrow
        p.append(ENTRANCE_XY)
        p.append((ENTRANCE_XY[0], LANE_TOP_Y))
    else:
        # Upper floors: the car has already driven up the ramp, so the
        # visible path on this floor starts at the UP-ramp arrival point
        p.append(UP_RAMP_XY)
        p.append((UP_RAMP_XY[0], LANE_TOP_Y))

    if section == "TOP":
        p.append((x, LANE_TOP_Y))
        p.append((x, y))
    elif section == "RIGHT":
        p.append((LANE_RIGHT_X, LANE_TOP_Y))
        p.append((LANE_RIGHT_X, y))
        p.append((x, y))
    elif section in ("UL_L", "UL_R"):
        p.append((AISLE_UL_X, LANE_TOP_Y))
        p.append((AISLE_UL_X, y))
        p.append((x, y))
    elif section in ("UR_L", "UR_R"):
        p.append((AISLE_UR_X, LANE_TOP_Y))
        p.append((AISLE_UR_X, y))
        p.append((x, y))
    elif section in ("LL_L", "LL_R"):
        p.append((AISLE_UL_X, LANE_TOP_Y))
        p.append((AISLE_UL_X, LANE_MID_Y))
        p.append((AISLE_UL_X, y))
        p.append((x, y))
    elif section in ("LR_L", "LR_R"):
        p.append((AISLE_UR_X, LANE_TOP_Y))
        p.append((AISLE_UR_X, LANE_MID_Y))
        p.append((AISLE_UR_X, y))
        p.append((x, y))
    elif section == "LEFT":
        p.append((LANE_LEFT_X, LANE_TOP_Y))
        p.append((LANE_LEFT_X, y))
        p.append((x, y))
    elif section == "BOT_COL":
        p.append((AISLE_UL_X, LANE_TOP_Y))
        p.append((AISLE_UL_X, LANE_MID_Y))
        p.append((x, LANE_MID_Y))
        p.append((x, y))
    elif section == "SOUTH":
        p.append((LANE_LEFT_X, LANE_TOP_Y))
        p.append((LANE_LEFT_X, LANE_BOT_Y))
        p.append((x, LANE_BOT_Y))
        p.append((x, y))
    elif section == "BOTTOM":
        p.append((LANE_RIGHT_X, LANE_TOP_Y))
        p.append((LANE_RIGHT_X, LANE_BOT_Y))
        p.append((x, LANE_BOT_Y))
        p.append((x, y))
    return p


def _exit_path(x: float, y: float, section: str, floor: int = 0) -> List[Tuple[float, float]]:
    """Path a car drives from its bay to the exit (ground floor) or to the
    down-ramp (upper floors)."""
    p: List[Tuple[float, float]] = [(x, y)]

    if section == "TOP":
        p.append((x, LANE_TOP_Y))
        p.append((LANE_LEFT_X, LANE_TOP_Y))
        p.append((LANE_LEFT_X, LANE_MID_Y))
    elif section == "RIGHT":
        p.append((LANE_RIGHT_X, y))
        p.append((LANE_RIGHT_X, LANE_MID_Y))
        p.append((LANE_LEFT_X, LANE_MID_Y))
    elif section in ("UL_L", "UL_R"):
        p.append((AISLE_UL_X, y))
        p.append((AISLE_UL_X, LANE_MID_Y))
    elif section in ("UR_L", "UR_R"):
        p.append((AISLE_UR_X, y))
        p.append((AISLE_UR_X, LANE_MID_Y))
        p.append((AISLE_UL_X, LANE_MID_Y))
    elif section in ("LL_L", "LL_R"):
        p.append((AISLE_UL_X, y))
        p.append((AISLE_UL_X, LANE_MID_Y))
    elif section in ("LR_L", "LR_R"):
        p.append((AISLE_UR_X, y))
        p.append((AISLE_UR_X, LANE_MID_Y))
        p.append((AISLE_UL_X, LANE_MID_Y))
    elif section == "LEFT":
        p.append((LANE_LEFT_X, y))
        p.append((LANE_LEFT_X, LANE_MID_Y))
    elif section == "BOT_COL":
        p.append((x, LANE_MID_Y))
        p.append((LANE_LEFT_X, LANE_MID_Y))
    elif section == "SOUTH":
        p.append((x, LANE_BOT_Y))
        p.append((LANE_LEFT_X, LANE_BOT_Y))
        p.append((LANE_LEFT_X, LANE_MID_Y))
    elif section == "BOTTOM":
        p.append((x, LANE_BOT_Y))
        p.append((LANE_LEFT_X, LANE_BOT_Y))
        p.append((LANE_LEFT_X, LANE_MID_Y))

    # Ground floor cars drive out of the real exit; upper-floor cars drive
    # to the DOWN ramp (near the exit side of the building).
    if floor == 0:
        p.append(EXIT_XY)
    else:
        p.append(DOWN_RAMP_XY)
    return p


# ── Bay layout (digitised from parking plan -2.png) ──────────────────────────
def _create_floor_bays(floor_level: int) -> List[ParkingBay]:
    prefix = f"F{floor_level}"
    defs: List[Tuple[int, float, float, str]] = []  # (number, x, y, section)

    # Top row 1-11 — horizontal, just below the entrance
    for i in range(11):
        defs.append((i + 1, 33 + i * 4.7, 95, "TOP"))

    # Right column 12-31 — vertical down the right side (20 bays)
    for i in range(20):
        defs.append((12 + i, 92, 88 - i * 3.05, "RIGHT"))

    # Upper-left facing pair: 80-87 (left col) | 79-72 (right col)
    for i in range(8):
        defs.append((80 + i, 32, 87 - i * 3.7, "UL_L"))
    for i in range(8):
        defs.append((79 - i, 44, 87 - i * 3.7, "UL_R"))

    # Upper-right facing pair: 49-55 (left col) | 48-42 (right col)
    for i in range(7):
        defs.append((49 + i, 56, 87 - i * 3.7, "UR_L"))
    for i in range(7):
        defs.append((48 - i, 70, 87 - i * 3.7, "UR_R"))

    # Lower-left facing pair: 88-96 (left col) | 71-65 (right col)
    for i in range(9):
        defs.append((88 + i, 32, 49 - i * 3.5, "LL_L"))
    for i in range(7):
        defs.append((71 - i, 44, 49 - i * 3.5, "LL_R"))

    # Lower-right facing pair: 56-64 (left col) | 41-32 (right col)
    for i in range(9):
        defs.append((56 + i, 56, 49 - i * 3.5, "LR_L"))
    for i in range(10):
        defs.append((41 - i, 70, 49 - i * 3.5, "LR_R"))

    # Bottom column 97-104 — short stub under the lower section
    for i in range(8):
        defs.append((97 + i, 32, 13 - i * 2.0, "BOT_COL"))

    # Left column 109-129 (skip 128 NO PARKING) — along the building wall
    left_nums = [n for n in range(129, 108, -1) if n != 128]
    for i, num in enumerate(left_nums):
        defs.append((num, 24, 87 - i * 4.1, "LEFT"))

    # South stub 105-108 — small group near the south-west corner
    for i in range(4):
        defs.append((105 + i, 13 + i * 4.0, 1, "SOUTH"))

    # Bottom row 150-130 — runs from far left (150) to right (130)
    for i in range(21):
        defs.append((150 - i, 30 + i * 3.0, -3, "BOTTOM"))

    bays: List[ParkingBay] = []
    # Each floor above ground adds RAMP_METRES_PER_FLOOR of ramp driving
    # to both the entry and the exit (up-ramp in, down-ramp out).
    floor_ramp = floor_level * RAMP_METRES_PER_FLOOR

    for num, x, y, section in defs:
        entry = _entry_path(x, y, section, floor_level)
        exit_ = _exit_path(x, y, section, floor_level)

        # Drive distances computed along the real lane path, not straight line
        drive_in  = _path_length(entry) * METRES_PER_UNIT + floor_ramp
        drive_out = _path_length(exit_) * METRES_PER_UNIT + floor_ramp

        # Walking: nearest shop access point, straight-line (through the
        # pedestrian walkway — cars don't obstruct pedestrians)
        d_shops = min(_dist((x, y), (se.x, se.y)) for se in SHOP_EXITS) * METRES_PER_UNIT
        d_ramp  = _dist((x, y), UP_RAMP_XY) * METRES_PER_UNIT

        # Derived TIMES
        cruise_in_s  = drive_in  / CAR_SPEED_MPS + PARK_MANEUVER_SECONDS
        cruise_out_s = drive_out / CAR_SPEED_MPS + UNPARK_MANEUVER_SECONDS
        walk_s       = d_shops   / WALK_SPEED_MPS

        bay = ParkingBay(
            id=f"{prefix}-{num:03d}",
            floor=floor_level,
            number=num,
            section=section,
            row=section,                                   # back-compat
            x=x, y=y,
            distance_to_entrance=drive_in,
            distance_to_exit=drive_out,
            distance_to_ramp=d_ramp,
            distance_to_shops=d_shops,
            cruise_in_seconds=cruise_in_s,
            cruise_out_seconds=cruise_out_s,
            walk_seconds_to_shops=walk_s,
            entry_path=entry,
            exit_path=exit_,
        )
        bays.append(bay)

    # Mark the disabled bays — bay 65 has the wheelchair symbol on the plan
    for b in bays:
        if b.number in (65, 71):
            b.bay_type = BayType.DISABLED

    return bays


# ── Building the full 3-floor car park ───────────────────────────────────────
def build_carpark() -> CarPark:
    floor_configs = [
        # spend_per_hour numbers are benchmarked to UK retail averages:
        #   Supermarket ~£22/hr, Cafe ~£14/hr, Fashion ~£28/hr, Electronics ~£35/hr,
        #   Restaurant ~£25/hr, Cinema ~£18/hr (incl. concessions), Gym ~£8/hr,
        #   Services ~£15/hr. These are the "extra spend per extra minute of dwell".
        (0, "Ground Floor", [
            Shop("FreshMart Supermarket", ShopCategory.SUPERMARKET, 0, 45, [10, 11, 12, 17, 18], spend_per_hour=22.0),
            Shop("Bean & Brew Café",      ShopCategory.CAFE,        0, 30, [8, 9, 12, 13],       spend_per_hour=14.0),
            Shop("WellCare Pharmacy",     ShopCategory.PHARMACY,    0, 15, [10, 11, 14, 15],     spend_per_hour=18.0),
            Shop("Post Office",           ShopCategory.SERVICES,    0, 20, [9, 10, 11, 14, 15],  spend_per_hour=10.0),
            Shop("Dry Cleaners",          ShopCategory.SERVICES,    0, 10, [8, 9, 17, 18],       spend_per_hour=15.0),
        ]),
        (1, "First Floor", [
            Shop("Urban Style Fashion",   ShopCategory.FASHION,     1, 60, [11, 12, 13, 14, 15], spend_per_hour=28.0),
            Shop("TechZone Electronics",  ShopCategory.ELECTRONICS, 1, 40, [11, 12, 14, 15, 16], spend_per_hour=35.0),
            Shop("Gourmet Kitchen",       ShopCategory.RESTAURANT,  1, 75, [12, 13, 18, 19, 20], spend_per_hour=25.0),
            Shop("Book Corner",           ShopCategory.SERVICES,    1, 45, [10, 11, 14, 15],     spend_per_hour=16.0),
            Shop("Hair Studio",           ShopCategory.SERVICES,    1, 50, [9, 10, 11, 14, 15],  spend_per_hour=30.0),
        ]),
        (2, "Second Floor", [
            Shop("StarScreen Cinema",     ShopCategory.CINEMA,      2, 150, [14, 15, 18, 19, 20, 21], spend_per_hour=18.0),
            Shop("The Food Hall",         ShopCategory.FOOD_COURT,  2, 45,  [12, 13, 18, 19],         spend_per_hour=20.0),
            Shop("FitLife Gym",           ShopCategory.GYM,         2, 90,  [7, 8, 17, 18, 19],       spend_per_hour=8.0),
            Shop("Dental Practice",       ShopCategory.SERVICES,    2, 30,  [9, 10, 11, 14, 15],      spend_per_hour=12.0),
            Shop("Co-Working Hub",        ShopCategory.SERVICES,    2, 180, [8, 9, 10, 11, 14, 15, 16], spend_per_hour=6.0),
        ]),
    ]

    floors = []
    for level, name, shops in floor_configs:
        bays = _create_floor_bays(level)
        floors.append(Floor(level=level, name=name, bays=bays, shops=shops, capacity=len(bays)))

    return CarPark(name="WPark Car Park A", floors=floors)


# Backward-compat alias
build_baker_street_carpark = build_carpark


# ═══════════════════════════════════════════════════════════════════════════
# AESTHETIC DEMO CAR PARK — clean rectangular 3D layout with lane segments
# ═══════════════════════════════════════════════════════════════════════════
# A symmetric 40-bays-per-floor rectangular garage designed to look good
# in a 3D render.  Two pairs of bay rows per floor (top + bottom of each
# one-way lane).  Central core contains stairs, lift and mall entrance.
# Ramps are on the west side (entry) as dramatic angled structures.
#
# Layout (coordinates 0-100 × 0-100, same as Car Park A):
#
#                      NORTH (y = 100)
#     ┌──────────────────────────────────────────┐
#     │ [01][02][03][04][05][06][07][08][09][10] │   y = 85  TOP row
#     │ ═══════════════════════════════════════► │   y = 75  LANE TOP  (in: east → west)
#     │ [11][12][13][14][15][16][17][18][19][20] │   y = 65
#     │                                          │
#     │ ▲▲ UP RAMP      [ STAIRS + LIFT          │   y = 50  core
#     │ ▼▼ DOWN RAMP      MALL ENTRANCE ]        │
#     │                                          │
#     │ [21][22][23][24][25][26][27][28][29][30] │   y = 35
#     │ ◄═══════════════════════════════════════ │   y = 25  LANE BOT  (out: west → east)
#     │ [31][32][33][34][35][36][37][38][39][40] │   y = 15  BOT row
#     └──────────────────────────────────────────┘
#        ENTRY (east side, top lane, ground only)
#        EXIT  (east side, bot lane, ground only)
#                      SOUTH (y = 0)
# ═══════════════════════════════════════════════════════════════════════════
# ════════════════════════════════════════════════════════════════════════
# Demo layout (side-by-side 3-floor design)
# ════════════════════════════════════════════════════════════════════════
# Flow rules (fixed, one-way, cars can never go backwards):
#
#   ENTRATA (F0 west, TOP_PATH)
#      → F0 TOP_PATH east → bridge → F1 TOP_PATH east → bridge → F2 TOP_PATH east
#                                                                    ↓
#                                                              F2 U-TURN (east)
#                                                                    ↓
#   USCITA ← F0 BOT_PATH west ← bridge ← F1 BOT_PATH west ← bridge ← F2 BOT_PATH west
#
# Shortcuts (optional):
#   On F0 and F1, a short vertical spur at x=SHORTCUT_X connects
#   TOP_PATH to BOT_PATH on the same floor (going DOWN only).
#   Cars using the shortcut skip the rest of the loop.
#
# On F2 there is no shortcut — the "U-turn" at the east edge is what
# lets cars transition from TOP_PATH to BOT_PATH.
#
# All three floors use the same local (x, y) coordinate system.
# The pygame renderer draws them side by side as three adjacent subplots.

DEMO_WEST_X         = 5            # west wall of every floor
DEMO_EAST_X         = 95           # east wall of every floor
DEMO_TOP_PATH_Y     = 70           # TOP_PATH y-coordinate (every floor)
DEMO_BOTTOM_PATH_Y  = 25           # BOT_PATH y-coordinate (every floor)
DEMO_TOP_ROW_Y      = 88           # TOP row of bays — y-coordinate
DEMO_BOT_ROW_Y      = 12           # BOT row of bays — y-coordinate
DEMO_SHORTCUT_X     = 50           # shortcut column (F0 and F1 only)
DEMO_UTURN_X        = 95           # U-turn column (F2 only — east edge)

DEMO_ENTRANCE_XY    = (DEMO_WEST_X, DEMO_TOP_PATH_Y)      # (5, 70)
DEMO_EXIT_XY        = (DEMO_WEST_X, DEMO_BOTTOM_PATH_Y)   # (5, 25)
DEMO_SHOP_EXIT_XY   = (50, 3)      # mall entrance — south-centre of each grid
DEMO_STAIRS_XY      = (50, 3)      # stairs next to mall entrance

# ── Backwards-compat aliases (for old render code that imports them) ──
# These are no longer used by the main pygame renderer but exist so
# older imports don't crash.
DEMO_UP_RAMP_XY     = (DEMO_EAST_X, DEMO_TOP_PATH_Y)
DEMO_UP_RAMP_Y_LO   = DEMO_BOTTOM_PATH_Y
DEMO_UP_RAMP_Y_HI   = DEMO_TOP_PATH_Y
DEMO_DOWN_RAMP_XY   = (DEMO_SHORTCUT_X, DEMO_BOTTOM_PATH_Y)
DEMO_DOWN_RAMP_Y_LEN = 8
DEMO_LANE_TOP_Y     = DEMO_TOP_PATH_Y
DEMO_LANE_BOT_Y     = DEMO_BOTTOM_PATH_Y
DEMO_LANE_RIGHT_X   = DEMO_EAST_X
DEMO_RAMP_RADIUS    = 5.0
DEMO_RAMP_NORTH_POLE = (DEMO_EAST_X, DEMO_TOP_PATH_Y)
DEMO_RAMP_STUB_X    = DEMO_EAST_X
DEMO_WEST_CONN_X    = DEMO_WEST_X
DEMO_EAST_CONN_X    = DEMO_EAST_X

# ─── Lane segments (Option A queuing) ────────────────────────────────────
# Passage times are the minimum time (seconds) a car occupies each segment
# when the segment is clear.  Derived from physical length / CAR_SPEED_MPS.

# ════════════════════════════════════════════════════════════════════════
# Cellular lane layout for the demo car park
# ════════════════════════════════════════════════════════════════════════
# Uniform cell spacing of 9 world units — cells line up with bay columns
# so each TOP/BOT cell either corresponds to a bay or to the shortcut/U-turn
# junction.
#
# Cell x-positions on TOP/BOT paths:
#   cell 0  = 5  (west entry/exit)
#   cell 1  = 14  ← bay 1 / 11
#   cell 2  = 23  ← bay 2 / 12
#   cell 3  = 32  ← bay 3 / 13
#   cell 4  = 41  ← bay 4 / 14
#   cell 5  = 50  ← bay 5 / 15   AND shortcut column
#   cell 6  = 59  ← bay 6 / 16
#   cell 7  = 68  ← bay 7 / 17
#   cell 8  = 77  ← bay 8 / 18
#   cell 9  = 86  ← bay 9 / 19
#   cell 10 = 95  ← bay 10 / 20  AND U-turn column (east edge)
#
# Shortcut cells on F0/F1 are the intermediate south-bound cells between
# TOP_PATH (y=70) and BOT_PATH (y=25): 4 cells at y = [61, 52, 43, 34].
# Same for the U-turn on F2.

DEMO_CELL_STEP         = 9
DEMO_CELL_XS           = [DEMO_WEST_X + i * DEMO_CELL_STEP for i in range(11)]
# Indices of key cells
DEMO_SHORTCUT_CELL_IDX = 5     # TOP/BOT cell 5 = shortcut junction (x=50)
DEMO_UTURN_CELL_IDX    = 10    # TOP/BOT cell 10 = U-turn junction (x=95)
DEMO_ENTRY_CELL_IDX    = 0     # TOP cell 0 = ENTRATA (x=5)
DEMO_EXIT_CELL_IDX     = 10    # BOT cell 10 = USCITA (x=5, west end of BOT)
# Intermediate y-values for the vertical spurs (shortcut / u-turn)
DEMO_SPUR_YS           = [61, 52, 43, 34]    # 4 cells, step 9


def build_demo_lanes() -> Dict[str, Lane]:
    """Construct every lane in the demo car park as a list of cells."""
    lanes: Dict[str, Lane] = {}

    # TOP lanes — east-bound, cells indexed west-to-east
    for fl in range(3):
        lid = f"TOP_F{fl}"
        cells = [LaneCell(lid, i, x, DEMO_TOP_PATH_Y, fl)
                 for i, x in enumerate(DEMO_CELL_XS)]
        lanes[lid] = Lane(lid, fl, "east", cells)

    # BOT lanes — west-bound, cells indexed east-to-west
    for fl in range(3):
        lid = f"BOT_F{fl}"
        # Reverse the x positions so cell[0] is east and cell[10] is west
        reverse_xs = list(reversed(DEMO_CELL_XS))
        cells = [LaneCell(lid, i, x, DEMO_BOTTOM_PATH_Y, fl)
                 for i, x in enumerate(reverse_xs)]
        lanes[lid] = Lane(lid, fl, "west", cells)

    # Shortcut spurs — south-bound, at x=50 on floors 0 and 1
    for fl in range(2):
        lid = f"SHORTCUT_F{fl}"
        cells = [LaneCell(lid, i, DEMO_SHORTCUT_X, y, fl)
                 for i, y in enumerate(DEMO_SPUR_YS)]
        lanes[lid] = Lane(lid, fl, "south", cells)

    # U-turn spur — south-bound, at x=95 on floor 2
    lid = "UTURN_F2"
    cells = [LaneCell(lid, i, DEMO_UTURN_X, y, 2)
             for i, y in enumerate(DEMO_SPUR_YS)]
    lanes[lid] = Lane(lid, 2, "south", cells)

    return lanes


def _demo_entry_cells(row: str, floor: int, bay_cell_idx: int,
                      lanes: Dict[str, Lane]) -> List[LaneCell]:
    """Pre-compute the full cell path a car takes to reach this bay.

    Starts at TOP_F0 cell 0 (entry) and ends at the bay's approach cell.
    """
    path: List[LaneCell] = []
    if row == "TOP":
        # Drive east on TOP of each floor until the destination floor,
        # then stop at the bay's approach cell.
        for fl in range(floor + 1):
            top = lanes[f"TOP_F{fl}"]
            if fl < floor:
                path.extend(top.cells)           # full lane traversal
            else:
                path.extend(top.cells[:bay_cell_idx + 1])
        return path

    # BOT row — two routes: shortcut or F2 U-turn
    use_shortcut = (floor < 2
                    and bay_cell_idx >= DEMO_SHORTCUT_CELL_IDX)
    # bay_cell_idx is on BOT (east-to-west indexing), so >= 5 means west of x=50

    if use_shortcut:
        # TOP_F0 → ... → TOP_Ffloor[0..5] → SHORTCUT_Ffloor → BOT_Ffloor[5..bay]
        for fl in range(floor + 1):
            top = lanes[f"TOP_F{fl}"]
            if fl < floor:
                path.extend(top.cells)
            else:
                path.extend(top.cells[:DEMO_SHORTCUT_CELL_IDX + 1])
        path.extend(lanes[f"SHORTCUT_F{floor}"].cells)
        bot = lanes[f"BOT_F{floor}"]
        # Arrives at BOT cell 5 (x=50) — the shortcut-bottom junction
        for i in range(DEMO_SHORTCUT_CELL_IDX, bay_cell_idx + 1):
            path.append(bot.cells[i])
        return path

    # U-turn route: full TOP traversal across all 3 floors, UTURN_F2,
    # BOT traversal from F2 west back to destination floor.
    for fl in range(3):
        path.extend(lanes[f"TOP_F{fl}"].cells)
    path.extend(lanes["UTURN_F2"].cells)
    # BOT_F2 starts at cell 0 (x=95, east end)
    bot2 = lanes["BOT_F2"]
    if floor == 2:
        for i in range(bay_cell_idx + 1):
            path.append(bot2.cells[i])
        return path
    path.extend(bot2.cells)   # full F2 traversal
    bot1 = lanes["BOT_F1"]
    if floor == 1:
        for i in range(bay_cell_idx + 1):
            path.append(bot1.cells[i])
        return path
    path.extend(bot1.cells)
    bot0 = lanes["BOT_F0"]
    for i in range(bay_cell_idx + 1):
        path.append(bot0.cells[i])
    return path


def _demo_exit_cells(row: str, floor: int, bay_cell_idx: int,
                     lanes: Dict[str, Lane]) -> List[LaneCell]:
    """Pre-compute the full cell path from the bay's approach cell to USCITA.
    The first cell in the list IS the approach cell (the car's current
    position when it starts exiting)."""
    path: List[LaneCell] = []

    if row == "TOP":
        top = lanes[f"TOP_F{floor}"]
        if floor < 2 and bay_cell_idx <= DEMO_SHORTCUT_CELL_IDX:
            # Use this floor's shortcut — drive east from bay cell to cell 5
            for i in range(bay_cell_idx, DEMO_SHORTCUT_CELL_IDX + 1):
                path.append(top.cells[i])
            path.extend(lanes[f"SHORTCUT_F{floor}"].cells)
            # BOT_Ffloor from cell 5 west to cell 10
            bot = lanes[f"BOT_F{floor}"]
            for i in range(DEMO_SHORTCUT_CELL_IDX, len(bot.cells)):
                path.append(bot.cells[i])
            # Bridges down to F0
            for fl in range(floor - 1, -1, -1):
                path.extend(lanes[f"BOT_F{fl}"].cells)
            return path
        # Must go east to next floor's shortcut or F2 U-turn
        for i in range(bay_cell_idx, len(top.cells)):
            path.append(top.cells[i])
        nf = floor + 1
        while nf < 2:
            nf_top = lanes[f"TOP_F{nf}"]
            for i in range(DEMO_SHORTCUT_CELL_IDX + 1):
                path.append(nf_top.cells[i])
            path.extend(lanes[f"SHORTCUT_F{nf}"].cells)
            bot_nf = lanes[f"BOT_F{nf}"]
            for i in range(DEMO_SHORTCUT_CELL_IDX, len(bot_nf.cells)):
                path.append(bot_nf.cells[i])
            for fl in range(nf - 1, -1, -1):
                path.extend(lanes[f"BOT_F{fl}"].cells)
            return path
        # nf == 2, use U-turn
        path.extend(lanes["TOP_F2"].cells)
        path.extend(lanes["UTURN_F2"].cells)
        path.extend(lanes["BOT_F2"].cells)
        path.extend(lanes["BOT_F1"].cells)
        path.extend(lanes["BOT_F0"].cells)
        return path

    # BOT row — just drive west from bay cell to west end, then bridges to F0
    bot = lanes[f"BOT_F{floor}"]
    for i in range(bay_cell_idx, len(bot.cells)):
        path.append(bot.cells[i])
    for fl in range(floor - 1, -1, -1):
        path.extend(lanes[f"BOT_F{fl}"].cells)
    return path


DEMO_LANE_SEGMENTS: Dict[str, LaneSegment] = {
    # Per-floor TOP_PATH (east-bound) and BOT_PATH (west-bound)
    "TOP_F0":            LaneSegment("TOP_F0",           18.0, 58.5),
    "TOP_F1":            LaneSegment("TOP_F1",           18.0, 58.5),
    "TOP_F2":            LaneSegment("TOP_F2",           18.0, 58.5),
    "BOT_F0":            LaneSegment("BOT_F0",           18.0, 58.5),
    "BOT_F1":            LaneSegment("BOT_F1",           18.0, 58.5),
    "BOT_F2":            LaneSegment("BOT_F2",           18.0, 58.5),
    # Shortcut spurs on F0 and F1 (TOP → BOT, going south)
    "SHORTCUT_F0":       LaneSegment("SHORTCUT_F0",       8.0, 29.25),
    "SHORTCUT_F1":       LaneSegment("SHORTCUT_F1",       8.0, 29.25),
    # U-turn on F2 (east-edge spur, TOP → BOT, going south)
    "UTURN_F2":          LaneSegment("UTURN_F2",          8.0, 29.25),
    # Bridges between adjacent floors — represent the physical ramp.
    # Each bridge is a short connector that a car crosses while going
    # east on TOP_PATH or west on BOT_PATH.  Ground-distance ≈ 7.5m.
    "BRIDGE_TOP_0_1":    LaneSegment("BRIDGE_TOP_0_1",    3.0, 7.5),
    "BRIDGE_TOP_1_2":    LaneSegment("BRIDGE_TOP_1_2",    3.0, 7.5),
    "BRIDGE_BOT_2_1":    LaneSegment("BRIDGE_BOT_2_1",    3.0, 7.5),
    "BRIDGE_BOT_1_0":    LaneSegment("BRIDGE_BOT_1_0",    3.0, 7.5),
}


def _segments_entry_for(row: str, floor_level: int, **kwargs) -> List[str]:
    """Entry segments for the side-by-side 3-floor demo.

    row: "TOP" or "BOT"  (top row = y=88, bottom row = y=12)
    floor_level: 0, 1, 2
    kwargs["bay_x"]: x-coordinate of the bay — decides shortcut vs U-turn
    """
    bay_x = kwargs.get("bay_x", 0.0)

    segs: List[str] = []

    # TOP-row bay — drive along TOP_PATH through floors 0..floor_level
    if row == "TOP":
        segs.append("TOP_F0")
        for lvl in range(1, floor_level + 1):
            segs.append(f"BRIDGE_TOP_{lvl-1}_{lvl}")
            segs.append(f"TOP_F{lvl}")
        return segs

    # BOT-row bay — need to decide access strategy
    if floor_level < 2 and bay_x < DEMO_SHORTCUT_X:
        # Shortcut route: drive TOP through floors 0..floor_level,
        # take the shortcut on this floor, arrive on BOT_PATH, drive west.
        segs.append("TOP_F0")
        for lvl in range(1, floor_level + 1):
            segs.append(f"BRIDGE_TOP_{lvl-1}_{lvl}")
            segs.append(f"TOP_F{lvl}")
        segs.append(f"SHORTCUT_F{floor_level}")
        segs.append(f"BOT_F{floor_level}")
        return segs

    # Long route: TOP all the way to F2 U-turn, then BOT back to destination
    segs.append("TOP_F0")
    segs.append("BRIDGE_TOP_0_1")
    segs.append("TOP_F1")
    segs.append("BRIDGE_TOP_1_2")
    segs.append("TOP_F2")
    segs.append("UTURN_F2")
    # Drive BOT_PATH west from F2 down to destination floor
    segs.append("BOT_F2")
    if floor_level <= 1:
        segs.append("BRIDGE_BOT_2_1")
        segs.append("BOT_F1")
    if floor_level == 0:
        segs.append("BRIDGE_BOT_1_0")
        segs.append("BOT_F0")
    return segs


def _segments_exit_for(row: str, floor_level: int, **kwargs) -> List[str]:
    """Exit segments: from the bay's lane back to USCITA on F0.
    USCITA is at the west end of F0 BOT_PATH, so every exiting car
    ends up on BOT_F0 heading west.
    """
    bay_x = kwargs.get("bay_x", 0.0)
    segs: List[str] = []

    if row == "TOP":
        # Car is on TOP_PATH of its floor; must reach BOT_PATH somehow
        # and then drive west to USCITA.
        if floor_level < 2 and bay_x < DEMO_SHORTCUT_X:
            # Can use the same-floor shortcut (east of bay, then down)
            # But the car already passed its bay going east then parked,
            # so on exit it continues east to the shortcut, then down.
            segs.append(f"TOP_F{floor_level}")
            segs.append(f"SHORTCUT_F{floor_level}")
            segs.append(f"BOT_F{floor_level}")
        else:
            # Must reach F2 U-turn (drive east through remaining floors)
            segs.append(f"TOP_F{floor_level}")
            for lvl in range(floor_level + 1, 3):
                segs.append(f"BRIDGE_TOP_{lvl-1}_{lvl}")
                segs.append(f"TOP_F{lvl}")
            segs.append("UTURN_F2")
            segs.append("BOT_F2")
            # Descend BOT_PATH back to F0
            for lvl in range(2, 0, -1):
                segs.append(f"BRIDGE_BOT_{lvl}_{lvl-1}")
                segs.append(f"BOT_F{lvl-1}")
            return segs
    else:
        # BOT-row bay — car is already on BOT_PATH of its floor heading west
        segs.append(f"BOT_F{floor_level}")

    # Drive west on BOT_PATH from current floor down to F0
    for lvl in range(floor_level, 0, -1):
        segs.append(f"BRIDGE_BOT_{lvl}_{lvl-1}")
        segs.append(f"BOT_F{lvl-1}")
    return segs


def _create_demo_floor_bays(floor_level: int,
                            lanes: Dict[str, Lane]) -> List[ParkingBay]:
    """20 bays per floor for the cellular-automaton demo.

    TOP row: 10 bays at y=88, aligned with TOP_PATH cells 1..10.
    BOT row: 10 bays at y=12, aligned with BOT_PATH cells 1..10.
    (Bays at cell 0 don't exist — that cell is reserved for entry/exit.)

    Each bay has:
      - approach_cell: the lane cell right next to it on the main path
      - entry_path_cells: full sequence of cells from TOP_F0 cell 0 → approach
      - exit_path_cells: from approach → BOT_F0 cell 10 (USCITA)
    """
    prefix = f"F{floor_level}"
    bays: List[ParkingBay] = []
    floor_ramp = floor_level * RAMP_METRES_PER_FLOOR

    # TOP row: 10 bays aligned with TOP cells 1..10
    #   bay_number 1..10, approach_cell = TOP_F{floor_level}.cells[1..10]
    # BOT row: 10 bays aligned with BOT cells 1..10
    #   bay_number 11..20, approach_cell = BOT_F{floor_level}.cells[1..10]

    # Bays 1-10 on TOP row: bay 1 = west-most (TOP cell 1, x=14),
    #                       bay 10 = east-most (TOP cell 10, x=95).
    # Bays 11-20 on BOT row: bay 11 = west-most (x=14, BOT cell 9),
    #                        bay 20 = east-most (x=95, BOT cell 0).
    # So BOT bay N → BOT cell (20 - N).
    top_iter = [("TOP", n, n) for n in range(1, 11)]           # (row, num, cell_idx)
    bot_iter = [("BOT", n, 20 - n) for n in range(11, 21)]
    for row_id, num, cell_idx in top_iter + bot_iter:
        if row_id == "TOP":
            lane = lanes[f"TOP_F{floor_level}"]
            by = DEMO_TOP_ROW_Y
        else:
            lane = lanes[f"BOT_F{floor_level}"]
            by = DEMO_BOT_ROW_Y
        approach_cell = lane.cells[cell_idx]
        x = approach_cell.x
        y = by
        # Pre-compute the full cell paths for entry and exit
        entry_cells = _demo_entry_cells(row_id, floor_level, cell_idx, lanes)
        exit_cells  = _demo_exit_cells(row_id,  floor_level, cell_idx, lanes)

        # Distances derived from cell count (one cell ≈ 9u ≈ 5.85m)
        CELL_METRES = DEMO_CELL_STEP * METRES_PER_UNIT
        drive_in  = len(entry_cells) * CELL_METRES
        drive_out = len(exit_cells)  * CELL_METRES
        d_shops = _dist((x, y), DEMO_SHOP_EXIT_XY) * METRES_PER_UNIT
        cruise_in_s  = drive_in  / CAR_SPEED_MPS + PARK_MANEUVER_SECONDS
        cruise_out_s = drive_out / CAR_SPEED_MPS + UNPARK_MANEUVER_SECONDS
        walk_s       = d_shops   / WALK_SPEED_MPS

        # Legacy journey/path lists for back-compat with the Streamlit
        # renderer (the pygame demo uses cell paths directly).
        entry_journey = [(c.floor, c.x, c.y) for c in entry_cells]
        exit_journey  = [(c.floor, c.x, c.y) for c in exit_cells]
        entry_path = [(c.x, c.y) for c in entry_cells if c.floor == floor_level]
        exit_path  = [(c.x, c.y) for c in exit_cells  if c.floor == floor_level]

        bay = ParkingBay(
            id=f"{prefix}-{num:03d}",
            floor=floor_level,
            number=num,
            section=row_id,
            row=row_id,
            x=x, y=y,
            distance_to_entrance=drive_in,
            distance_to_exit=drive_out,
            distance_to_ramp=0.0,
            distance_to_shops=d_shops,
            cruise_in_seconds=cruise_in_s,
            cruise_out_seconds=cruise_out_s,
            walk_seconds_to_shops=walk_s,
            entry_path=entry_path,
            exit_path=exit_path,
            entry_journey=entry_journey,
            exit_journey=exit_journey,
            entry_segments=[],
            exit_segments=[],
            approach_cell=approach_cell,
            entry_path_cells=entry_cells,
            exit_path_cells=exit_cells,
        )
        bays.append(bay)
    return bays


def build_demo_carpark() -> CarPark:
    """Side-by-side 3-floor demo: 20 bays/floor × 3 floors = 60 bays.
    Each floor is its own grid with a TOP row and a BOT row of 10 bays.
    Floors connect via visible bridges at the edges, representing ramps.
    Flow is strictly one-way (east on TOP, west on BOT)."""
    floor_configs = [
        (0, "Ground", [
            Shop("Demo Supermarket", ShopCategory.SUPERMARKET, 0, 35,
                 [9, 10, 11, 12, 17, 18], spend_per_hour=22.0),
        ]),
        (1, "First", [
            Shop("Demo Fashion Store", ShopCategory.FASHION, 1, 55,
                 [11, 12, 13, 14, 15], spend_per_hour=28.0),
        ]),
        (2, "Second", [
            Shop("Demo Cinema", ShopCategory.CINEMA, 2, 120,
                 [14, 15, 18, 19, 20], spend_per_hour=18.0),
        ]),
    ]
    # Build the cellular lane network once — shared across all floors.
    lanes = build_demo_lanes()
    floors = []
    for level, name, shops in floor_configs:
        bays = _create_demo_floor_bays(level, lanes)
        floors.append(Floor(level=level, name=name, bays=bays, shops=shops, capacity=len(bays)))
    cp = CarPark(name="WPark Demo Garage (60 bays)", floors=floors)
    cp.lanes = lanes      # stored on the car park for engine/renderer access
    cp.is_demo = True
    return cp


def build_scaled_carpark(bays_per_row: int = 20) -> CarPark:
    """Scaled version of the demo — same 3-floor layout, same physics,
    same cellular model, just more bays per row.

    bays_per_row=10 → 60 bays (identical to build_demo_carpark)
    bays_per_row=20 → 120 bays
    bays_per_row=30 → 180 bays

    The existing build_demo_carpark() is NOT touched.
    """
    n = bays_per_row
    # Scale cell positions to fit in world x=[5..95]
    total_cells = n + 1   # n bay cells + 1 entry/exit cell at x=5
    step = 90.0 / n       # spread evenly across x=5..95
    cell_xs = [5.0 + i * step for i in range(total_cells)]

    shortcut_idx = n // 2   # midpoint
    uturn_idx = n            # last cell (east edge)

    shortcut_x = cell_xs[shortcut_idx]
    uturn_x = cell_xs[uturn_idx]

    # Build lanes using the scaled cell positions
    lanes: Dict[str, Lane] = {}
    for fl in range(3):
        lid = f"TOP_F{fl}"
        cells = [LaneCell(lid, i, x, DEMO_TOP_PATH_Y, fl)
                 for i, x in enumerate(cell_xs)]
        lanes[lid] = Lane(lid, fl, "east", cells)

    for fl in range(3):
        lid = f"BOT_F{fl}"
        rev_xs = list(reversed(cell_xs))
        cells = [LaneCell(lid, i, x, DEMO_BOTTOM_PATH_Y, fl)
                 for i, x in enumerate(rev_xs)]
        lanes[lid] = Lane(lid, fl, "west", cells)

    for fl in range(2):
        lid = f"SHORTCUT_F{fl}"
        cells = [LaneCell(lid, i, shortcut_x, y, fl)
                 for i, y in enumerate(DEMO_SPUR_YS)]
        lanes[lid] = Lane(lid, fl, "south", cells)

    lid = "UTURN_F2"
    cells = [LaneCell(lid, i, uturn_x, y, 2)
             for i, y in enumerate(DEMO_SPUR_YS)]
    lanes[lid] = Lane(lid, 2, "south", cells)

    # Build bays — same logic as _create_demo_floor_bays but parametric
    def _make_bays(floor_level: int) -> List[ParkingBay]:
        prefix = f"F{floor_level}"
        bays_list: List[ParkingBay] = []
        floor_ramp = floor_level * RAMP_METRES_PER_FLOOR

        # TOP row: bays 1..n aligned with TOP cells 1..n
        top_iter = [("TOP", num, num) for num in range(1, n + 1)]
        # BOT row: bays (n+1)..(2n) aligned with BOT cells
        bot_iter = [("BOT", num, 2 * n - num) for num in range(n + 1, 2 * n + 1)]

        for row_id, num, cell_idx in top_iter + bot_iter:
            if row_id == "TOP":
                lane = lanes[f"TOP_F{floor_level}"]
                by = DEMO_TOP_ROW_Y
            else:
                lane = lanes[f"BOT_F{floor_level}"]
                by = DEMO_BOT_ROW_Y

            approach = lane.cells[cell_idx]
            bx = approach.x

            # Entry path cells
            entry_cells = _scaled_entry_cells(
                row_id, floor_level, cell_idx, lanes,
                shortcut_idx, n)
            # Exit path cells
            exit_cells = _scaled_exit_cells(
                row_id, floor_level, cell_idx, lanes,
                shortcut_idx, n)

            # Distances
            CELL_METRES = step * METRES_PER_UNIT
            n_entry = len(entry_cells)
            n_exit = len(exit_cells)
            drive_in = n_entry * CELL_METRES + floor_ramp
            drive_out = n_exit * CELL_METRES + floor_ramp
            d_shops = _dist((bx, by), DEMO_SHOP_EXIT_XY) * METRES_PER_UNIT

            cruise_in_s = drive_in / CAR_SPEED_MPS + PARK_MANEUVER_SECONDS
            cruise_out_s = drive_out / CAR_SPEED_MPS + UNPARK_MANEUVER_SECONDS
            walk_s = d_shops / WALK_SPEED_MPS

            entry_segs = _segments_entry_for(row_id, floor_level,
                                             shortcut_idx=shortcut_idx)
            exit_segs = _segments_exit_for(row_id, floor_level,
                                           shortcut_idx=shortcut_idx)

            # entry_journey for pygame renderer (backward compat)
            if row_id == "TOP":
                entry_journey = [(c.floor, c.x, c.y) for c in entry_cells]
            else:
                entry_journey = [(c.floor, c.x, c.y) for c in entry_cells]

            bay = ParkingBay(
                id=f"{prefix}-{num:03d}",
                floor=floor_level,
                number=num,
                section=row_id,
                row=row_id,
                x=bx, y=by,
                distance_to_entrance=drive_in,
                distance_to_exit=drive_out,
                distance_to_ramp=0.0,
                distance_to_shops=d_shops,
                cruise_in_seconds=cruise_in_s,
                cruise_out_seconds=cruise_out_s,
                walk_seconds_to_shops=walk_s,
                entry_path=[(c.x, c.y) for c in entry_cells],
                exit_path=[(c.x, c.y) for c in exit_cells],
                entry_segments=entry_segs,
                exit_segments=exit_segs,
                approach_cell=approach,
                entry_path_cells=entry_cells,
                exit_path_cells=exit_cells,
                entry_journey=entry_journey,
            )
            bays_list.append(bay)
        return bays_list

    floor_configs = [
        (0, "Ground", [
            Shop("FreshMart Supermarket", ShopCategory.SUPERMARKET, 0, 35,
                 [9, 10, 11, 12, 17, 18], spend_per_hour=22.0),
        ]),
        (1, "First", [
            Shop("Urban Fashion", ShopCategory.FASHION, 1, 55,
                 [11, 12, 13, 14, 15], spend_per_hour=28.0),
        ]),
        (2, "Second", [
            Shop("Stellar Cinema", ShopCategory.CINEMA, 2, 120,
                 [14, 15, 18, 19, 20], spend_per_hour=18.0),
        ]),
    ]

    floors = []
    for level, name, shops in floor_configs:
        bays = _make_bays(level)
        floors.append(Floor(level=level, name=name, bays=bays,
                            shops=shops, capacity=len(bays)))

    total = sum(f.capacity for f in floors)
    cp = CarPark(name=f"WPark Scaled ({total} bays)", floors=floors)
    cp.lanes = lanes
    cp.is_demo = True
    return cp


def _scaled_entry_cells(row, floor, cell_idx, lanes, shortcut_idx, n_bays):
    """Same logic as _demo_entry_cells but with parametric shortcut/uturn."""
    path: List[LaneCell] = []
    if row == "TOP":
        for fl in range(floor + 1):
            top = lanes[f"TOP_F{fl}"]
            if fl < floor:
                path.extend(top.cells)
            else:
                path.extend(top.cells[:cell_idx + 1])
        return path

    use_shortcut = (floor < 2 and cell_idx >= shortcut_idx)
    if use_shortcut:
        for fl in range(floor + 1):
            top = lanes[f"TOP_F{fl}"]
            if fl < floor:
                path.extend(top.cells)
            else:
                path.extend(top.cells[:shortcut_idx + 1])
        path.extend(lanes[f"SHORTCUT_F{floor}"].cells)
        bot = lanes[f"BOT_F{floor}"]
        for i in range(shortcut_idx, cell_idx + 1):
            path.append(bot.cells[i])
        return path

    for fl in range(3):
        path.extend(lanes[f"TOP_F{fl}"].cells)
    path.extend(lanes["UTURN_F2"].cells)
    bot2 = lanes["BOT_F2"]
    if floor == 2:
        for i in range(cell_idx + 1):
            path.append(bot2.cells[i])
        return path
    path.extend(bot2.cells)
    bot1 = lanes["BOT_F1"]
    if floor == 1:
        for i in range(cell_idx + 1):
            path.append(bot1.cells[i])
        return path
    path.extend(bot1.cells)
    bot0 = lanes["BOT_F0"]
    for i in range(cell_idx + 1):
        path.append(bot0.cells[i])
    return path


def _scaled_exit_cells(row, floor, cell_idx, lanes, shortcut_idx, n_bays):
    """Same logic as _demo_exit_cells but parametric."""
    path: List[LaneCell] = []
    if row == "TOP":
        top = lanes[f"TOP_F{floor}"]
        if floor < 2 and cell_idx <= shortcut_idx:
            for i in range(cell_idx, shortcut_idx + 1):
                path.append(top.cells[i])
            path.extend(lanes[f"SHORTCUT_F{floor}"].cells)
            bot = lanes[f"BOT_F{floor}"]
            for i in range(shortcut_idx, len(bot.cells)):
                path.append(bot.cells[i])
            for fl in range(floor - 1, -1, -1):
                path.extend(lanes[f"BOT_F{fl}"].cells)
            return path
        for i in range(cell_idx, len(top.cells)):
            path.append(top.cells[i])
        nf = floor + 1
        while nf < 2:
            nf_top = lanes[f"TOP_F{nf}"]
            for i in range(shortcut_idx + 1):
                path.append(nf_top.cells[i])
            path.extend(lanes[f"SHORTCUT_F{nf}"].cells)
            bot_nf = lanes[f"BOT_F{nf}"]
            for i in range(shortcut_idx, len(bot_nf.cells)):
                path.append(bot_nf.cells[i])
            for fl in range(nf - 1, -1, -1):
                path.extend(lanes[f"BOT_F{fl}"].cells)
            return path
        path.extend(lanes["TOP_F2"].cells)
        path.extend(lanes["UTURN_F2"].cells)
        path.extend(lanes["BOT_F2"].cells)
        path.extend(lanes["BOT_F1"].cells)
        path.extend(lanes["BOT_F0"].cells)
        return path

    bot = lanes[f"BOT_F{floor}"]
    for i in range(cell_idx, len(bot.cells)):
        path.append(bot.cells[i])
    for fl in range(floor - 1, -1, -1):
        path.extend(lanes[f"BOT_F{fl}"].cells)
    return path


def is_bay_blocked(bay: ParkingBay, floor: Floor) -> bool:
    """A car is blocked if both immediate neighbours in the same row of the same section are occupied."""
    if bay.status != BayStatus.OCCUPIED:
        return False
    same_section = sorted(
        [b for b in floor.bays if b.section == bay.section],
        key=lambda b: (b.y, b.x),
    )
    idx = next((i for i, b in enumerate(same_section) if b.id == bay.id), None)
    if idx is None:
        return False
    left  = idx > 0 and same_section[idx - 1].status == BayStatus.OCCUPIED
    right = idx < len(same_section) - 1 and same_section[idx + 1].status == BayStatus.OCCUPIED
    return left and right


if __name__ == "__main__":
    cp = build_carpark()
    print(f"{cp.name}: {cp.total_capacity} spaces ({len(cp.floors)} floors)")
    for f in cp.floors:
        print(f"  {f.name}: {f.capacity} spaces, {len(f.shops)} shops")
