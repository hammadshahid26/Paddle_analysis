# """
# court_zones.py
# --------------
# Per-frame court-zone classification for padel doubles.

# Zones (per client spec):
#     'net'      = within net-zone radius of the net line
#     'baseline' = behind net-zone radius (back of the court)

# Net Y position is auto-calibrated from the court polygon (the net sits
# ~20% down from the polygon's top because of perspective: far half
# compresses into a thin image strip, near half occupies most of the
# screen).

# Net-zone radius is per-team and proportional to that team's court depth
# in pixels (far team = narrow band, near team = wide band).
# """
# from collections import defaultdict
# from typing import List, Dict, Optional, Tuple

# import numpy as np


# # Padel court depth split — far team has 25% of polygon vertical span,
# # near team has 75% (perspective compressionss). Net at 20% down.
# NET_Y_RATIO_OF_POLYGON = 0.20

# # Net-zone radius as fraction of each team's court depth.
# # 0.40 means: net zone covers the half of the court closest to the net.
# # Smaller value = net zone narrower.
# NET_ZONE_RATIO = 0.30


# def calibrate_net_and_zones(floor_polygon: np.ndarray, frame_height: int = 720
#                             ) -> Dict[str, float]:
#     """
#     From the floor polygon, compute:
#       - net_y          : Y pixel of the net line
#       - far_back_y     : Y pixel of the far baseline (top of polygon)
#       - near_back_y    : Y pixel of the near baseline (bottom of polygon)
#       - far_net_radius : net-zone radius for far team (pixels)
#       - near_net_radius: net-zone radius for near team (pixels)
#     """
#     bp = np.asarray(floor_polygon, dtype=np.int32)
#     poly_top    = float(bp[:, 1].min())
#     poly_bottom = float(bp[:, 1].max())
#     poly_depth  = poly_bottom - poly_top

#     net_y = poly_top + NET_Y_RATIO_OF_POLYGON * poly_depth

#     far_team_depth  = net_y - poly_top
#     near_team_depth = poly_bottom - net_y

#     far_net_radius  = far_team_depth  * NET_ZONE_RATIO
#     near_net_radius = near_team_depth * NET_ZONE_RATIO

#     return {
#         'net_y'           : net_y,
#         'far_back_y'      : poly_top,
#         'near_back_y'     : poly_bottom,
#         'far_net_radius'  : far_net_radius,
#         'near_net_radius' : near_net_radius,
#     }


# def classify_player_zone(box: Optional[np.ndarray],
#                          seat_idx: int,
#                          calib: Dict[str, float]) -> Optional[str]:
#     """
#     Returns 'net' or 'baseline' for the given player bbox.
#     Returns None if box is None.

#     Zone is determined by feet (bbox bottom-center) Y position vs net_y,
#     using the team's specific net-zone radius.
#     """
#     if box is None:
#         return None
#     feet_y = float(box[3])

#     # Far team: P1 (idx 0), P2 (idx 1)
#     if seat_idx in (0, 1):
#         # Far players: in net zone if their feet are CLOSER to net than far_net_radius
#         # Feet Y between (net_y - far_net_radius) and net_y → net zone
#         if feet_y >= (calib['net_y'] - calib['far_net_radius']):
#             return 'net'
#         else:
#             return 'baseline'
#     # Near team: P3 (idx 2), P4 (idx 3)
#     else:
#         # Near players: in net zone if feet between net_y and (net_y + near_net_radius)
#         if feet_y <= (calib['net_y'] + calib['near_net_radius']):
#             return 'net'
#         else:
#             return 'baseline'


# class ZoneTracker:
#     """
#     Per-player zone tracking across the full video.
#     Computes time-in-zone and transition counts.
#     """

#     def __init__(self, fps: float, n_players: int = 4):
#         self.fps = fps
#         self.n_players = n_players
#         # frame_zones[player_idx] = list of zones per frame ('net'|'baseline'|None)
#         self.frame_zones: List[List[Optional[str]]] = []

#     def update(self, frame_zones_for_all_players: List[Optional[str]]):
#         self.frame_zones.append(list(frame_zones_for_all_players))

#     def per_player_stats(self) -> List[Dict]:
#         """
#         For each player, returns dict:
#           frames_in_net, frames_in_baseline, frames_unknown,
#           time_in_net_sec, time_in_baseline_sec,
#           pct_in_net, pct_in_baseline,
#           n_transitions
#         """
#         stats = []
#         for p in range(self.n_players):
#             zones = [fz[p] if p < len(fz) else None for fz in self.frame_zones]
#             n_net      = sum(1 for z in zones if z == 'net')
#             n_baseline = sum(1 for z in zones if z == 'baseline')
#             n_unknown  = sum(1 for z in zones if z is None)
#             n_visible  = n_net + n_baseline

#             t_net      = n_net      / self.fps
#             t_baseline = n_baseline / self.fps

#             pct_net      = (n_net      / n_visible * 100) if n_visible else 0
#             pct_baseline = (n_baseline / n_visible * 100) if n_visible else 0

#             # transitions: count how many times zone changed (ignoring None)
#             transitions = 0
#             prev = None
#             for z in zones:
#                 if z is None: continue
#                 if prev is not None and z != prev:
#                     transitions += 1
#                 prev = z

#             stats.append({
#                 'player_idx'         : p,
#                 'frames_in_net'      : n_net,
#                 'frames_in_baseline' : n_baseline,
#                 'frames_unknown'     : n_unknown,
#                 'time_in_net_sec'    : round(t_net, 1),
#                 'time_in_baseline_sec': round(t_baseline, 1),
#                 'pct_in_net'         : round(pct_net, 1),
#                 'pct_in_baseline'    : round(pct_baseline, 1),
#                 'n_transitions'      : transitions,
#             })
#         return stats


# def export_zones_csv(per_player_stats, seat_labels, path):
#     import csv
#     with open(path, 'w', newline='') as f:
#         w = csv.writer(f)
#         w.writerow(['player', 'seat', 'time_in_net_sec', 'time_in_baseline_sec',
#                     'pct_in_net', 'pct_in_baseline', 'n_transitions'])
#         for s in per_player_stats:
#             p = s['player_idx']
#             w.writerow([
#                 f'P{p+1}', seat_labels[p],
#                 s['time_in_net_sec'], s['time_in_baseline_sec'],
#                 s['pct_in_net'], s['pct_in_baseline'],
#                 s['n_transitions'],
#             ])


# def print_zone_summary(per_player_stats, seat_labels, n_players=4):
#     print('\n------------------ COURT ZONE SUMMARY ----------------')
#     for s in per_player_stats:
#         p = s['player_idx']
#         net_t = s['time_in_net_sec']
#         bl_t  = s['time_in_baseline_sec']
#         net_p = s['pct_in_net']
#         trans = s['n_transitions']
#         print(f'  {seat_labels[p]:14s} | '
#               f'net: {net_t:5.1f}s ({net_p:4.1f}%)  '
#               f'baseline: {bl_t:5.1f}s  '
#               f'transitions: {trans}')
#     print('-------------------------------------------------------\n')


# # Visualization colors for zones
# ZONE_COLORS = {
#     'net'     : (0, 255, 200),     # cyan-ish — aggressive position
#     'baseline': (180, 100, 255),   # purple — defensive position
#     None      : (120, 120, 120),
# }

"""
court_zones.py
--------------
Per-frame court-zone classification for padel doubles.

Zones (per client spec):
    'net'      = between the net and the service line on that team's side
    'baseline' = between the service line and back wall on that team's side

Primary calibration: detect actual court lines (white-line detection in
court_mask.detect_court_lines). If detection succeeds, zones are based on
real court geometry — camera-agnostic.

Fallback calibration: ratio-based using the polygon (used if line detection
fails or returns incomplete data).
"""
from collections import defaultdict
from typing import List, Dict, Optional

import numpy as np


# Fallback ratios (used only when line detection fails)
NET_Y_RATIO_OF_POLYGON = 0.20
NET_ZONE_RATIO         = 0.30


def calibrate_net_and_zones(
    floor_polygon: np.ndarray,
    frame_height: int = 720,
    detected_lines: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """
    Build the zone calibration. If `detected_lines` is provided AND has
    at least net_y + one service line, use real geometry. Otherwise use
    the ratio fallback.

    Returns dict with:
      net_y, far_back_y, near_back_y,
      far_net_radius, near_net_radius,
      method  ('lines' or 'ratios') for diagnostics
    """
    bp = np.asarray(floor_polygon, dtype=np.int32)
    poly_top    = float(bp[:, 1].min())
    poly_bottom = float(bp[:, 1].max())

    # Try real-geometry calibration first
    if detected_lines is not None:
        net_y = detected_lines.get('net_y')
        far_sl = detected_lines.get('far_service_y')
        near_sl = detected_lines.get('near_service_y')

        if net_y is not None and (far_sl is not None or near_sl is not None):
            # Far net radius = distance from net to far service line.
            # If far service not detected, fall back to mirroring near.
            if far_sl is not None:
                far_net_radius = net_y - far_sl
            elif near_sl is not None:
                # Mirror via court symmetry: scale near radius by
                # the ratio of far-half depth to near-half depth.
                near_net_radius_proxy = near_sl - net_y
                far_half = net_y - poly_top
                near_half = poly_bottom - net_y
                if near_half > 0:
                    far_net_radius = near_net_radius_proxy * (far_half / near_half)
                else:
                    far_net_radius = (net_y - poly_top) * 0.5
            else:
                far_net_radius = (net_y - poly_top) * 0.5

            if near_sl is not None:
                near_net_radius = near_sl - net_y
            elif far_sl is not None:
                far_net_radius_proxy = net_y - far_sl
                far_half = net_y - poly_top
                near_half = poly_bottom - net_y
                if far_half > 0:
                    near_net_radius = far_net_radius_proxy * (near_half / far_half)
                else:
                    near_net_radius = (poly_bottom - net_y) * 0.5
            else:
                near_net_radius = (poly_bottom - net_y) * 0.5

            return {
                'net_y'           : float(net_y),
                'far_back_y'      : poly_top,
                'near_back_y'     : poly_bottom,
                'far_net_radius'  : float(far_net_radius),
                'near_net_radius' : float(near_net_radius),
                'far_service_y'   : float(far_sl) if far_sl is not None else None,
                'near_service_y'  : float(near_sl) if near_sl is not None else None,
                'method'          : 'lines',
            }

    # Fallback: ratio-based
    poly_depth = poly_bottom - poly_top
    net_y = poly_top + NET_Y_RATIO_OF_POLYGON * poly_depth
    far_team_depth  = net_y - poly_top
    near_team_depth = poly_bottom - net_y
    far_net_radius  = far_team_depth  * NET_ZONE_RATIO
    near_net_radius = near_team_depth * NET_ZONE_RATIO

    return {
        'net_y'           : net_y,
        'far_back_y'      : poly_top,
        'near_back_y'     : poly_bottom,
        'far_net_radius'  : far_net_radius,
        'near_net_radius' : near_net_radius,
        'far_service_y'   : None,
        'near_service_y'  : None,
        'method'          : 'ratios',
    }


def classify_player_zone(box: Optional[np.ndarray],
                         seat_idx: int,
                         calib: Dict[str, float]) -> Optional[str]:
    if box is None:
        return None
    feet_y = float(box[3])

    if seat_idx in (0, 1):  # Far team
        if feet_y >= (calib['net_y'] - calib['far_net_radius']):
            return 'net'
        else:
            return 'baseline'
    else:  # Near team
        if feet_y <= (calib['net_y'] + calib['near_net_radius']):
            return 'net'
        else:
            return 'baseline'


class ZoneTracker:
    def __init__(self, fps: float, n_players: int = 4):
        self.fps = fps
        self.n_players = n_players
        self.frame_zones: List[List[Optional[str]]] = []

    def update(self, frame_zones_for_all_players: List[Optional[str]]):
        self.frame_zones.append(list(frame_zones_for_all_players))

    def per_player_stats(self) -> List[Dict]:
        stats = []
        for p in range(self.n_players):
            zones = [fz[p] if p < len(fz) else None for fz in self.frame_zones]
            n_net      = sum(1 for z in zones if z == 'net')
            n_baseline = sum(1 for z in zones if z == 'baseline')
            n_unknown  = sum(1 for z in zones if z is None)
            n_visible  = n_net + n_baseline
            t_net      = n_net      / self.fps
            t_baseline = n_baseline / self.fps
            pct_net      = (n_net      / n_visible * 100) if n_visible else 0
            pct_baseline = (n_baseline / n_visible * 100) if n_visible else 0
            transitions = 0
            prev = None
            for z in zones:
                if z is None: continue
                if prev is not None and z != prev:
                    transitions += 1
                prev = z
            stats.append({
                'player_idx'         : p,
                'frames_in_net'      : n_net,
                'frames_in_baseline' : n_baseline,
                'frames_unknown'     : n_unknown,
                'time_in_net_sec'    : round(t_net, 1),
                'time_in_baseline_sec': round(t_baseline, 1),
                'pct_in_net'         : round(pct_net, 1),
                'pct_in_baseline'    : round(pct_baseline, 1),
                'n_transitions'      : transitions,
            })
        return stats


def export_zones_csv(per_player_stats, seat_labels, path):
    import csv
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['player', 'seat', 'time_in_net_sec', 'time_in_baseline_sec',
                    'pct_in_net', 'pct_in_baseline', 'n_transitions'])
        for s in per_player_stats:
            p = s['player_idx']
            w.writerow([
                f'P{p+1}', seat_labels[p],
                s['time_in_net_sec'], s['time_in_baseline_sec'],
                s['pct_in_net'], s['pct_in_baseline'],
                s['n_transitions'],
            ])


def print_zone_summary(per_player_stats, seat_labels, n_players=4):
    print('\n------------------ COURT ZONE SUMMARY ----------------')
    for s in per_player_stats:
        p = s['player_idx']
        net_t = s['time_in_net_sec']
        bl_t  = s['time_in_baseline_sec']
        net_p = s['pct_in_net']
        trans = s['n_transitions']
        print(f'  {seat_labels[p]:14s} | '
              f'net: {net_t:5.1f}s ({net_p:4.1f}%)  '
              f'baseline: {bl_t:5.1f}s  '
              f'transitions: {trans}')
    print('-------------------------------------------------------\n')


ZONE_COLORS = {
    'net'     : (0, 255, 200),
    'baseline': (180, 100, 255),
    None      : (120, 120, 120),
}