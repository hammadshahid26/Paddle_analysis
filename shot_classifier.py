"""
shot_classifier.py (v5.3 — no velocity gate, tighter cooldown, movement tiebreak)
"""
from collections import defaultdict
from typing import List, Dict, Optional, Tuple

import numpy as np


SHOT_LABELS = ['forehand', 'backhand', 'smash', 'vibora']

SHOT_COLORS = {
    'neutral' : (160, 160, 160),
    'forehand': (  0, 200, 255),
    'backhand': (255, 120,   0),
    'smash'   : (  0,   0, 255),
    'vibora'  : (255,   0, 180),
}

L_SHOULDER = 11
R_SHOULDER = 12
L_ELBOW    = 13
R_ELBOW    = 14
L_WRIST    = 15
R_WRIST    = 16
L_HIP      = 23
R_HIP      = 24


def _safe_point(lms, idx, min_vis=0.3):
    if lms is None or idx >= len(lms): return None
    x, y, v = lms[idx]
    if v < min_vis: return None
    return (float(x), float(y))


def _dist(a, b):
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def _ball_valid(ball_xy):
    return ball_xy is not None and ball_xy[0] is not None


class ShotClassifier:

    WRIST_NEAR_BALL_MIN_PX = 120
    WRIST_NEAR_BBOX_RATIO  = 0.40
    TRAJ_REVERSAL_MIN_DY   = 4
    EVENT_COOLDOWN_SEC     = 0.25     # was 0.35 — padel shots can be fast
    EVENT_SCAN_RADIUS      = 5

    # False-positive gates
    MIN_WRIST_VELOCITY_PX_PER_FRAME = 0.0   # DISABLED — pose smoothing killed this
    REQUIRE_UPRIGHT                 = False
    MIN_UPRIGHT_SHOULDER_HIP_PX     = 35
    RALLY_ONLY                      = True
    RALLY_PAD_SEC                   = 2.0

    # Near-player attribution bonus
    NEAR_PLAYER_BONUS_PX = 20

    # Classification
    POST_WINDOW         = 12
    PRE_WINDOW          = 12
    BADGE_DURATION_SEC  = 1.0
    OVERHEAD_Y_MARGIN   = 15
    VIBORA_SPEED_MIN    = 22

    AUTO_HAND_VOTES_NEEDED = 6

    def __init__(self, fps, frame_height=720, n_players=4, handedness_hints=None):
        self.fps = fps
        self.H = frame_height
        self.n_players = n_players
        self.cooldown = int(self.EVENT_COOLDOWN_SEC * fps)

        self.handedness: List[Optional[str]] = list(
            handedness_hints if handedness_hints else [None] * n_players
        )
        self._hand_votes = [defaultdict(int) for _ in range(n_players)]

        self.events: List[dict] = []
        self.frame_labels: List[List[str]] = []
        self._last_shot_frame = [-10**9] * n_players

        self.diag = defaultdict(int)

    def fit(self, all_poses, all_persons, ball_track, bounces, rallies=None):
        self.all_poses   = all_poses
        self.all_persons = all_persons
        self.ball_track  = ball_track
        self.bounces     = bounces
        self.rallies     = rallies or []

    def _in_rally(self, frame_idx):
        pad = int(self.RALLY_PAD_SEC * self.fps)
        for s, e in self.rallies:
            if (s - pad) <= frame_idx <= (e + pad): return True
        return False

    def classify_all(self):
        n_frames = len(self.ball_track)
        self._detect_events()
        badge_frames = int(self.BADGE_DURATION_SEC * self.fps)
        self.frame_labels = [['neutral'] * self.n_players for _ in range(n_frames)]
        for ev in self.events:
            p = ev['player_idx']
            start = ev['frame']
            end = min(n_frames, start + badge_frames)
            for f in range(start, end):
                if self.frame_labels[f][p] == 'neutral':
                    self.frame_labels[f][p] = ev['shot_type']

        print(f'  event detection diagnostics:')
        for k in ['reversal_candidates', 'rally_gated', 'no_wrist_near_ball',
                  'cooldown_rejected', 'upright_rejected', 'velocity_rejected',
                  'accepted']:
            print(f'    {k:25s} : {self.diag[k]}')

    def _detect_events(self):
        n = len(self.ball_track)
        for i in range(3, n - self.POST_WINDOW):
            if not _ball_valid(self.ball_track[i]): continue
            if i in self.bounces: continue
            if not self._is_trajectory_reversal(i): continue
            self.diag['reversal_candidates'] += 1

            if self.RALLY_ONLY and self.rallies and not self._in_rally(i):
                self.diag['rally_gated'] += 1
                continue

            best_p, best_side, best_f, best_d, best_score = \
                None, None, i, float('inf'), float('inf')
            player_order = [2, 3, 0, 1]

            for p in player_order:
                box = self.all_persons[i][p]
                if box is not None:
                    bbox_h = float(box[3]) - float(box[1])
                    thresh = max(self.WRIST_NEAR_BALL_MIN_PX,
                                 bbox_h * self.WRIST_NEAR_BBOX_RATIO)
                else:
                    thresh = self.WRIST_NEAR_BALL_MIN_PX

                near_bonus = self.NEAR_PLAYER_BONUS_PX if p in (2, 3) else 0

                for df in range(-self.EVENT_SCAN_RADIUS, self.EVENT_SCAN_RADIUS + 1):
                    f = i + df
                    if f < 0 or f >= n: continue
                    if not _ball_valid(self.ball_track[f]): continue
                    ball = self.ball_track[f]
                    lms = self.all_poses[f][p] if f < len(self.all_poses) else None
                    if lms is None: continue

                    for joint_idx, side in [
                        (L_WRIST, 'left'), (R_WRIST, 'right'),
                        (L_ELBOW, 'left'), (R_ELBOW, 'right'),
                    ]:
                        joint = _safe_point(lms, joint_idx)
                        if joint is None: continue
                        d = np.hypot(joint[0] - ball[0], joint[1] - ball[1])
                        if d > thresh: continue

                        # Movement tiebreak: if a wrist is moving, it's more
                        # likely to be the hitter than a still wrist at same dist.
                        wrist_vel = self._wrist_velocity(f, p, side)
                        # Score = distance minus movement bonus minus near bonus
                        # (lower score = better candidate)
                        movement_bonus = min(wrist_vel * 3, 30)  # up to 30px bonus
                        score = d - near_bonus - movement_bonus

                        if score < best_score:
                            best_score = score
                            best_d = d
                            best_p = p
                            best_side = side
                            best_f = f

            if best_p is None:
                self.diag['no_wrist_near_ball'] += 1
                continue

            if best_f - self._last_shot_frame[best_p] < self.cooldown:
                self.diag['cooldown_rejected'] += 1
                continue

            if self.REQUIRE_UPRIGHT and not self._is_upright(best_f, best_p):
                self.diag['upright_rejected'] += 1
                continue

            wrist_vel = self._wrist_velocity(best_f, best_p, best_side)
            if wrist_vel < self.MIN_WRIST_VELOCITY_PX_PER_FRAME:
                self.diag['velocity_rejected'] += 1
                continue

            self._hand_votes[best_p][best_side] += 1
            self._maybe_lock_handedness(best_p)

            shot_type, confidence = self._classify_shot(
                frame=best_f, player_idx=best_p, wrist_side=best_side,
                wrist_velocity=wrist_vel,
            )
            if shot_type is None: continue

            self.diag['accepted'] += 1
            self.events.append({
                'frame'        : best_f,
                'time_sec'     : round(best_f / self.fps, 2),
                'player_idx'   : best_p,
                'wrist_side'   : best_side,
                'shot_type'    : shot_type,
                'confidence'   : round(confidence, 2),
                'wrist_vel'    : round(wrist_vel, 1),
                'wrist_dist_px': round(best_d, 1),
            })
            self._last_shot_frame[best_p] = best_f

    def _is_upright(self, frame, player_idx):
        lms = self.all_poses[frame][player_idx]
        if lms is None: return False
        for shoulder_idx, hip_idx in [(L_SHOULDER, L_HIP), (R_SHOULDER, R_HIP)]:
            sh = _safe_point(lms, shoulder_idx)
            hp = _safe_point(lms, hip_idx)
            if sh is None or hp is None: continue
            vertical_gap = hp[1] - sh[1]
            if vertical_gap >= self.MIN_UPRIGHT_SHOULDER_HIP_PX:
                return True
        return False

    def _wrist_velocity(self, frame, player_idx, side):
        wrist_idx = L_WRIST if side == 'left' else R_WRIST
        curr_lms = self.all_poses[frame][player_idx]
        prev_lms = self.all_poses[frame - 2][player_idx] if frame - 2 >= 0 else None
        curr = _safe_point(curr_lms, wrist_idx)
        prev = _safe_point(prev_lms, wrist_idx) if prev_lms else None
        if curr is None or prev is None: return 0.0
        return _dist(curr, prev) / 2.0

    def _is_trajectory_reversal(self, i):
        pre  = [self.ball_track[i - k] for k in (1, 2, 3)
                if _ball_valid(self.ball_track[i - k])]
        post = [self.ball_track[i + k] for k in (1, 2, 3)
                if _ball_valid(self.ball_track[i + k])]
        if len(pre) < 2 or len(post) < 2: return False

        pre_y  = [p[1] for p in pre]
        post_y = [p[1] for p in post]
        dy_pre  = pre_y[0]  - pre_y[-1]
        dy_post = post_y[-1] - post_y[0]
        y_reversal = abs(dy_pre - dy_post) >= self.TRAJ_REVERSAL_MIN_DY * 2

        pre_x  = [p[0] for p in pre]
        post_x = [p[0] for p in post]
        dx_pre  = pre_x[0]  - pre_x[-1]
        dx_post = post_x[-1] - post_x[0]
        x_sign_change = (dx_pre > 0) != (dx_post > 0)
        x_reversal = x_sign_change and abs(dx_pre - dx_post) > 10

        return y_reversal or x_reversal

    def _maybe_lock_handedness(self, p):
        if self.handedness[p] is not None: return
        total = sum(self._hand_votes[p].values())
        if total < self.AUTO_HAND_VOTES_NEEDED: return
        left_v = self._hand_votes[p]['left']
        right_v = self._hand_votes[p]['right']
        self.handedness[p] = 'left' if left_v > right_v else 'right'

    def _classify_shot(self, frame, player_idx, wrist_side, wrist_velocity):
        lms = self.all_poses[frame][player_idx]
        if lms is None: return None, 0.0

        hand = self.handedness[player_idx] or wrist_side
        wrist_idx    = L_WRIST    if hand == 'left' else R_WRIST
        shoulder_idx = L_SHOULDER if hand == 'left' else R_SHOULDER
        hip_idx      = L_HIP      if hand == 'left' else R_HIP

        wrist    = _safe_point(lms, wrist_idx)
        shoulder = _safe_point(lms, shoulder_idx)
        hip      = _safe_point(lms, hip_idx)
        if wrist is None or shoulder is None or hip is None:
            alt = 'left' if hand == 'right' else 'right'
            wrist    = _safe_point(lms, L_WRIST    if alt == 'left' else R_WRIST)
            shoulder = _safe_point(lms, L_SHOULDER if alt == 'left' else R_SHOULDER)
            hip      = _safe_point(lms, L_HIP      if alt == 'left' else R_HIP)
            if wrist is None or shoulder is None or hip is None:
                return None, 0.0
            hand = alt

        wrist_above_shoulder = wrist[1] < (shoulder[1] - self.OVERHEAD_Y_MARGIN)
        body_center_x = (shoulder[0] + hip[0]) / 2.0
        wrist_right_of_center = wrist[0] > body_center_x
        if hand == 'right':
            is_forehand_side = wrist_right_of_center
        else:
            is_forehand_side = not wrist_right_of_center

        if wrist_above_shoulder:
            if wrist_velocity >= self.VIBORA_SPEED_MIN:
                return 'vibora', 0.60
            else:
                return 'smash', 0.75
        else:
            if is_forehand_side:
                return 'forehand', 0.85
            else:
                return 'backhand', 0.85


def export_shot_events_csv(events, handedness, seat_labels, path):
    import csv
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['frame', 'time_sec', 'player', 'seat',
                    'wrist_side', 'shot_type', 'confidence',
                    'wrist_vel', 'wrist_dist_px'])
        for e in events:
            p = e['player_idx']
            w.writerow([
                e['frame'], e['time_sec'],
                f'P{p+1}', seat_labels[p],
                e['wrist_side'], e['shot_type'],
                e['confidence'],
                e.get('wrist_vel', ''),
                e.get('wrist_dist_px', ''),
            ])


def print_session_summary(events, handedness, seat_labels, n_players=4):
    counts = [defaultdict(int) for _ in range(n_players)]
    for e in events:
        counts[e['player_idx']][e['shot_type']] += 1
    print('\n------------------ SESSION SUMMARY ------------------')
    for p in range(n_players):
        hand = handedness[p] or 'unknown'
        total = sum(counts[p].values())
        if total == 0:
            print(f'  {seat_labels[p]} ({hand:>7}-handed): 0 shots')
            continue
        breakdown = ', '.join(f'{v} {k}' for k, v in
                              sorted(counts[p].items(), key=lambda x: -x[1]))
        print(f'  {seat_labels[p]} ({hand:>7}-handed): {total} shots | {breakdown}')
    print('-----------------------------------------------------\n')