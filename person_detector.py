"""
person_detector.py (v8 — upgraded YOLO + position hysteresis)
-------------------------------------------------------------
Changes from v7:
  - MIN_BOX_HEIGHT 60 -> 35 (don't drop small far-side players)
  - STABLE_TRACK_MIN_SEC 0.3 -> 0.15 (faster re-lock after dropout)
  - Default model yolov8n.pt -> yolo11m.pt (much better small-person accuracy)
  - POSITION HYSTERESIS: when a seat frees, remember its last feet
    position. A new nearby unassigned track is automatically re-assigned
    to that seat (within 120 px). Huge fix for P1/P2 flicker where the
    same far-side player keeps losing and regaining detection.
"""
from typing import Optional, List, Dict, Tuple
import json
from pathlib import Path

import numpy as np
import supervision as sv
from tqdm import tqdm
from ultralytics import YOLO


def _feet(box) -> tuple:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, float(y2))


def _assign_all_four(tracked_detections, frame_height, midline_ratio=0.6):
    entries = []
    for i in range(len(tracked_detections)):
        tid = int(tracked_detections.tracker_id[i])
        box = tracked_detections.xyxy[i]
        fx, fy = _feet(box)
        entries.append((tid, fx, fy))
    entries.sort(key=lambda e: e[2])
    midline = midline_ratio * frame_height

    far, near = [], []
    n = len(entries)
    if n >= 4:
        far, near = entries[:2], entries[-2:]
    elif n == 3:
        gap1 = entries[1][2] - entries[0][2]
        gap2 = entries[2][2] - entries[1][2]
        if gap1 > gap2: far, near = entries[:1], entries[1:]
        else:           far, near = entries[:2], entries[2:]
    elif n == 2:
        y0, y1 = entries[0][2], entries[1][2]
        if (y1 - y0) > 100:
            far, near = [entries[0]], [entries[1]]
        else:
            if (y0 + y1) / 2 > midline: near = entries
            else:                        far  = entries
    elif n == 1:
        if entries[0][2] > midline: near = entries
        else:                       far  = entries

    far.sort(key=lambda e: e[1])
    near.sort(key=lambda e: e[1])

    mapping: Dict[int, int] = {}
    if len(far)  >= 1: mapping[far[0][0]]  = 0
    if len(far)  >= 2: mapping[far[1][0]]  = 1
    if len(near) >= 1: mapping[near[0][0]] = 2
    if len(near) >= 2: mapping[near[1][0]] = 3
    return mapping


def _seat_is_far(seat_idx: int) -> bool:
    return seat_idx in (0, 1)


class PersonDetector:

    STABLE_TRACK_MIN_SEC = 0.15
    SEAT_FREE_AFTER_SEC  = 1.0
    CONF  = 0.4              # lower conf — paired with polygon/size filters
    IOU   = 0.7
    IMGSZ = 640
    MIN_BOX_HEIGHT = 35
    MAX_BOX_HEIGHT = 650
    # Position-hysteresis radius: a new track within this many pixels of
    # a recently-freed seat's last feet position reclaims that seat.
    HYSTERESIS_RADIUS_PX = 120

    def __init__(self, device: str = 'cpu', model_name: str = 'yolo11m.pt'):
        self.device = device
        self.model = YOLO(model_name)
        self.model.to(device)

    def detect_and_track(
        self,
        frames: list,
        fps: float,
        polygon: Optional[np.ndarray] = None,
        min_score: float = 0.4,
        midline_ratio: float = 0.6,
        batch_size: int = 16,
        cache_path: Optional[str] = None,
    ) -> List[List[Optional[np.ndarray]]]:
        if cache_path and Path(cache_path).exists():
            print(f'  loading cached tracking -> {cache_path}')
            with open(cache_path, 'r') as f:
                raw = json.load(f)
            return [
                [np.asarray(b, dtype=np.float32) if b is not None else None
                 for b in frame_seats]
                for frame_seats in raw
            ]

        h, w = frames[0].shape[:2]
        midline_y = midline_ratio * h
        byte_track = sv.ByteTrack(frame_rate=int(round(fps)))

        zone = None
        if polygon is not None:
            poly = np.asarray(polygon, dtype=np.int32)
            zone = sv.PolygonZone(
                polygon=poly,
                triggering_anchors=(sv.Position.BOTTOM_CENTER,),
            )

        seat_of: Dict[int, int]              = {}
        tid_of:  Dict[int, int]              = {}
        track_history: Dict[int, int]        = {}
        last_seen:     Dict[int, int]        = {}
        last_feet_of_tid: Dict[int, Tuple[float, float]] = {}
        # When a seat is freed, remember its last feet position so we can
        # reclaim it if the same player reappears nearby.
        freed_seat_last_feet: Dict[int, Tuple[float, float]] = {}

        stable_frames = max(1, int(self.STABLE_TRACK_MIN_SEC * fps))
        free_frames   = max(1, int(self.SEAT_FREE_AFTER_SEC  * fps))

        initial_lock_done = False
        all_persons: List[List[Optional[np.ndarray]]] = []

        print(f'  yolo inference ({len(frames)} frames, batch={batch_size})')
        pbar = tqdm(total=len(frames), desc='  person detect+track')
        for start in range(0, len(frames), batch_size):
            batch = frames[start:start + batch_size]
            rgb_batch = [f[:, :, ::-1] for f in batch]
            results = self.model.predict(
                rgb_batch,
                conf=max(self.CONF, min_score),
                iou=self.IOU,
                imgsz=self.IMGSZ,
                device=self.device,
                classes=[0],
                verbose=False,
            )

            for local_idx, result in enumerate(results):
                frame_idx = start + local_idx
                det = sv.Detections.from_ultralytics(result)

                if zone is not None and len(det) > 0:
                    mask = zone.trigger(detections=det)
                    det = det[mask]

                if len(det) > 0:
                    heights = det.xyxy[:, 3] - det.xyxy[:, 1]
                    size_ok = (heights >= self.MIN_BOX_HEIGHT) & \
                              (heights <= self.MAX_BOX_HEIGHT)
                    det = det[size_ok]

                det = byte_track.update_with_detections(det)

                # Record current state of every visible track
                for i in range(len(det)):
                    tid = int(det.tracker_id[i])
                    track_history[tid] = track_history.get(tid, 0) + 1
                    last_seen[tid] = frame_idx
                    last_feet_of_tid[tid] = _feet(det.xyxy[i])

                # Free seats whose tracks went dark — REMEMBER last feet
                stale = [tid for tid in list(seat_of.keys())
                         if frame_idx - last_seen.get(tid, -10**9) > free_frames]
                for tid in stale:
                    seat = seat_of.pop(tid)
                    if tid_of.get(seat) == tid:
                        tid_of.pop(seat, None)
                    if tid in last_feet_of_tid:
                        freed_seat_last_feet[seat] = last_feet_of_tid[tid]

                # Gather stable unassigned tracks in this frame
                stable_unassigned = []
                for i in range(len(det)):
                    tid = int(det.tracker_id[i])
                    if tid in seat_of: continue
                    if track_history[tid] < stable_frames: continue
                    stable_unassigned.append(i)

                empty_seats = [s for s in range(4) if s not in tid_of]

                # -- HYSTERESIS: reclaim freed seats by proximity ---------
                if empty_seats and stable_unassigned and freed_seat_last_feet:
                    used_cand_indices = set()
                    for seat in list(empty_seats):
                        if seat not in freed_seat_last_feet:
                            continue
                        sx, sy = freed_seat_last_feet[seat]
                        # find closest unassigned candidate to this seat's
                        # last known feet
                        best_i = None
                        best_d = self.HYSTERESIS_RADIUS_PX
                        for ci in stable_unassigned:
                            if ci in used_cand_indices: continue
                            tid = int(det.tracker_id[ci])
                            if tid in seat_of: continue
                            fx, fy = _feet(det.xyxy[ci])
                            # also require team match (near/far)
                            cand_is_near = fy > midline_y
                            seat_is_near = not _seat_is_far(seat)
                            if cand_is_near != seat_is_near: continue
                            d = np.hypot(fx - sx, fy - sy)
                            if d < best_d:
                                best_d = d
                                best_i = ci
                        if best_i is not None:
                            tid = int(det.tracker_id[best_i])
                            seat_of[tid] = seat
                            tid_of[seat] = tid
                            freed_seat_last_feet.pop(seat, None)
                            used_cand_indices.add(best_i)
                            empty_seats.remove(seat)
                    # remove any hysteresis-used candidates from the pool
                    stable_unassigned = [i for i in stable_unassigned
                                         if i not in used_cand_indices]

                # -- initial 4-player lock (once only) --------------------
                if not initial_lock_done:
                    if len(stable_unassigned) + len(seat_of) >= 4:
                        all_stable_idx = []
                        for i in range(len(det)):
                            tid = int(det.tracker_id[i])
                            if track_history[tid] >= stable_frames:
                                all_stable_idx.append(i)
                        if len(all_stable_idx) >= 4:
                            stable_det = det[np.array(all_stable_idx)]
                            new_mapping = _assign_all_four(
                                stable_det, frame_height=h,
                                midline_ratio=midline_ratio,
                            )
                            for tid, seat in new_mapping.items():
                                seat_of[tid] = seat
                                tid_of[seat] = tid
                            initial_lock_done = True
                else:
                    # Incremental refill with team sanity check
                    if empty_seats and stable_unassigned:
                        for cand_i in stable_unassigned:
                            tid = int(det.tracker_id[cand_i])
                            if tid in seat_of: continue
                            _fx, fy = _feet(det.xyxy[cand_i])
                            cand_is_near = fy > midline_y
                            picked_seat = None
                            for s in empty_seats:
                                seat_is_near = not _seat_is_far(s)
                                if seat_is_near == cand_is_near:
                                    picked_seat = s
                                    break
                            if picked_seat is None:
                                continue
                            team_seats = [s for s in empty_seats
                                          if (not _seat_is_far(s)) == cand_is_near]
                            if len(team_seats) == 2:
                                # need two candidates to assign both L/R
                                other_same_team = None
                                for other_i in stable_unassigned:
                                    if other_i == cand_i: continue
                                    otid = int(det.tracker_id[other_i])
                                    if otid in seat_of: continue
                                    _ox, oy = _feet(det.xyxy[other_i])
                                    if (oy > midline_y) == cand_is_near:
                                        other_same_team = other_i
                                        break
                                if other_same_team is not None:
                                    _fx, _ = _feet(det.xyxy[cand_i])
                                    ofx, _ = _feet(det.xyxy[other_same_team])
                                    l_seat = min(team_seats)
                                    r_seat = max(team_seats)
                                    if _fx < ofx:
                                        l_tid, r_tid = tid, int(det.tracker_id[other_same_team])
                                    else:
                                        l_tid, r_tid = int(det.tracker_id[other_same_team]), tid
                                    seat_of[l_tid] = l_seat
                                    tid_of[l_seat] = l_tid
                                    seat_of[r_tid] = r_seat
                                    tid_of[r_seat] = r_tid
                                    empty_seats = [s for s in range(4) if s not in tid_of]
                                    continue
                                continue
                            seat_of[tid] = picked_seat
                            tid_of[picked_seat] = tid
                            empty_seats = [s for s in range(4) if s not in tid_of]

                frame_out: List[Optional[np.ndarray]] = [None, None, None, None]
                for i in range(len(det)):
                    tid = int(det.tracker_id[i])
                    if tid in seat_of:
                        frame_out[seat_of[tid]] = det.xyxy[i]
                all_persons.append(frame_out)
                pbar.update(1)
        pbar.close()

        if not initial_lock_done:
            print('  WARNING: never saw 4 stable tracks simultaneously.')

        if cache_path:
            serializable = [
                [b.tolist() if b is not None else None for b in seats]
                for seats in all_persons
            ]
            with open(cache_path, 'w') as f:
                json.dump(serializable, f)
            print(f'  wrote tracking cache -> {cache_path}')

        return all_persons