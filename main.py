import argparse
import json

import cv2
import mediapipe as mp
import numpy as np
import torch
from tqdm import tqdm
from coaching_llm import generate_coaching_report
from ball_detector     import BallDetector
from bounce_detector   import BounceDetector
from person_detector   import PersonDetector
from one_euro          import PoseFilter
from court_mask        import build_court_polygon
from shot_classifier   import (
    ShotClassifier, SHOT_COLORS,
    export_shot_events_csv, print_session_summary,
)
from court_zones       import (
    calibrate_net_and_zones, classify_player_zone, ZoneTracker,
    ZONE_COLORS, export_zones_csv, print_zone_summary,
)


N_PLAYERS = 4

POSE_COLORS = [
    (0,   255, 255),
    (255,   0, 255),
    (0,   220, 255),
    (80,  255,  80),
]
SEAT_LABELS = ['P1 (far-L)', 'P2 (far-R)', 'P3 (near-L)', 'P4 (near-R)']


def read_video(path_video):
    cap = cv2.VideoCapture(path_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frames.append(frame)
    cap.release()
    return frames, fps


def detect_rallies(ball_track, fps, min_duration_sec=1.5, max_gap_sec=0.5):
    max_gap    = int(max_gap_sec * fps)
    min_frames = int(min_duration_sec * fps)
    rallies, in_rally, rally_start, gap_counter = [], False, 0, 0
    for i, (x, _y) in enumerate(ball_track):
        if not in_rally:
            if x is not None:
                in_rally, rally_start, gap_counter = True, i, 0
        else:
            if x is not None:
                gap_counter = 0
            else:
                gap_counter += 1
                if gap_counter > max_gap:
                    end = i - gap_counter
                    if end - rally_start >= min_frames:
                        rallies.append((rally_start, end))
                    in_rally, gap_counter = False, 0
    if in_rally:
        end = len(ball_track) - 1
        if end - rally_start >= min_frames:
            rallies.append((rally_start, end))
    return rallies


def build_ball_polygon(floor_polygon, frame_shape):
    H, W = frame_shape[:2]
    bp = np.asarray(floor_polygon, dtype=np.int32)
    x_left  = int(bp[:, 0].min())
    x_right = int(bp[:, 0].max())
    y_mid = (bp[:, 1].max() + bp[:, 1].min()) / 2
    bottom_verts = bp[bp[:, 1] > y_mid]
    if len(bottom_verts) < 2:
        return np.array([[x_left, 0], [x_right, 0],
                         [x_right, H - 1], [x_left, H - 1]], dtype=np.int32)
    order = np.argsort(bottom_verts[:, 0])
    verts_l_to_r = bottom_verts[order]
    pts = [[x_left, 0], [x_right, 0]]
    for v in verts_l_to_r[::-1]:
        pts.append([int(v[0]), int(v[1])])
    return np.asarray(pts, dtype=np.int32)


def _adaptive_crop(frame, box, scale=1.3, min_size=256, max_size=512):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]
    bw = x2 - x1; bh = y2 - y1
    side = int(max(bw, bh) * scale)
    side = max(min_size, min(max_size, side))
    cx = (x1 + x2) // 2; cy = (y1 + y2) // 2
    half = side // 2
    cx1 = max(0, cx - half); cy1 = max(0, cy - half)
    cx2 = min(w, cx1 + side); cy2 = min(h, cy1 + side)
    cx1 = max(0, cx2 - side); cy1 = max(0, cy2 - side)
    crop = frame[cy1:cy2, cx1:cx2]
    ch, cw = crop.shape[:2]
    if ch != side or cw != side:
        padded = np.zeros((side, side, 3), dtype=crop.dtype)
        padded[:ch, :cw] = crop
        crop = padded
    return crop, cx1, cy1, side


def run_pose_estimation(frames, all_persons, fps, min_cutoff=1.0, beta=0.007):
    mp_pose = mp.solutions.pose
    pose_instances = [
        mp_pose.Pose(
            static_image_mode=False, model_complexity=1,
            smooth_landmarks=True, enable_segmentation=False,
            min_detection_confidence=0.5, min_tracking_confidence=0.5,
        ) for _ in range(N_PLAYERS)
    ]
    pose_filters = [PoseFilter(fps=fps, min_cutoff=min_cutoff, beta=beta)
                    for _ in range(N_PLAYERS)]
    all_poses = []
    print('  pose (persistent + one-euro + adaptive-crop)...')
    for frame_idx, frame in enumerate(tqdm(frames)):
        frame_poses = []
        for p in range(N_PLAYERS):
            box = all_persons[frame_idx][p]
            if box is None:
                frame_poses.append(pose_filters[p].apply(None, frame_idx))
                continue
            crop, ox, oy, side = _adaptive_crop(frame, box)
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            res = pose_instances[p].process(rgb)
            if res.pose_landmarks is None:
                frame_poses.append(pose_filters[p].apply(None, frame_idx))
                continue
            lms_raw = [(lm.x * side + ox, lm.y * side + oy, lm.visibility)
                       for lm in res.pose_landmarks.landmark]
            frame_poses.append(pose_filters[p].apply(lms_raw, frame_idx))
        all_poses.append(frame_poses)
    for pi in pose_instances:
        pi.close()
    return all_poses


def filter_bounces_by_trajectory(bounces_set, ball_track, window=4, min_rise_after=3.0):
    filtered = set()
    n = len(ball_track)
    for b in sorted(bounces_set):
        ys_before = [ball_track[b - k][1]
                     for k in range(1, window + 1) if 0 <= b - k < n
                     and ball_track[b - k][1] is not None]
        ys_after  = [ball_track[b + k][1]
                     for k in range(1, window + 1) if 0 <= b + k < n
                     and ball_track[b + k][1] is not None]
        if len(ys_before) < 2 or len(ys_after) < 2: continue
        falling_before = ys_before[-1] < ys_before[0]
        rising_after   = ys_after[-1] < ys_after[0]
        rise_amount    = ys_after[0] - ys_after[-1]
        if falling_before and rising_after and rise_amount >= min_rise_after:
            filtered.add(b)
    return filtered


POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (24, 26), (26, 28), (28, 30),
    (15, 17), (15, 19), (15, 21), (16, 18), (16, 20), (16, 22),
]


def draw_pose(img, landmarks, color, vis_threshold=0.5):
    if landmarks is None: return
    for a, b in POSE_CONNECTIONS:
        if a >= len(landmarks) or b >= len(landmarks): continue
        ax, ay, av = landmarks[a]
        bx, by, bv = landmarks[b]
        if av > vis_threshold and bv > vis_threshold:
            cv2.line(img, (ax, ay), (bx, by), color, 2, cv2.LINE_AA)
    for (px, py, vis) in landmarks:
        if vis > vis_threshold:
            cv2.circle(img, (px, py), 4, color, -1, cv2.LINE_AA)


def _overlay_rect(img, x1, y1, x2, y2, color, alpha=0.70):
    ov = img.copy()
    cv2.rectangle(ov, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(ov, alpha, img, 1 - alpha, 0, img)


def _text(img, txt, x, y, scale=0.5, color=(255, 255, 255),
          thickness=1, font=cv2.FONT_HERSHEY_DUPLEX):
    cv2.putText(img, txt, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def draw_shot_badge(img, label, box):
    if label == 'neutral' or box is None: return
    x1, y1 = int(box[0]), int(box[1])
    stroke_color = SHOT_COLORS.get(label, (160, 160, 160))
    text = label.upper()
    font = cv2.FONT_HERSHEY_DUPLEX
    scale, thick = 0.55, 2
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    pad = 6
    bx1 = x1
    by1 = y1 - th - pad * 2 - 26
    bx2 = x1 + tw + pad * 2
    by2 = y1 - 26
    _overlay_rect(img, bx1, by1, bx2, by2, stroke_color, alpha=0.9)
    cv2.putText(img, text, (bx1 + pad, by2 - pad),
                font, scale, (255, 255, 255), thick, cv2.LINE_AA)


def draw_timeline(img, frame_idx, total_frames, fps, rallies, bounces, events):
    H, W = img.shape[:2]
    BAR_H = 10
    bar_y = H - BAR_H - 4
    cv2.rectangle(img, (0, bar_y - 2), (W, H), (0, 0, 0), -1)
    cv2.rectangle(img, (0, bar_y), (W, bar_y + BAR_H), (40, 40, 40), -1)
    progress = (frame_idx + 1) / total_frames
    cv2.rectangle(img, (0, bar_y), (int(W * progress), bar_y + BAR_H),
                  (0, 160, 255), -1)
    for s, e in rallies:
        rx1 = int(W * s / total_frames); rx2 = int(W * e / total_frames)
        cv2.rectangle(img, (rx1, bar_y), (rx2, bar_y + BAR_H), (0, 210, 80), -1)
    for b in bounces:
        bx = int(W * b / total_frames)
        cv2.line(img, (bx, bar_y), (bx, bar_y + BAR_H), (0, 0, 255), 2)
    for ev in events:
        ex = int(W * ev['frame'] / total_frames)
        col = SHOT_COLORS.get(ev['shot_type'], (200, 200, 200))
        cv2.line(img, (ex, bar_y - 5), (ex, bar_y - 1), col, 2)
    ts = f'{frame_idx/fps:.1f}s / {total_frames/fps:.1f}s'
    (tw, _), _ = cv2.getTextSize(ts, cv2.FONT_HERSHEY_DUPLEX, 0.38, 1)
    _text(img, ts, (W - tw) // 2, bar_y - 2, scale=0.38, color=(150, 150, 150))


def render(frames, bounces, ball_track, all_persons, all_poses,
           rallies, fps, shot_events, frame_shot_labels,
           handedness, floor_polygon=None,
           zone_calib=None, frame_zones=None,
           draw_trace=True, trace=7):

    imgs_res = []
    total = len(frames)
    for i in range(total):
        img = frames[i].copy()
        in_rally = any(s <= i <= e for s, e in rallies)

        if floor_polygon is not None:
            pts = np.asarray(floor_polygon, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(img, [pts], True, (100, 200, 255), 2, cv2.LINE_AA)

        # Net-line visualization (faint horizontal line)
        # Net & service lines (sanity-check overlay)
        if zone_calib is not None:
            net_y = int(zone_calib['net_y'])
            cv2.line(img, (0, net_y), (img.shape[1], net_y),
                     (200, 200, 100), 1, cv2.LINE_AA)
            far_sl = zone_calib.get('far_service_y')
            if far_sl is not None:
                cv2.line(img, (0, int(far_sl)),
                         (img.shape[1], int(far_sl)),
                         (100, 200, 100), 1, cv2.LINE_AA)
            near_sl = zone_calib.get('near_service_y')
            if near_sl is not None:
                cv2.line(img, (0, int(near_sl)),
                         (img.shape[1], int(near_sl)),
                         (100, 200, 100), 1, cv2.LINE_AA)
        _overlay_rect(img, 0, 0, img.shape[1], 40, (0, 0, 0), alpha=0.72)
        status = 'RALLY' if in_rally else 'DEAD TIME'
        status_col = (0, 220, 0) if in_rally else (0, 80, 220)
        _text(img, f'frame {i}/{total}  |  {status}', 10, 27,
              scale=0.6, color=status_col, thickness=2)
        hands_str = ' '.join(
            f'P{p+1}:{(h or "?")[0].upper()}' for p, h in enumerate(handedness)
        )
        _text(img, hands_str, img.shape[1] - 260, 27,
              scale=0.5, color=(200, 200, 200))

        if ball_track[i][0] is not None:
            if draw_trace:
                for j in range(trace):
                    if i - j >= 0 and ball_track[i - j][0] is not None:
                        cv2.circle(img,
                                   (int(ball_track[i - j][0]),
                                    int(ball_track[i - j][1])),
                                   max(2, 6 - j), (0, 255, 0), -1)
            else:
                cv2.circle(img, (int(ball_track[i][0]), int(ball_track[i][1])),
                           6, (0, 255, 0), 2)

        if i in bounces and ball_track[i][0] is not None:
            bx2 = int(ball_track[i][0]); by2 = int(ball_track[i][1])
            for r in [18, 14, 10]:
                cv2.circle(img, (bx2, by2), r, (0, 0, 255), 2)
            _text(img, 'BOUNCE', bx2 + 22, by2 + 4,
                  scale=0.5, color=(0, 0, 255), thickness=2)

        for p in range(N_PLAYERS):
            box = all_persons[i][p]
            if box is None: continue
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            pc = POSE_COLORS[p]
            cv2.rectangle(img, (x1, y1), (x2, y2), pc, 2)
            tag = SEAT_LABELS[p]
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_DUPLEX, 0.55, 2)
            _overlay_rect(img, x1 - 3, y1 - th - 10, x1 + tw + 5, y1 - 2,
                          (0, 0, 0), 0.70)
            _text(img, tag, x1, y1 - 5, scale=0.55, color=pc, thickness=2)

            # Zone badge — small, next to player tag
            if frame_zones is not None and i < len(frame_zones):
                zone = frame_zones[i][p] if p < len(frame_zones[i]) else None
                if zone is not None:
                    zone_text = f'[{zone.upper()}]'
                    zc = ZONE_COLORS[zone]
                    (zw, zh), _ = cv2.getTextSize(zone_text, cv2.FONT_HERSHEY_DUPLEX, 0.42, 1)
                    zx = x1 + tw + 12
                    zy = y1 - 5
                    _overlay_rect(img, zx - 3, zy - zh - 4, zx + zw + 5, zy + 3,
                                  (0, 0, 0), 0.65)
                    _text(img, zone_text, zx, zy, scale=0.42,
                          color=zc, thickness=1)

            if i < len(all_poses) and p < len(all_poses[i]):
                draw_pose(img, all_poses[i][p], pc)
            if i < len(frame_shot_labels) and p < len(frame_shot_labels[i]):
                draw_shot_badge(img, frame_shot_labels[i][p], box)

        draw_timeline(img, i, total, fps, rallies, bounces, shot_events)
        imgs_res.append(img)
    return imgs_res

def _wrap_text(text: str, max_chars_per_line: int = 78):
    """Word-wrap a paragraph to fit a max character width per line."""
    out = []
    for raw_para in text.split('\n'):
        para = raw_para.strip()
        if not para:
            out.append('')
            continue
        words = para.split()
        line = ''
        for w in words:
            if len(line) + len(w) + 1 <= max_chars_per_line:
                line = (line + ' ' + w) if line else w
            else:
                out.append(line)
                line = w
        if line:
            out.append(line)
    return out


def build_coaching_summary_frames(
    text: str, frame_shape, fps: float, duration_sec: float = 12.0,
):
    """
    Generate frames showing the coaching summary as a full-frame text card.
    Returns a list of identical frames (it's a static screen).
    """
    H, W = frame_shape[:2]
    n_frames = int(duration_sec * fps)

    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    # subtle gradient background
    for y in range(H):
        v = int(15 + (y / H) * 25)
        canvas[y, :] = (v, v // 2, v // 3)

    title = 'PADEL COACHING SUMMARY'
    title_font = cv2.FONT_HERSHEY_DUPLEX
    title_scale = 1.1
    title_thick = 2
    (tw, th), _ = cv2.getTextSize(title, title_font, title_scale, title_thick)
    title_x = (W - tw) // 2
    title_y = 70
    cv2.putText(canvas, title, (title_x, title_y),
                title_font, title_scale, (0, 200, 255), title_thick, cv2.LINE_AA)
    # underline
    cv2.line(canvas, (title_x, title_y + 12),
             (title_x + tw, title_y + 12), (0, 200, 255), 2, cv2.LINE_AA)

    # Body text
    body_font = cv2.FONT_HERSHEY_DUPLEX
    body_scale = 0.55
    body_thick = 1
    line_height = 26
    margin_x = 60
    margin_top = 130

    # Wrap text
    max_chars = max(50, int((W - 2 * margin_x) / 9))
    wrapped = _wrap_text(text, max_chars)

    # If too tall, shrink line height a bit
    avail_height = H - margin_top - 40
    if len(wrapped) * line_height > avail_height:
        line_height = max(18, avail_height // max(1, len(wrapped)))

    y = margin_top
    for line in wrapped:
        if y + line_height > H - 30:
            break
        cv2.putText(canvas, line, (margin_x, y),
                    body_font, body_scale, (235, 235, 235), body_thick, cv2.LINE_AA)
        y += line_height

    # Footer
    footer = 'Generated by Padel Analytics  |  Subscriber focus: P3 (near-L), P4 (near-R)'
    (fw, fh), _ = cv2.getTextSize(footer, body_font, 0.45, 1)
    cv2.putText(canvas, footer, ((W - fw) // 2, H - 20),
                body_font, 0.45, (140, 140, 160), 1, cv2.LINE_AA)

    return [canvas.copy() for _ in range(n_frames)]

def write(imgs_res, fps, path_output_video):
    height, width = imgs_res[0].shape[:2]
    out = cv2.VideoWriter(path_output_video,
                          cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    for frame in imgs_res:
        out.write(frame)
    out.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_ball_track_model', type=str, required=True)
    parser.add_argument('--path_bounce_model',     type=str, required=True)
    parser.add_argument('--path_input_video',      type=str, required=True)
    parser.add_argument('--path_output_video',     type=str, required=True)
    parser.add_argument('--yolo_model',            type=str, default='yolo11m.pt')
    parser.add_argument('--min_score',             type=float, default=0.5)
    parser.add_argument('--midline_ratio',         type=float, default=0.6)
    parser.add_argument('--cache_tracking',        type=str, default=None)
    parser.add_argument('--save_polygon',          type=str, default=None)
    parser.add_argument('--path_shot_csv',         type=str, default='shot_events.csv')
    parser.add_argument('--path_zones_csv',        type=str, default='zones.csv')
    parser.add_argument('--path_coaching_report', type=str,
                        default='coaching_report.txt')
    parser.add_argument('--skip_coaching', action='store_true',
                        help='Skip LLM coaching step (saves ~30s).')
    parser.add_argument('--skip_polygon',          action='store_true')
    parser.add_argument('--p1_hand', type=str, default=None, choices=[None, 'left', 'right'])
    parser.add_argument('--p2_hand', type=str, default=None, choices=[None, 'left', 'right'])
    parser.add_argument('--p3_hand', type=str, default=None, choices=[None, 'left', 'right'])
    parser.add_argument('--p4_hand', type=str, default=None, choices=[None, 'left', 'right'])
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    frames, fps = read_video(args.path_input_video)
    print(f'Frames: {len(frames)}   FPS: {fps:.2f}')

    floor_polygon = None
    ball_polygon  = None
    if not args.skip_polygon:
        print('[0/6] building court polygon via blue-floor segmentation...')
        floor_polygon, _ = build_court_polygon(frames)
        if floor_polygon is not None:
            print(f'  floor polygon : {len(floor_polygon)} vertices')
            ball_polygon = build_ball_polygon(floor_polygon, frames[0].shape)
            print(f'  ball polygon  : {len(ball_polygon)} vertices')
            if args.save_polygon:
                with open(args.save_polygon, 'w') as f:
                    json.dump({'floor': floor_polygon.tolist(),
                               'ball':  ball_polygon.tolist()}, f)

    print('[1/6] ball detection')
    ball_detector = BallDetector(args.path_ball_track_model, device)
    ball_track = ball_detector.infer_model(
        frames, polygon=ball_polygon if ball_polygon is not None else floor_polygon
    )

    print('[2/6] person detection + identity lock')
    person_detector = PersonDetector(device, model_name=args.yolo_model)
    all_persons = person_detector.detect_and_track(
        frames, fps=fps, polygon=floor_polygon, min_score=args.min_score,
        midline_ratio=args.midline_ratio, cache_path=args.cache_tracking,
    )

    print('[3/6] bounce detection')
    bounce_detector = BounceDetector(args.path_bounce_model)
    raw_bounces = bounce_detector.predict(
        [x[0] for x in ball_track], [x[1] for x in ball_track],
    )
    bounces = filter_bounces_by_trajectory(raw_bounces, ball_track)
    print(f'  raw={len(raw_bounces)}, kept={len(bounces)}')

    rallies = detect_rallies(ball_track, fps)
    print(f'  rallies: {len(rallies)}')

    print('[4/6] pose estimation')
    all_poses = run_pose_estimation(frames, all_persons, fps=fps)

    print('[5/6] shot classification')
    hand_hints = [args.p1_hand, args.p2_hand, args.p3_hand, args.p4_hand]
    clf = ShotClassifier(fps=fps, frame_height=frames[0].shape[0],
                         n_players=N_PLAYERS, handedness_hints=hand_hints)
    clf.fit(all_poses, all_persons, ball_track, bounces, rallies=rallies)
    clf.classify_all()
    print(f'  shot events detected: {len(clf.events)}')
    print(f'  handedness (auto): {clf.handedness}')

    export_shot_events_csv(clf.events, clf.handedness, SEAT_LABELS,
                           args.path_shot_csv)
    print(f'  csv -> {args.path_shot_csv}')
    print_session_summary(clf.events, clf.handedness, SEAT_LABELS, N_PLAYERS)

    print('[6/6] court zone tracking')
    zone_calib = None
    zone_tracker = ZoneTracker(fps=fps, n_players=N_PLAYERS)
    frame_zones = []
    if floor_polygon is not None:
        # Try to detect actual court lines first — camera-agnostic
        from court_mask import detect_court_lines
        detected = detect_court_lines(frames, floor_polygon)
        if detected is not None:
            print(f'  detected lines: net={detected["net_y"]:.0f} '
                  f'far_sl={detected.get("far_service_y")} '
                  f'near_sl={detected.get("near_service_y")}')
        else:
            print(f'  line detection failed -> using ratio fallback')
        zone_calib = calibrate_net_and_zones(
            floor_polygon, frames[0].shape[0], detected_lines=detected,
        )
        print(f'  method: {zone_calib["method"]}  '
              f'net_y: {zone_calib["net_y"]:.0f}px  '
              f'far_radius: {zone_calib["far_net_radius"]:.0f}px  '
              f'near_radius: {zone_calib["near_net_radius"]:.0f}px')
        for i in range(len(frames)):
            zones_this_frame = [
                classify_player_zone(all_persons[i][p], p, zone_calib)
                for p in range(N_PLAYERS)
            ]
            zone_tracker.update(zones_this_frame)
            frame_zones.append(zones_this_frame)
        zone_stats = zone_tracker.per_player_stats()
        print_zone_summary(zone_stats, SEAT_LABELS, N_PLAYERS)
        export_zones_csv(zone_stats, SEAT_LABELS, args.path_zones_csv)
        print(f'  csv -> {args.path_zones_csv}')
    else:
        print('  no polygon — zones skipped')
        for i in range(len(frames)):
            frame_zones.append([None] * N_PLAYERS)
        zone_stats = []

    coaching_report = None
    if not args.skip_coaching:
        print('[7/7] LLM coaching report')
        coaching_report = generate_coaching_report(
            shot_events=clf.events,
            zone_stats=zone_stats,
            rallies=rallies,
            bounces=bounces,
            handedness=clf.handedness,
            seat_labels=SEAT_LABELS,
            fps=fps,
            n_frames=len(frames),
            focus_seats=('P3', 'P4'),
            out_path=args.path_coaching_report,
        )
        print('\n' + '=' * 60)
        print(coaching_report)
        print('=' * 60 + '\n')


    print('rendering')
    imgs_res = render(
        frames, bounces, ball_track, all_persons, all_poses,
        rallies, fps,
        shot_events=clf.events, frame_shot_labels=clf.frame_labels,
        handedness=clf.handedness, floor_polygon=floor_polygon,
        zone_calib=zone_calib, frame_zones=frame_zones,
        draw_trace=True,
    )

    # Append coaching summary card to the end of the video
    if coaching_report:
        print('  appending coaching summary card to video...')
        summary_frames = build_coaching_summary_frames(
            coaching_report, frames[0].shape, fps,
            duration_sec=12.0,
        )
        imgs_res.extend(summary_frames)

    write(imgs_res, fps, args.path_output_video)
    print(f'done -> {args.path_output_video}')