"""
ball_detector.py  (v2 — polygon-filtered)
-----------------------------------------
Same TrackNet inference as before, but with two upgrades:

1. Polygon ROI filtering on ball candidates. When TrackNet detects a
   ball outside the court polygon (e.g. a ball on the adjacent court),
   it's rejected and the next-best candidate is considered instead.

2. Ball-candidate gating by distance-from-previous still works the same.
"""
from tracknet import BallTrackerNet
import torch
import cv2
import numpy as np
from scipy.spatial import distance
from tqdm import tqdm


def _point_in_polygon(pt, polygon):
    """polygon: np.ndarray shape (N, 2). pt: (x, y). Uses cv2."""
    if polygon is None:
        return True
    return cv2.pointPolygonTest(polygon.astype(np.int32), (float(pt[0]), float(pt[1])), False) >= 0


class BallDetector:
    def __init__(self, path_model=None, device='cpu'):
        self.model = BallTrackerNet(input_channels=9, out_channels=256)
        self.device = device
        if path_model:
            self.model.load_state_dict(torch.load(path_model, map_location=device))
            self.model = self.model.to(device)
            self.model.eval()
        self.width = 640
        self.height = 360

    def infer_model(self, frames, polygon=None):
        """
        polygon: optional Nx2 np.ndarray of court ROI in SOURCE-frame (720p) coords.
                 Ball candidates outside polygon are rejected.
        """
        ball_track = [(None, None)] * 2
        prev_pred = [None, None]
        for num in tqdm(range(2, len(frames)), desc='  ball detection'):
            img = cv2.resize(frames[num], (self.width, self.height))
            img_prev = cv2.resize(frames[num - 1], (self.width, self.height))
            img_preprev = cv2.resize(frames[num - 2], (self.width, self.height))
            imgs = np.concatenate((img, img_prev, img_preprev), axis=2)
            imgs = imgs.astype(np.float32) / 255.0
            imgs = np.rollaxis(imgs, 2, 0)
            inp = np.expand_dims(imgs, axis=0)

            out = self.model(torch.from_numpy(inp).float().to(self.device))
            output = out.argmax(dim=1).detach().cpu().numpy()
            x_pred, y_pred = self.postprocess(output, prev_pred, polygon=polygon)
            prev_pred = [x_pred, y_pred]
            ball_track.append((x_pred, y_pred))
        return ball_track

    def postprocess(self, feature_map, prev_pred, scale=2, max_dist=80, polygon=None):
        feature_map *= 255
        feature_map = feature_map.reshape((self.height, self.width))
        feature_map = feature_map.astype(np.uint8)
        ret, heatmap = cv2.threshold(feature_map, 127, 255, cv2.THRESH_BINARY)
        circles = cv2.HoughCircles(
            heatmap, cv2.HOUGH_GRADIENT, dp=1, minDist=1,
            param1=50, param2=2, minRadius=2, maxRadius=7,
        )
        x, y = None, None
        if circles is not None:
            # Gather all candidates, score by (a) distance to prev if prev
            # exists, (b) else first one. ALSO reject any outside polygon.
            cands = []
            for i in range(len(circles[0])):
                cx = circles[0][i][0] * scale
                cy = circles[0][i][1] * scale
                if not _point_in_polygon((cx, cy), polygon):
                    continue
                cands.append((cx, cy))

            if cands:
                if prev_pred[0] is not None:
                    for cx, cy in cands:
                        if distance.euclidean((cx, cy), prev_pred) < max_dist:
                            x, y = cx, cy
                            break
                    # If no candidate within max_dist, REJECT (don't fall
                    # back to nearest — that's how side-court balls sneak
                    # in during detection gaps). Ball gap will be handled
                    # downstream by rally/trajectory logic.
                else:
                    x, y = cands[0]
        return x, y