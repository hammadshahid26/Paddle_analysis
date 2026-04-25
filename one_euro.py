"""
one_euro.py
-----------
One Euro Filter for jitter-free pose landmark smoothing.

Reference:
    Casiez, Roussel, Vogel. 2012. "1€ Filter: A Simple Speed-based
    Low-pass Filter for Noisy Input in Interactive Systems."

Core idea: use a low cutoff when the signal is still (kills jitter)
and a high cutoff when the signal moves fast (preserves responsiveness).
Industry standard for pose smoothing — used in MediaPipe's internal
smoothing, hand-tracking UIs, AR systems, etc.

Tuned defaults for MediaPipe pose @ 30 fps:
    min_cutoff = 1.0   (higher = less lag but more jitter)
    beta       = 0.007 (higher = more responsive to fast motion)
    d_cutoff   = 1.0
"""

import math


class _LowPass:
    """Exponential low-pass filter with adaptive alpha."""
    def __init__(self):
        self._y_prev = None

    def filter(self, x: float, alpha: float) -> float:
        if self._y_prev is None:
            self._y_prev = x
            return x
        y = alpha * x + (1 - alpha) * self._y_prev
        self._y_prev = y
        return y

    def last(self):
        return self._y_prev


def _alpha(cutoff: float, dt: float) -> float:
    tau = 1.0 / (2 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class OneEuroFilter:
    """
    Per-channel 1-D One Euro filter. Use one instance per scalar stream
    (e.g. one for landmark[k].x, one for landmark[k].y).
    """

    def __init__(
        self,
        min_cutoff: float = 1.0,
        beta: float = 0.007,
        d_cutoff: float = 1.0,
    ):
        self.min_cutoff = min_cutoff
        self.beta       = beta
        self.d_cutoff   = d_cutoff
        self._x_filter  = _LowPass()
        self._dx_filter = _LowPass()
        self._x_prev    = None
        self._t_prev    = None

    def filter(self, x: float, t: float) -> float:
        if self._t_prev is None:
            self._t_prev = t
            self._x_prev = x
            self._x_filter.filter(x, 1.0)
            return x

        dt = max(t - self._t_prev, 1e-6)
        dx = (x - self._x_prev) / dt
        edx = self._dx_filter.filter(dx, _alpha(self.d_cutoff, dt))

        cutoff = self.min_cutoff + self.beta * abs(edx)
        ex = self._x_filter.filter(x, _alpha(cutoff, dt))

        self._x_prev = x
        self._t_prev = t
        return ex


class PoseFilter:
    """
    Manages One Euro filters for a full 33-landmark MediaPipe pose.
    Maintains separate filters per landmark per dimension (x, y).
    Visibility passes through unfiltered.
    """

    N_LANDMARKS = 33

    def __init__(
        self,
        fps: float,
        min_cutoff: float = 1.0,
        beta: float = 0.007,
    ):
        self.fps = fps
        self._filters = [
            (OneEuroFilter(min_cutoff=min_cutoff, beta=beta),   # x
             OneEuroFilter(min_cutoff=min_cutoff, beta=beta))   # y
            for _ in range(self.N_LANDMARKS)
        ]
        self._last_seen = None   # timestamp of the last accepted frame
        self._last_pose = None   # last smoothed pose (for gap-fill)

    def reset(self):
        for fx, fy in self._filters:
            fx.__init__(min_cutoff=fx.min_cutoff, beta=fx.beta)
            fy.__init__(min_cutoff=fy.min_cutoff, beta=fy.beta)
        self._last_seen = None
        self._last_pose = None

    def apply(self, landmarks, frame_idx: int):
        """
        landmarks: list of (x, y, visibility) tuples, or None
        frame_idx: integer frame number, used as timestamp/fps.

        Returns smoothed landmarks list, or None if no pose ever seen.
        Gap-fills for up to ~0.5 s by holding the last smoothed pose.
        """
        t = frame_idx / self.fps

        if landmarks is None:
            # Gap-fill: hold last pose for up to half a second, else return None
            if self._last_pose is not None and self._last_seen is not None:
                if (t - self._last_seen) < 0.5:
                    return self._last_pose
            return None

        smoothed = []
        for i, (x, y, v) in enumerate(landmarks):
            fx, fy = self._filters[i]
            sx = fx.filter(float(x), t)
            sy = fy.filter(float(y), t)
            smoothed.append((int(sx), int(sy), float(v)))

        self._last_seen = t
        self._last_pose = smoothed
        return smoothed
