import argparse
import os
import time
from collections import deque
from datetime import datetime

import cv2
import numpy as np

# SETTINGS - every tunable number lives here
CAMERA_INDEX = 0           # default webcam (0 = first camera)
FRAME_WIDTH = 640          # (a) every frame is resized to this width for speed

# --- segmentation / box building (b) ---
MIN_BLOB_AREA = 300        # ignore tiny specks before merging pieces together
MERGE_GAP = 40             # boxes closer than this (pixels) become ONE object
MIN_AREA = 1500            # a merged box must be at least this big (w x h)
WARMUP_FRAMES = 60         # frames used to learn the empty background first

# --- tracking (e) ---
MIN_HITS = 5               # object must be seen this many frames to get an ID
MAX_DISAPPEARED = 25       # frames a lost object is remembered
MAX_DISTANCE = 120         # max pixel jump to match the same object
TRAIL_LENGTH = 30          # number of past positions drawn as the trail

# --- alert logic ---
MIN_MOTION_FRAMES = 5      # motion must last this many frames to raise ALERT
SNAPSHOT_COOLDOWN = 5      # seconds between saved alert snapshots

# --- Haar cascade (c) ---
FACE_SCALE_FACTOR = 1.1    # image-pyramid step: smaller = slower, more thorough
FACE_MIN_NEIGHBORS = 5     # higher = fewer false faces, but more missed faces
FACE_MIN_SIZE = (60, 60)   # ignore faces smaller than this (pixels)
FACE_EVERY_N_FRAMES = 3    # run the slow face detector every Nth frame only

# --- output folders ---
ALERT_DIR = "alerts"       # evidence snapshots
FIG_DIR = "figures"        # report figures (key f)


# (e) OBJECT TRACKING
class CentroidTracker:
    """Keeps the same ID for an object by matching nearest centroids.

    How it works, every frame:
        1. take the centre point (centroid) of every detected box;
        2. measure the distance from every known object to every new centroid;
        3. match closest pairs one-to-one (only if closer than max_distance);
        4. matched objects keep their ID; unmatched new points start as
            *candidates*; unmatched old objects are counted as "missing".

    An object only gets a visible ID after it has been seen `min_hits` times,
    so one-frame noise never receives an ID. Visible IDs start at 1.
    """

    def __init__(self, max_disappeared=25, max_distance=120, trail_len=30,
                    min_hits=5):
        self.next_key = 0        # internal key for every candidate object
        self.next_id = 1         # visible ID, given only to confirmed objects
        self.objects = {}        # key -> (x, y) last known centroid
        self.disappeared = {}    # key -> consecutive frames without a match
        self.trails = {}         # key -> deque of recent centroids
        self.hits = {}           # key -> number of frames matched so far
        self.shown_id = {}       # key -> visible ID (None until confirmed)
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.trail_len = trail_len
        self.min_hits = min_hits

    def _register(self, c):
        """Start tracking a new candidate object at centroid c."""
        k = self.next_key
        self.next_key += 1
        self.objects[k] = c
        self.disappeared[k] = 0
        self.trails[k] = deque([c], maxlen=self.trail_len)
        self.hits[k] = 1
        self.shown_id[k] = None
        if self.min_hits <= 1:
            self._confirm(k)

    def _confirm(self, k):
        """Give a candidate its visible ID number."""
        self.shown_id[k] = self.next_id
        self.next_id += 1

    def _deregister(self, k):
        """Forget an object completely."""
        for d in (self.objects, self.disappeared, self.trails,
                    self.hits, self.shown_id):
            d.pop(k, None)

    def _missed(self, k):
        """Count one frame without a match; drop the object if lost too long."""
        self.disappeared[k] += 1
        # unconfirmed objects (probably noise) are dropped after 3 misses
        limit = self.max_disappeared if self.shown_id[k] is not None else 3
        if self.disappeared[k] > limit:
            self._deregister(k)

    def confirmed(self):
        """Objects visible right now that have an ID: {key: (id, centroid)}."""
        return {k: (self.shown_id[k], self.objects[k]) for k in self.objects
                if self.shown_id[k] is not None and self.disappeared[k] == 0}

    def update(self, centroids):
        """Feed this frame's centroids; returns all tracked objects."""
        # Case 1: nothing detected -> every known object missed a frame
        if len(centroids) == 0:
            for k in list(self.objects):
                self._missed(k)
            return self.objects

        # Case 2: nothing tracked yet -> every centroid is a new candidate
        if len(self.objects) == 0:
            for c in centroids:
                self._register(c)
            return self.objects

        # Case 3: match new centroids to existing objects by distance
        keys = list(self.objects.keys())
        old = np.array([self.objects[k] for k in keys])
        new = np.array(centroids)
        # distance matrix: rows = existing objects, cols = new centroids
        D = np.linalg.norm(old[:, None, :] - new[None, :, :], axis=2)

        # visit objects starting with the one that has the closest match
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]
        used_r, used_c = set(), set()
        for r, c in zip(rows, cols):
            # skip if either side is already matched or the jump is too big
            if r in used_r or c in used_c or D[r, c] > self.max_distance:
                continue
            k = keys[r]
            pt = (int(new[c][0]), int(new[c][1]))   # plain ints for OpenCV
            self.objects[k] = pt
            self.disappeared[k] = 0
            self.trails[k].append(pt)
            self.hits[k] += 1
            if self.shown_id[k] is None and self.hits[k] >= self.min_hits:
                self._confirm(k)
            used_r.add(r)
            used_c.add(c)

        # old objects without a match -> missed a frame
        for r in set(range(D.shape[0])) - used_r:
            self._missed(keys[r])
        # new centroids without a match -> new candidate objects
        for c in set(range(D.shape[1])) - used_c:
            self._register((int(new[c][0]), int(new[c][1])))
        return self.objects


# HELPER FUNCTIONS
def merge_boxes(boxes, gap=40):
    """Join boxes that overlap or are closer than `gap` pixels, so one person
    (head, body, arms...) becomes ONE box instead of several.

    boxes: list of (x, y, w, h).  Returns a list of (x, y, w, h).
    """
    # work with corner coordinates (x1, y1, x2, y2)
    rects = [(x, y, x + w, y + h) for (x, y, w, h) in boxes]

    def near(a, b):
        # True if the two rectangles overlap or are within `gap` pixels
        return not (a[2] + gap < b[0] or b[2] + gap < a[0] or
                    a[3] + gap < b[1] or b[3] + gap < a[1])

    merged_any = True
    while merged_any:                 # repeat until no more boxes can merge
        merged_any = False
        out = []
        while rects:
            a = rects.pop()
            i = 0
            while i < len(rects):
                if near(a, rects[i]):
                    b = rects.pop(i)
                    # replace a with the rectangle that covers both
                    a = (min(a[0], b[0]), min(a[1], b[1]),
                            max(a[2], b[2]), max(a[3], b[3]))
                    merged_any = True
                    i = 0             # a grew, so re-check against the rest
                else:
                    i += 1
            out.append(a)
        rects = out
    return [(x1, y1, x2 - x1, y2 - y1) for (x1, y1, x2, y2) in rects]


def make_bg_subtractor():
    """(b) MOG2 background model.

    history=300      : learns from roughly the last 300 frames
    varThreshold=25  : how different a pixel must be to count as moving
    detectShadows    : shadows are marked 127 so they can be thrown away
    shadow threshold 0.7: only mild darkening (~30% or less) is a shadow.
    """
    bg = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=25,
                                            detectShadows=True)
    bg.setShadowThreshold(0.7)
    return bg


def label(img, text):
    """Return a BGR copy of img with a caption (used for Figure 1)."""
    out = img.copy()
    if out.ndim == 2:                       # grayscale -> 3 channels
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    cv2.putText(out, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 255, 255), 2)
    return out


# PIPELINE STAGES
def preprocess(frame):
    """(a) IMAGE OPERATIONS.

    Returns (resized colour frame, grayscale, blurred colour, equalized gray).
        resize   : fewer pixels -> every later step is faster
        gray     : the Haar detector only needs brightness
        blur     : Gaussian blur removes camera noise so noise is not "motion"
        equalize : histogram equalization evens out lighting for face detection
    The blurred COLOUR frame feeds the motion model because MOG2 uses colour
    to tell real objects from shadows.
    """
    h, w = frame.shape[:2]
    frame = cv2.resize(frame, (FRAME_WIDTH, int(h * FRAME_WIDTH / w)))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(frame, (5, 5), 0)
    eq = cv2.equalizeHist(gray)
    return frame, gray, blur, eq


def segment_motion(blur, bg, kernel):
    """(b) IMAGE SEGMENTATION: turn a blurred frame into a clean motion mask
    (white = moving pixels, black = background)."""
    fg = bg.apply(blur)                       # 255 = motion, 127 = shadow
    _, mask = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)  # drop shadows
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)     # remove specks
    mask = cv2.dilate(mask, kernel, iterations=2)             # fill gaps
    return mask


def find_motion_boxes(mask):
    """Contours -> bounding boxes -> merge close pieces -> drop small boxes."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    raw = [cv2.boundingRect(c) for c in contours
            if cv2.contourArea(c) >= MIN_BLOB_AREA]
    return [b for b in merge_boxes(raw, MERGE_GAP)
            if b[2] * b[3] >= MIN_AREA]


def detect_faces(face_cascade, eq):
    """(c) HAAR CASCADE: returns an array of face boxes (x, y, w, h)."""
    return face_cascade.detectMultiScale(
        eq, scaleFactor=FACE_SCALE_FACTOR,
        minNeighbors=FACE_MIN_NEIGHBORS, minSize=FACE_MIN_SIZE)


# VISUALIZATION
def draw_tracked_objects(output, tracker, shown, box_of):
    """Draw box, ID tag, trail and centre dot for every confirmed object."""
    for key, (disp_id, c) in shown.items():
        b = box_of.get(tuple(c))
        if b:
            x, y, bw, bh = b
            cv2.rectangle(output, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
            # ID tag: green label above the box with dark text
            tag = f"ID {disp_id}"
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX,
                                            0.55, 2)
            ty = max(y, th + 8)               # keep the tag on screen
            cv2.rectangle(output, (x, ty - th - 8), (x + tw + 8, ty),
                            (0, 255, 0), -1)
            cv2.putText(output, tag, (x + 4, ty - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
        # trail = line through the last positions of this object
        pts = np.array(tracker.trails[key], dtype=np.int32).reshape(-1, 1, 2)
        if len(pts) > 1:
            cv2.polylines(output, [pts], False, (0, 255, 255), 2)
        cv2.circle(output, tuple(c), 4, (0, 0, 255), -1)


def draw_faces(output, faces):
    """Draw a blue box labelled 'Face' for every detected face."""
    for (fx, fy, fw, fh) in faces:
        cv2.rectangle(output, (fx, fy), (fx + fw, fy + fh), (255, 0, 0), 2)
        cv2.putText(output, "Face", (fx, fy - 6), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 0, 0), 2)


def draw_panel(output, status, alert, n_objects, n_faces, n_alerts, fps):
    """Semi-transparent status panel in the bottom-left corner (red when ALERT
    is active). It sits at the bottom so it does not cover ID tags, which are
    drawn above people's heads."""
    h = output.shape[0]
    y0 = h - 125                                   # top edge of the panel
    overlay = output.copy()
    color = (0, 0, 200) if alert else (60, 60, 60)
    cv2.rectangle(overlay, (0, y0), (250, h), color, -1)
    output = cv2.addWeighted(overlay, 0.6, output, 0.4, 0)   # 60% panel
    lines = [status,
                f"Moving objects: {n_objects}",
                f"Faces: {n_faces}",
                f"Alerts: {n_alerts}",
                f"FPS: {fps:.1f}"]
    for i, t in enumerate(lines):
        cv2.putText(output, t, (8, y0 + 22 + i * 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 2 if i == 0 else 1)
    return output


def save_figures(frame, gray, blur, eq, mask, output):
    """Key f: save Figure 1 (preprocessing), Figure 2 (mask), Figure 3 (output)."""
    stamp = datetime.now().strftime("%H%M%S")
    panels = [label(frame, "Original"), label(gray, "Grayscale"),
                label(blur, "Blurred"), label(eq, "Equalized")]
    montage = np.vstack([np.hstack(panels[:2]), np.hstack(panels[2:])])
    cv2.imwrite(os.path.join(FIG_DIR, f"fig1_preprocess_{stamp}.png"), montage)
    cv2.imwrite(os.path.join(FIG_DIR, f"fig2_mask_{stamp}.png"), mask)
    cv2.imwrite(os.path.join(FIG_DIR, f"fig3_output_{stamp}.png"), output)
    print("Figures saved in ./" + FIG_DIR)


# INPUT HANDLING
def parse_args(argv=None):
    """Command-line options."""
    p = argparse.ArgumentParser(
        description="Smart Room Monitor: motion-triggered security with "
                    "face detection.")
    p.add_argument("--source", type=int, default=CAMERA_INDEX,
                    help="webcam index (0, 1, ...) (default: %(default)s)")
    return p.parse_args(argv)


def open_source(index):
    """Open the live webcam with the given index."""
    return cv2.VideoCapture(index)


# MAIN PROGRAM
def main(argv=None):
    args = parse_args(argv)

    # Haar cascades were removed from the main package in OpenCV 5.x
    if not hasattr(cv2, "CascadeClassifier"):
        print("OpenCV %s has no Haar CascadeClassifier (OpenCV 5.x moved it)."
                % cv2.__version__)
        print('Fix: pip uninstall -y opencv-python')
        print('     pip install "opencv-python<5"')
        return

    os.makedirs(ALERT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    cap = open_source(args.source)
    if not cap.isOpened():
        print(f"Cannot open webcam {args.source}. Try --source 1.")
        return
    print(f"Webcam: {args.source}   (press q in the video window to quit)")

    # --- one-time setup ---
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    bg = make_bg_subtractor()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    tracker = CentroidTracker(MAX_DISAPPEARED, MAX_DISTANCE, TRAIL_LENGTH,
                                MIN_HITS)

    # --- state kept between frames ---
    frame_idx = 0              # frames processed since start / last reset
    faces = []                 # last face detections (reused between runs)
    motion_frames = 0          # consecutive frames that contained motion
    last_snapshot = 0.0        # time of the last saved evidence image
    show_mask = False          # toggled with key d
    fps, prev_t = 0.0, time.time()
    fps_samples = []
    tests = {"correct": 0, "false_alarm": 0, "miss": 0}   # keys 1 / 2 / 3
    alert_count = 0
    was_alert = False

    # ---------------------------- main loop ----------------------------
    while True:
        ret, frame = cap.read()
        if not ret:                       # camera lost
            break

        # (a) preprocessing
        frame, gray, blur, eq = preprocess(frame)

        # (b) segmentation: motion mask (always updated so the model learns)
        mask = segment_motion(blur, bg, kernel)

        calibrating = frame_idx < WARMUP_FRAMES
        frame_idx += 1

        # boxes are ignored while the background is still being learned
        boxes = [] if calibrating else find_motion_boxes(mask)

        # (e) tracking: centroid of each box -> tracker -> IDs and trails
        centroids = [(x + bw // 2, y + bh // 2) for (x, y, bw, bh) in boxes]
        box_of = {c: b for c, b in zip(centroids, boxes)}
        tracker.update(centroids)
        shown = tracker.confirmed()

        # draw on a copy so the processed frames stay clean for Figure 1
        output = frame.copy()
        draw_tracked_objects(output, tracker, shown, box_of)

        # (c) face detection (every Nth frame only, result is reused)
        if frame_idx % FACE_EVERY_N_FRAMES == 0:
            faces = detect_faces(face_cascade, eq)
        draw_faces(output, faces)

        # alert logic: motion must last MIN_MOTION_FRAMES frames in a row
        motion_frames = motion_frames + 1 if boxes else 0
        alert = motion_frames >= MIN_MOTION_FRAMES
        if alert and not was_alert:       # count each new alert once
            alert_count += 1
        was_alert = alert

        # speed measurement (smoothed FPS for display, raw for the average)
        now = time.time()
        inst = 1.0 / max(now - prev_t, 1e-6)
        prev_t = now
        fps = inst if fps == 0 else 0.9 * fps + 0.1 * inst
        if not calibrating:
            fps_samples.append(inst)

        # status panel
        if calibrating:
            status = "CALIBRATING..."
        else:
            status = "ALERT: INTRUDER?" if alert else "Status: Clear"
        output = draw_panel(output, status, alert, len(shown), len(faces),
                            alert_count, fps)

        # evidence: save the annotated frame (at most once per cooldown)
        if alert and (time.time() - last_snapshot) > SNAPSHOT_COOLDOWN:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(ALERT_DIR,
                                f"alert_{stamp}_faces{len(faces)}.jpg")
            cv2.imwrite(path, output)
            last_snapshot = time.time()
            print("Snapshot saved:", path)

        # show windows and handle keys
        cv2.imshow("Smart Room Monitor", output)
        if show_mask:
            cv2.imshow("Motion Mask", mask)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):             # forget the background, start over
            bg = make_bg_subtractor()
            tracker = CentroidTracker(MAX_DISAPPEARED, MAX_DISTANCE,
                                        TRAIL_LENGTH, MIN_HITS)
            frame_idx = 0
        elif key == ord("d"):
            show_mask = not show_mask
            if not show_mask:
                cv2.destroyWindow("Motion Mask")
        elif key == ord("s"):
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(os.path.join(ALERT_DIR, f"manual_{stamp}.jpg"), output)
            print("Manual snapshot saved.")
        elif key == ord("f"):
            save_figures(frame, gray, blur, eq, mask, output)
        elif key == ord("1"):
            tests["correct"] += 1
            print("Logged: correct detection", tests)
        elif key == ord("2"):
            tests["false_alarm"] += 1
            print("Logged: false alarm", tests)
        elif key == ord("3"):
            tests["miss"] += 1
            print("Logged: miss", tests)

    # ---------------------------- clean up ----------------------------
    cap.release()
    cv2.destroyAllWindows()

    avg_fps = sum(fps_samples) / len(fps_samples) if fps_samples else 0
    total = sum(tests.values())
    print("\n===== SESSION SUMMARY (copy into your report) =====")
    print(f"Average FPS      : {avg_fps:.1f}")
    print(f"Alerts raised    : {alert_count}")
    print(f"Tests logged     : {total}")
    print(f"Correct          : {tests['correct']}")
    print(f"False alarms     : {tests['false_alarm']}")
    print(f"Misses           : {tests['miss']}")


if __name__ == "__main__":
    main()