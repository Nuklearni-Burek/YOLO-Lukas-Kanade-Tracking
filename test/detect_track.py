import cv2
import numpy as np
import glob
import os
from ultralytics import YOLO


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "version1_100ep.pt"

FRAMES_FOLDER = "nearir_frames"

# YOLO confidence
CONF_THRESHOLD = 0.15

# COCO person class
PERSON_CLASS = 0

# ------------------------------------------------------------
# LK PARAMETERS
# ------------------------------------------------------------

# Very small objects -> don't demand many points
MAX_CORNERS = 10
MIN_POINTS = 2

LK_PARAMS = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(
        cv2.TERM_CRITERIA_EPS |
        cv2.TERM_CRITERIA_COUNT,
        20,
        0.01
    )
)

# Search around the previous BB.
#
# This is important because a 6-10 pixel-wide person may not
# contain enough useful features by itself.
SEARCH_PADDING_X = 10
SEARCH_PADDING_Y = 10

# Maximum number of frames LK can predict without YOLO
MAX_LK_FRAMES = 5

# ------------------------------------------------------------
# MATCHING
# ------------------------------------------------------------

# If YOLO BB and LK BB overlap this much,
# they are considered the same pedestrian.
YOLO_MATCH_IOU = 0.10

# If two tracks overlap this much,
# treat them as duplicates.
DUPLICATE_IOU = 0.40

# Maximum amount of disagreement between optical-flow
# points before rejecting the motion estimate.
MAX_MOTION_ERROR = 5.0


# ============================================================
# LOAD MODEL
# ============================================================

model = YOLO(MODEL_PATH)


# ============================================================
# LOAD PNG FRAMES
# ============================================================

frame_paths = sorted(
    glob.glob(
        os.path.join(
            FRAMES_FOLDER,
            "*.png"
        )
    )
)

if len(frame_paths) == 0:
    raise RuntimeError(
        f"No PNG files found in: {FRAMES_FOLDER}"
    )

print(f"Found {len(frame_paths)} frames")


# ============================================================
# TRACK STRUCTURE
# ============================================================

tracks = []

next_track_id = 0

previous_gray = None


# ============================================================
# IOU
# ============================================================

def bbox_iou(box1, box2):

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])

    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = (
        max(0, x2 - x1) *
        max(0, y2 - y1)
    )

    area1 = (
        max(0, box1[2] - box1[0]) *
        max(0, box1[3] - box1[1])
    )

    area2 = (
        max(0, box2[2] - box2[0]) *
        max(0, box2[3] - box2[1])
    )

    union = area1 + area2 - intersection

    if union <= 0:
        return 0.0

    return intersection / union


# ============================================================
# GET FEATURES
# ============================================================

def get_features(gray, bbox):

    x1, y1, x2, y2 = bbox

    h, w = gray.shape

    # --------------------------------------------------------
    # Expand search region around object
    # --------------------------------------------------------

    x1 = int(
        max(
            0,
            x1 - SEARCH_PADDING_X
        )
    )

    y1 = int(
        max(
            0,
            y1 - SEARCH_PADDING_Y
        )
    )

    x2 = int(
        min(
            w - 1,
            x2 + SEARCH_PADDING_X
        )
    )

    y2 = int(
        min(
            h - 1,
            y2 + SEARCH_PADDING_Y
        )
    )

    if x2 <= x1 or y2 <= y1:
        return None

    roi = gray[y1:y2, x1:x2]

    if roi.size == 0:
        return None

    # --------------------------------------------------------
    # Find corners/features
    # --------------------------------------------------------

    points = cv2.goodFeaturesToTrack(
        roi,
        maxCorners=MAX_CORNERS,
        qualityLevel=0.005,
        minDistance=1,
        blockSize=3
    )

    if points is None:
        return None

    # Convert ROI coordinates back into
    # full-image coordinates

    points[:, 0, 0] += x1
    points[:, 0, 1] += y1

    return points


# ============================================================
# OPTICAL FLOW
# ============================================================

def calculate_optical_flow(
    old_gray,
    new_gray,
    points
):

    if points is None:
        return None, None

    if len(points) < MIN_POINTS:
        return None, None

    new_points, status, error = (
        cv2.calcOpticalFlowPyrLK(
            old_gray,
            new_gray,
            points,
            None,
            **LK_PARAMS
        )
    )

    if new_points is None:
        return None, None

    status = status.flatten()

    good_old = points[status == 1]
    good_new = new_points[status == 1]

    if len(good_old) < MIN_POINTS:
        return None, None

    # Convert to (N, 2)
    good_old = good_old.reshape(-1, 2)
    good_new = good_new.reshape(-1, 2)

    # --------------------------------------------------------
    # Displacement of every point
    # --------------------------------------------------------

    displacement = (
        good_new - good_old
    )

    dx_values = displacement[:, 0]
    dy_values = displacement[:, 1]

    # --------------------------------------------------------
    # Median movement
    # --------------------------------------------------------

    dx = np.median(dx_values)
    dy = np.median(dy_values)

    # --------------------------------------------------------
    # Check whether points agree
    # --------------------------------------------------------

    dx_error = np.median(
        np.abs(dx_values - dx)
    )

    dy_error = np.median(
        np.abs(dy_values - dy)
    )

    if (
        dx_error > MAX_MOTION_ERROR
        or
        dy_error > MAX_MOTION_ERROR
    ):
        return None, None

    return (
        float(dx),
        float(dy)
    ), good_new.reshape(-1, 1, 2)


# ============================================================
# MAIN LOOP
# ============================================================

for frame_number, frame_path in enumerate(
    frame_paths
):

    frame = cv2.imread(frame_path)

    if frame is None:
        continue

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    # ========================================================
    # STEP 1
    # YOLO DETECTION
    # ========================================================

    results = model(
        frame,
        conf=CONF_THRESHOLD,
        verbose=False
    )

    yolo_detections = []

    for result in results:

        if result.boxes is None:
            continue

        boxes = (
            result.boxes.xyxy
            .cpu()
            .numpy()
        )

        classes = (
            result.boxes.cls
            .cpu()
            .numpy()
        )

        confidences = (
            result.boxes.conf
            .cpu()
            .numpy()
        )

        for box, cls, confidence in zip(
            boxes,
            classes,
            confidences
        ):

            if int(cls) != PERSON_CLASS:
                continue

            yolo_detections.append({
                "bbox": box.astype(float),
                "confidence": float(
                    confidence
                )
            })


    # ========================================================
    # STEP 2
    # MOVE EXISTING TRACKS USING LK
    # ========================================================

    if previous_gray is not None:

        for track in tracks:

            if not track["active"]:
                continue

            motion, new_points = (
                calculate_optical_flow(
                    previous_gray,
                    gray,
                    track["points"]
                )
            )

            # ------------------------------------------------
            # LK failed
            # ------------------------------------------------

            if motion is None:

                track["active"] = False
                continue

            dx, dy = motion

            # ------------------------------------------------
            # Move BB
            # ------------------------------------------------

            x1, y1, x2, y2 = (
                track["bbox"]
            )

            track["bbox"] = np.array([
                x1 + dx,
                y1 + dy,
                x2 + dx,
                y2 + dy
            ])

            track["points"] = new_points

            track["lk_frames"] += 1

            track["source"] = "LK"

            # ------------------------------------------------
            # LK can only survive a few frames
            # ------------------------------------------------

            if (
                track["lk_frames"]
                > MAX_LK_FRAMES
            ):

                track["active"] = False


    # ========================================================
    # STEP 3
    # MATCH YOLO WITH EXISTING TRACKS
    # ========================================================

    matched_yolo = set()

    for track in tracks:

        if not track["active"]:
            continue

        best_iou = 0.0
        best_index = None

        for i, detection in enumerate(
            yolo_detections
        ):

            if i in matched_yolo:
                continue

            iou = bbox_iou(
                track["bbox"],
                detection["bbox"]
            )

            if iou > best_iou:

                best_iou = iou
                best_index = i

        # ----------------------------------------------------
        # YOLO FOUND THE OBJECT AGAIN
        # ----------------------------------------------------

        if (
            best_index is not None
            and
            best_iou >= YOLO_MATCH_IOU
        ):

            detection = (
                yolo_detections[
                    best_index
                ]
            )

            # YOLO ALWAYS WINS
            track["bbox"] = (
                detection["bbox"].copy()
            )

            track["confidence"] = (
                detection["confidence"]
            )

            track["source"] = "YOLO"

            track["lk_frames"] = 0

            # Reinitialize LK
            track["points"] = (
                get_features(
                    gray,
                    track["bbox"]
                )
            )

            matched_yolo.add(
                best_index
            )


    # ========================================================
    # STEP 4
    # CREATE TRACKS ONLY FROM YOLO
    # ========================================================

    for i, detection in enumerate(
        yolo_detections
    ):

        if i in matched_yolo:
            continue

        bbox = detection["bbox"]

        duplicate = False

        # Check against all active tracks
        for track in tracks:

            if not track["active"]:
                continue

            iou = bbox_iou(
                bbox,
                track["bbox"]
            )

            if iou >= YOLO_MATCH_IOU:

                duplicate = True
                break

        if duplicate:
            continue

        # ----------------------------------------------------
        # NEW OBJECT
        # ----------------------------------------------------

        points = get_features(
            gray,
            bbox
        )

        tracks.append({

            "id": next_track_id,

            "bbox": bbox.copy(),

            "confidence":
                detection["confidence"],

            "points": points,

            "lk_frames": 0,

            "active": True,

            "source": "YOLO"
        })

        next_track_id += 1


    # ========================================================
    # STEP 5
    # REMOVE OVERLAPPING TRACKS
    # ========================================================

    active_tracks = [
        t for t in tracks
        if t["active"]
    ]

    for i in range(
        len(active_tracks)
    ):

        a = active_tracks[i]

        if not a["active"]:
            continue

        for j in range(
            i + 1,
            len(active_tracks)
        ):

            b = active_tracks[j]

            if not b["active"]:
                continue

            iou = bbox_iou(
                a["bbox"],
                b["bbox"]
            )

            if iou < DUPLICATE_IOU:
                continue

            # ------------------------------------------------
            # YOLO beats LK
            # ------------------------------------------------

            if (
                a["source"] == "YOLO"
                and
                b["source"] == "LK"
            ):

                b["active"] = False

            elif (
                b["source"] == "YOLO"
                and
                a["source"] == "LK"
            ):

                a["active"] = False

            else:

                # Both same source.
                # Keep the older ID.
                if a["id"] < b["id"]:
                    b["active"] = False
                else:
                    a["active"] = False


    # ========================================================
    # STEP 6
    # DRAW
    # ========================================================

    output = frame.copy()

    active_count = 0

    for track in tracks:

        if not track["active"]:
            continue

        active_count += 1

        x1, y1, x2, y2 = (
            track["bbox"]
        )

        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)

        # Keep BB inside image

        x1 = max(
            0,
            min(
                frame.shape[1] - 1,
                x1
            )
        )

        y1 = max(
            0,
            min(
                frame.shape[0] - 1,
                y1
            )
        )

        x2 = max(
            0,
            min(
                frame.shape[1] - 1,
                x2
            )
        )

        y2 = max(
            0,
            min(
                frame.shape[0] - 1,
                y2
            )
        )

        # ----------------------------------------------------
        # LABEL
        # ----------------------------------------------------

        if track["source"] == "YOLO":

            label = (
                f"ID {track['id']} "
                f"YOLO "
                f"{track['confidence']:.2f}"
            )

        else:

            label = (
                f"ID {track['id']} "
                f"LK "
                f"{track['lk_frames']}"
            )

        # ----------------------------------------------------
        # BB
        # ----------------------------------------------------

        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            output,
            label,
            (
                x1,
                max(15, y1 - 5)
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 0),
            1,
            cv2.LINE_AA
        )


    # ========================================================
    # DISPLAY INFORMATION
    # ========================================================

    info = (
        f"Frame: {frame_number + 1}/"
        f"{len(frame_paths)}"
        f" | YOLO detections: "
        f"{len(yolo_detections)}"
        f" | Tracks: {active_count}"
    )

    cv2.putText(
        output,
        info,
        (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 255),
        1,
        cv2.LINE_AA
    )


    # ========================================================
    # SHOW
    # ========================================================

    cv2.imshow(
        "YOLO + Lucas-Kanade",
        output
    )

    key = cv2.waitKey(1)

    if key == 27 or key == ord("q"):
        break


    # IMPORTANT:
    # Current frame becomes previous frame
    previous_gray = gray.copy()


cv2.destroyAllWindows()

print("Finished.")