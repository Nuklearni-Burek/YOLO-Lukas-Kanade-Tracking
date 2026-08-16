# Monochrome IR YOLO + Lucas-Kanade Pedestrian Tracking

Pedestrian detection and tracking in **monochrome infrared (IR) imagery** using a custom-trained YOLO model and Lucas-Kanade (LK) optical flow.

The system is designed for a **stationary camera** and focuses on **very small pedestrians**, some only **6–10 pixels wide**.

## Overview

YOLO is used as the primary pedestrian detector. When YOLO temporarily loses a pedestrian, Lucas-Kanade optical flow continues tracking the existing bounding box for a limited number of frames.

```text
Monochrome IR Frame
        │
        ▼
   YOLO Detection
        │
        ├── Detected ──► Initialize / update LK
        │
        └── Not detected
                 │
                 ▼
            LK Tracking
                 │
                 ▼
          Move Bounding Box
                 │
                 ▼
          Wait for YOLO
```

YOLO always has priority over LK. Optical flow cannot create new pedestrian tracks independently, which helps prevent false bounding boxes.

## Model Performance

| Metric | Score |
|---|---:|
| Precision | **0.976** |
| Recall | **0.942** |
| mAP50 | **0.972** |
| mAP50-95 | **0.689** |

The model achieves high precision and recall, while the lower mAP50-95 indicates that there is still room for improvement in precise bounding-box localization.

## Tracking

Lucas-Kanade optical flow is used to estimate pedestrian movement between YOLO detections.

The tracker:

- Uses up to **10 feature points**
- Requires at least **2 valid points**
- Searches around the previous bounding box
- Supports movement in different directions
- Tracks for up to **5 frames** without YOLO
- Removes overlapping duplicate tracks
- Gives priority to YOLO detections

This is particularly useful when small pedestrians temporarily disappear from YOLO due to their size, contrast changes, or partial occlusion.

## Input

The system processes individual PNG frames from a monochrome infrared stream:

```text
nearir_frames/
├── frame_000001.png
├── frame_000002.png
├── frame_000003.png
└── ...
```

No video file is required.

## Output

Processed frames are saved with bounding boxes, tracking IDs, detection confidence, and the tracking source (YOLO or LK).

```text
tracked_frames/
├── frame_000001.png
├── frame_000002.png
└── ...
```

## Requirements

```bash
pip install ultralytics opencv-python numpy
```

## Usage

Place the trained model in the project directory:

```text
version1_100ep.pt
```

Then run:

```bash
python detect_track.py
```

## Summary

**YOLO detects; Lucas-Kanade temporarily tracks.**

The goal is to improve pedestrian tracking continuity in monochrome infrared imagery, particularly for extremely small objects that are difficult to detect consistently.
