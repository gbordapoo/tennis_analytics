import argparse
import json
from pathlib import Path

import cv2


COURT_LENGTH_M = 23.77
WIDTH_BY_TYPE = {
    "singles": 8.23,
    "doubles": 10.97,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Manual click calibration helper")
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--output", default="calibration.json", help="Output JSON path")
    return parser.parse_args()


def collect_four_points(frame):
    window_name = "Manual Calibration"
    display = frame.copy()
    clicked_points = []

    def on_mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN and len(clicked_points) < 4:
            point = [int(x), int(y)]
            clicked_points.append(point)
            idx = len(clicked_points)
            cv2.circle(display, tuple(point), 6, (0, 0, 255), -1)
            cv2.putText(
                display,
                str(idx),
                (x + 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

    print("Click 4 court corners in this order:")
    print("1) Near left corner")
    print("2) Near right corner")
    print("3) Far left corner")
    print("4) Far right corner")

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, on_mouse)

    while True:
        cv2.imshow(window_name, display)
        key = cv2.waitKey(20) & 0xFF
        if len(clicked_points) == 4:
            break
        if key == 27:  # ESC
            cv2.destroyWindow(window_name)
            return None

    cv2.destroyWindow(window_name)
    return clicked_points


def prompt_court_type():
    raw = input("Court type? (singles/doubles) [default: singles]: ").strip().lower()
    if raw == "":
        return "singles"
    if raw in WIDTH_BY_TYPE:
        return raw
    print("Invalid selection, defaulting to singles.")
    return "singles"


def main():
    args = parse_args()
    video_path = Path(args.video)
    output_path = Path(args.output)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Could not read first frame from video: {video_path}")

    pixel_points = collect_four_points(frame)
    if pixel_points is None or len(pixel_points) < 4:
        print("Calibration canceled before 4 clicks. No file saved.")
        return

    court_type = prompt_court_type()
    width = WIDTH_BY_TYPE[court_type]

    payload = {
        "court_type": court_type,
        "pixel_points": pixel_points,
        "world_points_m": [
            [0.0, 0.0],
            [float(width), 0.0],
            [0.0, float(COURT_LENGTH_M)],
            [float(width), float(COURT_LENGTH_M)],
        ],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved calibration to {output_path}")


if __name__ == "__main__":
    main()
