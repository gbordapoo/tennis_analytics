import cv2


def draw_players(frame, near_player=None, far_player=None):
    if near_player is not None:
        bbox = near_player[2]
        x1, y1, x2, y2 = [int(v) for v in bbox]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
        cv2.putText(frame, "near", (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

    if far_player is not None:
        bbox = far_player[2]
        x1, y1, x2, y2 = [int(v) for v in bbox]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(frame, "far", (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    return frame


def draw_ball(frame, balls):
    for bbox in balls:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        cv2.circle(frame, (cx, cy), 5, (0, 165, 255), -1)
    return frame


def draw_keypoints(frame, keypoints):
    for kp in keypoints:
        x, y = int(kp[0]), int(kp[1])
        cv2.circle(frame, (x, y), 5, (0, 255, 255), -1)
    return frame
