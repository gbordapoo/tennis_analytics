import numpy as np


def compute_speed(p1, p2, fps):
    dist = np.linalg.norm(np.array(p1) - np.array(p2))
    time = 1 / fps
    return dist / time
