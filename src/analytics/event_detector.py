class EventDetector:
    def detect_hits(self, ball_positions):
        hits = []
        for i in range(2, len(ball_positions)):
            dy1 = ball_positions[i][1] - ball_positions[i - 1][1]
            dy2 = ball_positions[i - 1][1] - ball_positions[i - 2][1]
            if dy1 * dy2 < 0:
                hits.append(i)
        return hits
