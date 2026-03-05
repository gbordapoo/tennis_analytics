class CourtGeometry:
    def __init__(self, keypoints) -> None:
        self.keypoints = keypoints

    def filter_players(self, players):
        valid = []
        left_sideline = self.keypoints[0]
        right_sideline = self.keypoints[3]
        for bbox in players:
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2
            if cx > left_sideline[0] and cx < right_sideline[0]:
                valid.append((cx, cy, bbox))
        return valid

    def assign_players(self, players):
        if len(players) < 2:
            return None, None
        players = sorted(players, key=lambda x: x[1])
        far_player = players[0]
        near_player = players[-1]
        return near_player, far_player
