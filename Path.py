import copy


class Path:
    def __init__(self, path, distance):
        if path is None:
            self.path = None
            self.distance = None
            self.vehicle_number = None
        else:
            self.path = copy.deepcopy(path)
            self.distance = copy.deepcopy(distance)
            self.vehicle_number = self.path.count(0) - 1

    def get_path_info(self):
        return self.path, self.distance, self.vehicle_number
