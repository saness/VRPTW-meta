import numpy as np

class Customer:
    def __init__(self, id, x, y, demand, ready_time, due_date, service_time):
        super()
        self.id = id
        self.x = x
        self.y = y
        self.demand = demand
        self.ready_time = ready_time
        self.due_date = due_date
        self.service_time = service_time

        if self.id == 0:
            self.is_depot = True
        else:
            self.is_depot = False
