from graph import Graph


class Agent:
    def __init__(self, graph: Graph, start = 0):
        super()
        self.graph = graph
        self.index = start
        self.load = 0
        self.time = 0
        self.path = [start]
        self.arrival_time = [0]
        self.indexes = list(range(graph.customer_numbers))
        self.indexes.remove(start)
        self.total_distance = 0

    def travel_to_next(self, next_index):
        self.path.append(next_index)
        self.total_distance += self.graph.customer_distance_matrix[self.index][next_index]

        distance = self.graph.customer_distance_matrix[self.index][next_index]
        self.arrival_time.append(self.time + distance)

        if self.graph.customers[next_index].is_depot:
            self.load = 0
            self.time = 0
        else:
            self.load += self.graph.customers[next_index].demand
            service_time = self.graph.customers[next_index].service_time
            waiting_time = max(self.graph.customers[next_index].ready_time - self.time - distance, 0)
            self.time += distance + waiting_time + service_time
            self.indexes.remove(next_index)

        self.index = next_index

    def check(self, next_index):
        demand = self.graph.customers[next_index].demand
        if self.load + demand > self.graph.vehicle_capacity:
            return False

        distance = self.graph.customer_distance_matrix[self.index][next_index]
        waiting_time =  max(self.graph.customers[next_index].ready_time - self.time - distance, 0)
        service_time = self.graph.customers[next_index].service_time

        total_time = self.time + distance + waiting_time + service_time + self.graph.customer_distance_matrix[next_index][0]
        due_time = self.graph.customers[0].due_date

        if total_time > due_time or self.time + distance > due_time:
            return False

        return True

    def index_to_visit_empty(self):
        return len(self.indexes) == 0

