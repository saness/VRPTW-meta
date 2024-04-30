import copy
from threading import Event

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

    def helper_function(self, next_index, distance):
        service_time = self.graph.customers[next_index].service_time
        waiting_time = max(self.graph.customers[next_index].ready_time - self.time - distance, 0)
        all_time = distance + waiting_time + service_time
        return all_time

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
            all_time = self.helper_function(next_index, distance)
            self.time += all_time
            self.indexes.remove(next_index)

        self.index = next_index

    def check(self, next_index):
        demand = self.graph.customers[next_index].demand
        if self.load + demand > self.graph.vehicle_capacity:
            return False

        distance = self.graph.customer_distance_matrix[self.index][next_index]
        all_time = self.helper_function(next_index, distance)

        total_time = self.time + all_time + self.graph.customer_distance_matrix[next_index][0]
        due_time = self.graph.customers[0].due_date

        if total_time > due_time or self.time + distance > due_time:
            return False

        return True

    def is_empty(self):
        return len(self.indexes) == 0

    # def clear(self):
    #     self.path.clear()
    #     self.indexes.clear()
    #
    # @staticmethod
    # def calculate_total_travel_distance(graph: Graph, travel_path):
    #     distance = 0
    #     current_ind = travel_path[0]
    #     for next_ind in travel_path[1:]:
    #         distance += graph.customer_distance_matrix[current_ind][next_ind]
    #         current_ind = next_ind
    #     return distance
    #
    # @staticmethod
    # def local_search_once(graph: Graph, travel_path: list, travel_distance: float, i_start):
    #
    #     # Find the locations of all depots in path
    #     depot_ind = []
    #     for ind in range(len(travel_path)):
    #         if graph.customers[travel_path[ind]].is_depot:
    #             depot_ind.append(ind)
    #
    #     # Divide self.travel_path into multiple segments, each segment starts with depot and ends with depot, called route
    #     for i in range(i_start, len(depot_ind)):
    #         for j in range(i + 1, len(depot_ind)):
    #
    #             for start_a in range(depot_ind[i - 1] + 1, depot_ind[i]):
    #                 for end_a in range(start_a, min(depot_ind[i], start_a + 6)):
    #                     for start_b in range(depot_ind[j - 1] + 1, depot_ind[j]):
    #                         for end_b in range(start_b, min(depot_ind[j], start_b + 6)):
    #                             if start_a == end_a and start_b == end_b:
    #                                 continue
    #                             new_path = []
    #                             new_path.extend(travel_path[:start_a])
    #                             new_path.extend(travel_path[start_b:end_b + 1])
    #                             new_path.extend(travel_path[end_a:start_b])
    #                             new_path.extend(travel_path[start_a:end_a])
    #                             new_path.extend(travel_path[end_b + 1:])
    #
    #                             depot_before_start_a = depot_ind[i - 1]
    #
    #                             depot_before_start_b = depot_ind[j - 1] + (end_b - start_b) - (end_a - start_a) + 1
    #                             if not graph.customers[new_path[depot_before_start_b]].is_depot:
    #                                 raise RuntimeError('error')
    #
    #                             # Determine whether the changed route a is feasible
    #                             success_route_a = False
    #                             check_ant = Agent(graph, new_path[depot_before_start_a])
    #                             for ind in new_path[depot_before_start_a + 1:]:
    #                                 if check_ant.check(ind):
    #                                     check_ant.travel_to_next(ind)
    #                                     if graph.customers[ind].is_depot:
    #                                         success_route_a = True
    #                                         break
    #                                 else:
    #                                     break
    #
    #                             check_ant.clear()
    #                             del check_ant
    #
    #                             # Determine whether the changed route b is feasible
    #                             success_route_b = False
    #                             check_ant = Agent(graph, new_path[depot_before_start_b])
    #                             for ind in new_path[depot_before_start_b + 1:]:
    #                                 if check_ant.check(ind):
    #                                     check_ant.travel_to_next(ind)
    #                                     if graph.customers[ind].is_depot:
    #                                         success_route_b = True
    #                                         break
    #                                 else:
    #                                     break
    #                             check_ant.clear()
    #                             del check_ant
    #
    #                             if success_route_a and success_route_b:
    #                                 new_path_distance = Agent.calculate_total_travel_distance(graph, new_path)
    #                                 if new_path_distance < travel_distance:
    #                                     # print('success to search')
    #
    #                                     # Determine whether the changed route b is to delete one of the depots connected together in the path.
    #                                     # It is feasible.
    #                                     for temp_ind in range(1, len(new_path)):
    #                                         if graph.customers[new_path[temp_ind]].is_depot and graph.customers[new_path[temp_ind - 1]].is_depot:
    #                                             new_path.pop(temp_ind)
    #                                             break
    #                                     return new_path, new_path_distance, i
    #                             else:
    #                                 new_path.clear()
    #
    #     return None, None, None
    #
    # def local_search_procedure(self):
    #     """
    #     Use cross to perform a local search on the current travel_path that has visited all nodes in the graph.
    #
    #     :return:
    #     """
    #     new_path = copy.deepcopy(self.path)
    #     new_path_distance = self.total_distance
    #     times = 1
    #     count = 0
    #     i_start = 1
    #     while count < times:
    #         temp_path, temp_distance, temp_i = Agent.local_search_once(self.graph, new_path, new_path_distance, i_start)
    #         if temp_path is not None:
    #             count += 1
    #
    #             del new_path, new_path_distance
    #             new_path = temp_path
    #             new_path_distance = temp_distance
    #
    #             # Set i_start
    #             i_start = (i_start + 1) % (new_path.count(0) - 1)
    #             i_start = max(i_start, 1)
    #         else:
    #             break
    #
    #     self.path = new_path
    #     self.total_distance = new_path_distance

