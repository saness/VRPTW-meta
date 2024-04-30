import copy

import numpy as np
import random

from pip._internal.utils.misc import tabulate

from graph import Graph
from Agent import Agent
from threading import Thread, Event
import time


class ACO:
    def __init__(self, graph: Graph, ants=10, maximum_iteration=200, beta=2, q0=0.1):
        super()
        # The location and service time information of graph nodes
        self.graph = graph
        # ants_num number of ants
        self.ants = ants
        # max_iter maximum number of iterations
        self.maximum_iteration = maximum_iteration
        # vehicle_capacity represents the maximum load of each vehicle
        self.load = graph.vehicle_capacity
        # beta heuristic information importance
        self.beta = beta
        # q0 represents the probability of directly selecting the next point with the highest probability
        self.q0 = q0
        # best path
        self.best_distance = None
        self.best_path = None
        self.best_vehicle_number = None

    def find_best_path(self, paths_distance, ants, start_iteration, iter, start_time):
        best_index = np.argmin(paths_distance)
        if self.best_path is None or paths_distance[best_index] < self.best_distance:
            best_ant = ants[int(best_index)]
            self.best_path, self.best_distance = best_ant.path,paths_distance[best_index]
            self.best_vehicle_number = self.best_path.count(0) - 1
            start_iteration = iter
            time_running = time.time() - start_time
            print("|     {}     |      {}     |   {:.3f}  |".format(iter, self.best_distance, time_running))

        return self.best_path, self.best_distance, start_iteration

    def ant_colony_optimization(self):
        """
        The most basic ant colony algorithm
        :return:
        """
        print('----------------------------------------------------')
        print('| Iteration |          Distance          |   Time  |')
        print('-----------------------------------------------------')
        start_time = time.time()
        # The maximum number of iterations
        start_iteration = 0
        for iter in range(self.maximum_iteration):
            # Set the current vehicle load, current travel distance, and current time for each ant
            ants = []
            for _ in range(self.ants):
                ants.append(Agent(self.graph))

            for j in range(self.ants):
                # Ant needs to visit all customers
                while not ants[j].is_empty():
                    next_index = self.chose_next(ants[j])
                    # Determine whether the constraint conditions are still satisfied after adding the position.
                    # If not, select it again and then make the judgment again.
                    checked = ants[j].check(next_index)
                    if not checked and not ants[j].check(next_index := self.chose_next(ants[j])):next_index = 0

                    # Update ant path
                    ants[j].travel_to_next(next_index)
                    self.graph.update_local_pheromone(ants[j].index, next_index)

                # Finally return to 0 position
                ants[j].travel_to_next(0)
                self.graph.update_local_pheromone(ants[j].index, 0)
                # ants[j].local_search_procedure()



            # Calculate the path length of all ants
            paths_distance = np.array(list(map(lambda ant: ant.total_distance, ants)))

            # Record the current best path
            self.best_path,self.best_distance, start_iteration \
                = self.find_best_path(paths_distance, ants, start_iteration, iter, start_time)
            self.apply_local_search(ants, self.graph.customer_distance_matrix)
            # Update pheromone table
            self.graph.update_global_pheromone(self.best_path, self.best_distance)

            # self.local_search_procedure(ants)

            if iter - start_iteration > 500:
                print('\n')
                print('Cannot find better solution in %d iteration' % 100)
                break

        print('\n')
        print(f'Final best path distance is: {self.best_distance}')
        print(f'Number of vehicles is: {self.best_vehicle_number}')
        print(f'Algorithm runtime: {time.time() - start_time:.3f} seconds')



    def calculate_transition_probability(self, index, indexes):
        transition_probability = self.graph.pheromone_matrix[index][indexes] * \
                                 np.power(self.graph.information_matrix[index][indexes], self.beta)
        return transition_probability / np.sum(transition_probability)

    def chose_next(self, ant):
        index = ant.index
        indexes = ant.indexes

        transition_probability = self.calculate_transition_probability(index, indexes)

        # Use the roulette algorithm if random number < q0, else choose maximum probability index
        maximum_probability_index = np.argmax(transition_probability)
        next_index = indexes[maximum_probability_index] if np.random.rand() < self.q0 else ACO.roulette_selection(indexes,
                                                                                                 transition_probability)

        return next_index



    @staticmethod
    def roulette_selection(indexes, transition_probability):
        """
        Roulette
        :param indexes: a list of N index (list or tuple)
        :param transition_probability:
        :return: selected index
        """
        # calculate N and max fitness value
        N = len(indexes)
        # normalize
        normal_transition_prob = transition_probability/np.sum(transition_probability)

        # select: O(1)
        while True:
            # randomly select an individual with uniform probability
            index = int(N * random.random())
            if random.random() <= normal_transition_prob[index]: return indexes[index]

    def two_opt_move(self, route, distance_matrix):
        best_route = route.copy()
        best_distance = self.calculate_route_distance(best_route, distance_matrix)

        for i in range(1, len(route) - 2):
            for j in range(i + 1, len(route)):
                new_route = route.copy()
                new_route[i:j] = reversed(new_route[i:j])
                new_distance = self.calculate_route_distance(new_route, distance_matrix)

                if new_distance < best_distance:
                    best_route = new_route
                    best_distance = new_distance

        return best_route, best_distance

    def calculate_route_distance(self,route, distance_matrix):
        total_distance = 0
        for i in range(len(route) - 1):
            node1, node2 = route[i], route[i + 1]
            total_distance += distance_matrix[node1][node2]
        return total_distance

    def apply_local_search(self, ants, distance_matrix):
        for ant in ants:
            path = ant.path
            routes = self.get_routes_from_path(path)
            improved_routes = []

            for route in routes:
                improved_route, improved_distance = self.two_opt_move(route, distance_matrix)
                improved_routes.append(improved_route)

            ant.path = self.flatten_routes(improved_routes)
            ant.total_distance = self.calculate_path_distance(ant.path, distance_matrix)

    def get_routes_from_path(self, path):
        routes = []
        current_route = []
        for node in path:
            if node == 0:
                if current_route:
                    routes.append(current_route)
                    current_route = []
            else:
                current_route.append(node)
        if current_route:
            routes.append(current_route)
        return routes

    def flatten_routes(self, routes):
        flattened_path = []
        for route in routes:
            flattened_path.extend(route)
            flattened_path.append(0)
        return flattened_path[:-1]


    def calculate_path_distance(self,path, distance_matrix):
        total_distance = 0
        for i in range(len(path) - 1):
            node1, node2 = path[i], path[i + 1]
            total_distance += distance_matrix[node1][node2]
        return total_distance

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
    #
    # def local_search_once(self, graph: Graph, travel_path: list, travel_distance: float, i_start):
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
    #                                 new_path_distance = self.calculate_total_travel_distance(graph, new_path)
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
    # def local_search_procedure(self, ants):
    #     """
    #     Use cross to perform a local search on the current travel_path that has visited all nodes in the graph.
    #
    #     :return:
    #     """
    #     for ant in ants:
    #         new_path = copy.deepcopy(ant.path)
    #         new_path_distance = ant.total_distance
    #         times = 1
    #         count = 0
    #         i_start = 1
    #         while count < times:
    #             temp_path, temp_distance, temp_i = self.local_search_once(self.graph, new_path, new_path_distance, i_start)
    #             if temp_path is not None:
    #                 count += 1
    #
    #                 del new_path, new_path_distance
    #                 new_path = temp_path
    #                 new_path_distance = temp_distance
    #
    #                 # Set i_start
    #                 i_start = (i_start + 1) % (new_path.count(0) - 1)
    #                 i_start = max(i_start, 1)
    #             else:
    #                 break
    #
    #         ant.path = new_path
    #         ant.total_distance = new_path_distance
    #
