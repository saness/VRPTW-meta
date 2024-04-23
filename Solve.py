import numpy as np
import random
from graph import Graph
from Agent import Agent
from threading import Thread
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
            # print('\n')
            # print(f'[iteration {iter}]: The improved path distance is {self.best_distance}')
            # print(f'It took {time.time() - start_time:.3f} second running the ant colony algorithm')

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

            # Calculate the path length of all ants
            paths_distance = np.array(list(map(lambda ant: ant.total_distance, ants)))

            # Record the current best path
            self.best_path,self.best_distance, start_iteration \
                = self.find_best_path(paths_distance, ants, start_iteration, iter, start_time)

            # Update pheromone table
            self.graph.update_global_pheromone(self.best_path, self.best_distance)

            if iter - start_iteration > 100:
                print('\n')
                print('Cannot find better solution in %d iteration' % 100)
                break

        print('\n')
        print(f'Final best path distance is: {self.best_distance}')
        print(f'Number of vehicles is: {self.best_vehicle_number}')
        print(f'Algorithm runtime: {time.time() - start_time:.3f}')



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

