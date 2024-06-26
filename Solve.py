"""
file: Solve.py
description: This program consists of implementation of local search procedure
language: python3
author: Sanish Suwal(ss4657@rit.edu), Jay Nair(an1147@rit.edu), Bhavdeep Khileri(bk2281@rit.edu)
"""

import numpy as np
import random

from graph import Graph
from Agent import Agent
from figure import plot_customers_and_routes
from localsearch import apply_local_search
import time


class ACO:
    def __init__(self, graph: Graph, ants=10, maximum_iteration=200, beta=2, q0=0.1):
        """
        Constructor to the class
        :param graph: the ant colony graph
        :param ants: ants or vehicles
        :param maximum_iteration: maximum number of iterations
        :param beta: relative importance of heuristic value
        :param q0: probability of selecting next point
        """
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
        # best path distance
        self.best_distance = None
        # best rout
        self.best_path = None
        # least number of vehicles
        self.best_vehicle_number = None

    def find_best_path(self, paths_distance, ants, start_iteration, iter, start_time, i):
        """
        Finds the best path and its corresponding distance
        :param paths_distance: the total distance travelled by ants
        :param ants: ants
        :param start_iteration: starting iteration
        :param iter: the current iteration
        :param start_time: the starting time
        :return: best path, distance and start iteration
        """

        best_index = np.argmin(paths_distance)
        if self.best_path is None or paths_distance[best_index] < self.best_distance:
            best_ant = ants[int(best_index)]
            self.best_path, self.best_distance = best_ant.path,paths_distance[best_index]
            self.best_vehicle_number = self.best_path.count(0) - 1
            start_iteration = iter
            time_running = time.time() - start_time
            # creating images
            plot_customers_and_routes(self.graph.customers, self.best_path, "image.{i}.png".format(i=i))
            i += 1
            print("|     {}     |      {}     |   {:.3f}  |".format(iter, self.best_distance, time_running))

        return self.best_path, self.best_distance, start_iteration, i

    def ant_colony_optimization(self):
        """
        Implements the ant colony algorithm
        :return: best distance,best vehicle number
        """
        print('-----------------------------------------------------')
        print('| Iteration |          Distance          |   Time   |')
        print('-----------------------------------------------------')
        start_time = time.time()
        # The maximum number of iterations
        start_iteration = 0
        i = 0
        for iter in range(self.maximum_iteration):
            # initializing the ants
            ants = []
            # Set the current vehicle load, current travel distance, and current time for each ant
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
            self.best_path,self.best_distance, start_iteration, i \
                = self.find_best_path(paths_distance, ants, start_iteration, iter, start_time, i)

            # local search procedure to refine the paths and distance found by the ants
            apply_local_search(ants, self.graph.customer_distance_matrix)

            # Update global pheromone
            self.graph.update_global_pheromone(self.best_path, self.best_distance)


            if iter - start_iteration > 300:
                print('\n')
                print('Cannot find better solution in %d iteration' % 300)
                break

        print('\n')
        print(f'Final best path distance is: {self.best_distance}')
        print(f'Number of vehicles is: {self.best_vehicle_number}')
        print(f'Algorithm runtime: {time.time() - start_time:.3f} seconds')

        return self.best_distance, self.best_vehicle_number



    def calculate_transition_probability(self, index, indexes):
        """
        Calculates the probability of choosing the next node or customer
        :param index: current index
        :param indexes: all the nodes left to visit
        :return: the probability of choosing the next node
        """
        transition_probability = self.graph.pheromone_matrix[index][indexes] * \
                                 np.power(self.graph.information_matrix[index][indexes], self.beta)
        return transition_probability / np.sum(transition_probability)

    def chose_next(self, ant):
        """
        Chooses the next node or index to travel
        :param ant: ant or vehicle
        :return: the next node or index to visit
        """
        # current index
        index = ant.index
        # indexes to visit
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
        Roulette way of selecting index
        :param indexes: a list of N index or nodes
        :param transition_probability: the transition probability
        :return: index to move to
        """
        # calculate N and max fitness value
        N = len(indexes)
        # normalize
        normal_transition_prob = transition_probability/np.sum(transition_probability)

        while True:
            # randomly select an individual with uniform probability
            index = int(N * random.random())
            if random.random() <= normal_transition_prob[index]: return indexes[index]







