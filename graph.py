"""
file: graph.py
description: This program is a blueprint for Graph to implement the Ant Colony Optimization
language: python3
author: Sanish Suwal(ss4657@rit.edu), Jay Nair(an1147@rit.edu), Bhavdeep Khileri(bk2281@rit.edu)
"""
from dataextract import *
from utils import helper_function_two

class Graph:
    def __init__(self, file_path, rho = 0.1):
        """
        Constructor for the class
        :param file_path: path of the file
        :param rho: pheromone evaporation rate
        """
        super()
        # pheromone evaporation rate
        self.rho = rho
        self.customer_numbers, self.customers, self.customer_distance_matrix, self.vehicle_number, self.vehicle_capacity \
            = create_customer(file_path)
        self.nearest_travel_path, self.pheromone_value = self.nearest_heuristic()
        # initial pheromone value
        self.pheromone_value = (1/self.pheromone_value * self.customer_numbers)
        # pheromone matrix for the ACO graph
        self.pheromone_matrix = np.ones((self.customer_numbers, self.customer_numbers)) * self.pheromone_value
        # Closeness (n_ij)
        self.information_matrix = 1/ self.customer_distance_matrix



    def nearest_heuristic(self,maximum_vehicle_number=None):
        """
        Calculates the closest node heuristic from a current node
        :param maximum_vehicle_number: maximum number of vehicles
        :return: path and total distance to nearest node
        """
        # nodes to visit
        indexes = []
        # current index
        index = 0
        # current load
        load = 0
        # current time
        time = 0
        # total distance
        total_distance = 0
        path = [0]

        for i in range(1, self.customer_numbers):
            indexes.append(i)

        if maximum_vehicle_number is None:
            maximum_vehicle_number = self.customer_numbers

        while len(indexes) > 0 and maximum_vehicle_number > 0:
            closest_index = self.calculate_closest_index(indexes, index, load, time)

            if closest_index is not None:
                load += self.customers[closest_index].demand
                distance = self.customer_distance_matrix[index][closest_index]

                all_time = helper_function_two(closest_index, time, distance, self.customers)
                time += all_time

                total_distance += self.customer_distance_matrix[index][closest_index]
                # add node to path
                path.append(closest_index)
                # change the current index to nearest node
                index = closest_index
                # remove visited index
                indexes.remove(closest_index)
            else:
                total_distance += self.customer_distance_matrix[index][0]
                load = 0
                time = 0
                path.append(0)
                index = 0
                maximum_vehicle_number -= 1

        total_distance += self.customer_distance_matrix[index][0]
        path.append(0)
        vehicle_number = path.count(0) - 1

        return path, total_distance

    def calculate_closest_index(self, indexes, index, load, time):
        """
        Calculates the closest index to the current one
        :param indexes: all the nodes in graph
        :param index: current index
        :param load: load
        :param time: time
        :return: nearest index or node to the current one
        """
        closest_index = None
        closest_distance = None

        for i in indexes:
            if load + self.customers[i].demand > self.vehicle_capacity:
                continue

            distance = self.customer_distance_matrix[index][i]
            all_time = helper_function_two(i, time, distance, self.customers)
            total_time = time + all_time + self.customer_distance_matrix[i][0]
            # Checks if you can return to service store after visiting a passenger or
            # Checks if the vehicle reaches after due time
            if total_time > self.customers[0].due_date or time + distance > self.customers[i].due_date:
                continue

            # comparing nearest node with current next node
            if closest_distance is None or self.customer_distance_matrix[index][i] < closest_distance:
                closest_distance = self.customer_distance_matrix[index][i]
                closest_index = i

        return closest_index


    def update_local_pheromone(self, i, j):
        """
        Updates the pheromone density between two nodes or customers in an arc
        :param i: index of first node or customer
        :param j:index of second node or customer
        :return: updated pheromone density between two nodes
        """
        self.pheromone_matrix[i][j] = (1-self.rho) * self.pheromone_matrix[i][j] + self.rho * self.pheromone_value

    def update_global_pheromone(self, path, distance):
        """
        Updates the pheromone value when the ants retrace the same tour backwards
        :param path: the path taken to destination
        :param distance: total distance of the path
        :return:
        """
        pheromone_increase = self.rho/distance
        self.pheromone_matrix *= (1-self.rho)

        index = path[0]
        for i in path[1:]:
            self.pheromone_matrix[index][i] += pheromone_increase
            index = i





