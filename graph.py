"""
file: graph.py
description: This program is a blueprint for Graph to implement the Ant Colony Optimization
language: python3
author: Sanish Suwal(ss4657@rit.edu), Jay Nair(an1147@rit.edu), Bhavdeep Khileri(bk2281@rit.edu)
"""
import numpy as np
from Customer import *

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
            = self.create_customer(file_path)
        self.nearest_travel_path, self.pheromone_value = self.nearest_heuristic()
        # initial pheromone value
        self.pheromone_value = (1/self.pheromone_value * self.customer_numbers)
        # pheromone matrix for the ACO graph
        self.pheromone_matrix = np.ones((self.customer_numbers, self.customer_numbers)) * self.pheromone_value
        # Closeness (n_ij)
        self.information_matrix = 1/ self.customer_distance_matrix


    def create_customer(self, file_path):
        """
        Calculates the customer and vehicle information from the file
        :param file_path: path of the file
        :return: returns customer and vehicle information
        """
        customer_list = []
        with open(file_path, 'rt') as f:
            count = 1
            for line in f:
                if count == 5:
                    vehicle_number, vehicle_capacity = line.split()
                    vehicle_number, vehicle_capacity = int(vehicle_number), int(vehicle_capacity)
                elif count >= 10:
                    customer_list.append(line.split())
                count += 1
        customer_numbers = len(customer_list)
        customers = []

        self.add_customer_to_list(customers, customer_list)

        customer_distance_matrix = np.zeros((customer_numbers, customer_numbers))
        customer_distance_matrix = Graph.calculate_distance_matrix(customer_numbers, customer_distance_matrix, customers)

        return customer_numbers, customers, customer_distance_matrix, vehicle_number, vehicle_capacity

    def add_customer_to_list(self, customers, customer_list):
        """
        Adds customer details  to customer list as Customer class
        :param customers: list of customers as Customer node
        :param customer_list: list of extracted customer details from dataset
        :return:
        """
        for customer in customer_list:
            customers.append(Customer(int(customer[0]), float(customer[1]), float(customer[2]), float(customer[3]),
                                      float(customer[4]), float(customer[5]), float(customer[6])))

    @staticmethod
    def calculate_distance_matrix(customer_numbers, customer_distance_matrix, customers):
        """
        Fills the distance between customers in customer distance matrix
        :param customer_numbers: number of customers
        :param customer_distance_matrix: customer distance matrix
        :param customers:customers or nodes in graph
        :return: filled customer distance matrix
        """
        for i in range(customer_numbers):
            customer_1 = customers[i]
            customer_distance_matrix[i][i] = 1e-8
            for j in range(i + 1, customer_numbers):
                customer_2 = customers[j]
                customer_distance_matrix[i][j] = Graph.calculate_distance(customer_1, customer_2)
                customer_distance_matrix[j][i] = customer_distance_matrix[i][j]

        return customer_distance_matrix

    def helper_function(self, index, time, distance):
        """
        Calculates the total time of distance, waiting and service
        :param index: index where ant is
        :param time: time elapsed
        :param distance: distance between nodes
        :return: total time elapsed
        """
        waiting_time = max(self.customers[index].ready_time - time - distance, 0)
        service_time = self.customers[index].service_time
        all_time = distance + waiting_time + service_time

        return all_time

    def nearest_heuristic(self,maximum_vehicle_number=None):
        """
        Calculates the nearest node heuristic from a current node
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
            nearest_index = self.calculate_nearest_index(indexes, index, load, time)

            if nearest_index is not None:
                load += self.customers[nearest_index].demand
                distance = self.customer_distance_matrix[index][nearest_index]

                all_time = self.helper_function(nearest_index, time, distance)
                time += all_time

                total_distance += self.customer_distance_matrix[index][nearest_index]
                # add node to path
                path.append(nearest_index)
                # change the current index to nearest node
                index = nearest_index
                # remove visited index
                indexes.remove(nearest_index)
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

    def calculate_nearest_index(self, indexes, index, load, time):
        """
        Calculates the nearest index to the current one
        :param indexes: all the nodes in graph
        :param index: current index
        :param load: load
        :param time: time
        :return: nearest index or node to the current one
        """
        nearest_index = None
        nearest_distance = None

        for i in indexes:
            if load + self.customers[i].demand > self.vehicle_capacity:
                continue

            distance = self.customer_distance_matrix[index][i]
            all_time = self.helper_function(i, time, distance)
            total_time = time + all_time + self.customer_distance_matrix[i][0]
            # Checks if you can return to service store after visiting a passenger or
            # Checks if the vehicle reaches after due time
            if total_time > self.customers[0].due_date or time + distance > self.customers[i].due_date:
                continue

            # comparing nearest node with current next node
            if nearest_distance is None or self.customer_distance_matrix[index][i] < nearest_distance:
                nearest_distance = self.customer_distance_matrix[index][i]
                nearest_index = i

        return nearest_index

    @staticmethod
    def calculate_distance(customer_1, customer_2):
        """
        Calculates distance between two customers or nodes
        :param customer_1: first customer (C_i)
        :param customer_2: second customer (C_j)
        :return: distance between customers or nodes
        """
        return np.linalg.norm((customer_1.x - customer_2.x, customer_1.y - customer_2.y))

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





