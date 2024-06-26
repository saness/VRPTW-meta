"""
file: Agent.py
description: This program consists of blueprint for search agent i.e ant or vehicle
language: python3
author: Sanish Suwal(ss4657@rit.edu), Jay Nair(an1147@rit.edu), Bhavdeep Khileri(bk2281@rit.edu)
"""

from graph import Graph
from utils import helper_function

class Agent:
    def __init__(self, graph: Graph, start= 0):
        """
        Constructor of the class
        :param graph: the ant colony graph
        :param start: starting index
        """
        super()
        self.graph = graph
        # current index
        self.index = start
        # vehicle load
        self.load = 0
        # vehicle travel time
        self.time = 0
        # vehicle travel path
        self.path = [start]
        self.arrival_time = [0]
        self.indexes = list(range(graph.customer_numbers))
        # nodes to visit
        self.indexes.remove(start)
        # total travel distance
        self.total_distance = 0



    def travel_to_next(self, next_index):
        """
        Moves the ant to the next node
        :param next_index: the index of next customer or node
        :return: None
        """
        self.path.append(next_index)
        self.total_distance += self.graph.customer_distance_matrix[self.index][next_index]

        distance = self.graph.customer_distance_matrix[self.index][next_index]
        self.arrival_time.append(self.time + distance)

        # If the next location is a server point, the vehicle load, etc. must be cleared.
        if self.graph.customers[next_index].is_depot:
            self.load = 0
            self.time = 0
        # Update vehicle load, travel distance, time
        else:
            self.load += self.graph.customers[next_index].demand
            all_time = helper_function(next_index, distance, self.graph, self.time)
            self.time += all_time
            self.indexes.remove(next_index)

        self.index = next_index

    def check(self, next_index):
        """
        Checks if the constraints of the problem are met or not
        :param next_index: teh index of next node or customer
        :return: true if the move to next customer is feasible else false
        """
        demand = self.graph.customers[next_index].demand
        if self.load + demand > self.graph.vehicle_capacity:
            return False

        distance = self.graph.customer_distance_matrix[self.index][next_index]
        all_time = helper_function(next_index, distance, self.graph, self.time)

        total_time = self.time + all_time + self.graph.customer_distance_matrix[next_index][0]
        due_time = self.graph.customers[0].due_date

        # Checks if you can return to service store after visiting a passenger or
        # Checks if the vehicle reaches after due time
        if total_time > due_time or self.time + distance > due_time:
            return False

        return True

    def is_empty(self):
        """
        Checks if there are nodes to visit or not
        :return: true if no more nodes to visit else false
        """
        return len(self.indexes) == 0


