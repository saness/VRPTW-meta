import numpy as np
from Customer import *

class Graph:
    def __init__(self, file_path, rho = 0.1):
        super()
        self.rho = rho
        self.customer_numbers, self.customers, self.customer_distance_matrix, self.vehicle_number, self.vehicle_capacity \
            = self.create_customer(file_path)
        self.nearest_travel_path, self.pheromone_value, _ = self.nearest_heuristic()
        self.pheromone_value = (1/self.pheromone_value * self.customer_numbers)

        self.pheromone_matrix = np.ones((self.customer_numbers, self.customers)) * self.pheromone_value
        self.information_matrix = 1/ self.customer_distance_matrix



    def create_customer(self, file_path):
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
        for customer in customer_list:
            customers.append(Customer(int(customer[0]), float(customer[1]), float(customer[2]), float(customer[3]),
                                      float(customer[4]), float(customer[5]), float(customer[6])))

        customer_distance_matrix = np.zeros((customer_numbers, customer_numbers))
        customer_distance_matrix = Graph.calculate_distance_matrix(customer_numbers, customer_distance_matrix, customers)

        return customer_numbers, customers, customer_distance_matrix, vehicle_number, vehicle_capacity

    def calculate_distance_matrix(self,customer_numbers, customer_distance_matrix, customers):
        for i in range(customer_numbers):
            customer_1 = customers[i]
            customer_distance_matrix[i][i] = 1e-8
            for j in range(i + 1, customer_numbers):
                customer_2 = customers[j]
                customer_distance_matrix[i][j] = Graph.calculate_distance(customer_1, customer_2)
                customer_distance_matrix[j][i] = customer_distance_matrix[i][j]

        return customer_distance_matrix

    def nearest_heuristic(self,maximum_vehicle_number = None):
        indexes = []
        index = 0
        load = 0
        time = 0
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
                waiting_time = max(self.customers[nearest_index].ready_time - time - distance, 0)
                service_time = self.customers[nearest_index].service_time
                time += distance + waiting_time + service_time

                total_distance += self.customer_distance_matrix[index][nearest_index]
                path.append(nearest_index)
                index = nearest_index
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

        return path, total_distance, vehicle_number



    def calculate_nearest_index(self, indexes, index, load, time):
        nearest_index = None
        nearest_distance = None

        for i in indexes:
            if load + self.customers[i].demand > self.vehicle_capacity:
                continue

            distance = self.customer_distance_matrix[index][i]
            waiting_time = max(self.customers[i].ready_time - time-distance, 0)
            service_time = self.customers[i].service_time
            total_time = time + distance + waiting_time + service_time + self.customer_distance_matrix[i][0]

            if total_time > self.customers[0].due_date or time + distance > self.customers[i].due_date:
                continue

            if nearest_distance is None or self.customer_distance_matrix[index][i] < nearest_distance:
                nearest_distance = self.customer_distance_matrix[index][i]
                nearest_index = i

        return nearest_index

    @staticmethod
    def calculate_distance(self, customer_1, customer_2):
        return np.linalg.norm((customer_1.x - customer_2.x, customer_1.y - customer_2.y))

    def update_local_pheromone(self, i, j):
        self.pheromone_matrix[i][j] = (1-self.rho) * self.pheromone_matrix[i][j] + self.rho * self.pheromone_value

    def update_global_pheromone(self, path, distance):
        pheromone_increase = self.rho/distance
        self.pheromone_matrix *= (1-self.rho)

        index = path[0]
        for i in path[1:]:
            self.pheromone_matrix[index][i] += pheromone_increase
            index = i





