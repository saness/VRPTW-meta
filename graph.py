import numpy as np
from Customer import *

class Graph:
    def __init__(self, file_path, rho = 0.1):
        super()

        self.customer_numbers, self.customers, self.customer_distance_matrix, self.vehicle_number, self.vehicle_capacity \
            = self.create_customer(file_path)



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
        for i in range(customer_numbers):
            customer_1 = customers[i]
            customer_distance_matrix[i][i] = 1e-8
            for j in range(i + 1, customer_numbers):
                customer_2 = customers[j]
                customer_distance_matrix[i][j] = Graph.calculate_distance(customer_1, customer_2)
                customer_distance_matrix[j][i] = customer_distance_matrix[i][j]

        return customer_numbers, customers, customer_distance_matrix, vehicle_number, vehicle_capacity


    def calculate_distance(self, customer_1, customer_2):
        return np.linalg.norm((customer_1.x - customer_2.x, customer_1.y - customer_2.y))







