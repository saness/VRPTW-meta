"""
file: dataextraxt.py
description: This program extracts the data from dataset and creates customer nodes and vehicle details
language: python3
author: Sanish Suwal(ss4657@rit.edu), Jay Nair(an1147@rit.edu), Bhavdeep Khileri(bk2281@rit.edu)
"""
import numpy as np

from Customer import *

def create_customer(file_path):
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

    create_customer_nodes(customers, customer_list)

    customer_distance_matrix = np.zeros((customer_numbers, customer_numbers))
    customer_distance_matrix = calculate_distance_matrix(customer_numbers, customer_distance_matrix, customers)

    return customer_numbers, customers, customer_distance_matrix, vehicle_number, vehicle_capacity
def calculate_distance(customer_1, customer_2):
    """
    Calculates distance between two customers or nodes
    :param customer_1: first customer (C_i)
    :param customer_2: second customer (C_j)
    :return: distance between customers or nodes
    """
    return np.linalg.norm((customer_1.x_coordinate - customer_2.x_coordinate, customer_1.y_coordinate - customer_2.y_coordinate))

def create_customer_nodes(customers, customer_list):
    """
    Adds customer details  to customer list as Customer class
    :param customers: list of customers as Customer node
    :param customer_list: list of extracted customer details from dataset
    :return:
    """
    for customer in customer_list:
        customers.append(Customer(int(customer[0]), float(customer[1]), float(customer[2]), float(customer[3]),
                                  float(customer[4]), float(customer[5]), float(customer[6])))

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
            customer_distance_matrix[i][j] = calculate_distance(customer_1, customer_2)
            customer_distance_matrix[j][i] = customer_distance_matrix[i][j]

    return customer_distance_matrix