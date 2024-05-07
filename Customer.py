"""
file: Customer.py
description: This program is blueprint for a customer node.
language: python3
author: Sanish Suwal(ss4657@rit.edu), Jay Nair(an1147@rit.edu), Bhavdeep Khileri(bk2281@rit.edu)
"""

class Customer:
    def __init__(self, id, x_coordinate, y_coordinate, demand, ready_time, due_date, service_time):
        """
        Constructor for Customer class
        :param id: customer id
        :param x: x coordinate
        :param y: y coordinate
        :param demand: demand of customer
        :param ready_time: ready time
        :param due_date: due date of delivery
        :param service_time: service time
        """
        super()
        self.id = id
        self.x_coordinate = x_coordinate
        self.y_coordinate = y_coordinate
        self.demand = demand
        self.ready_time = ready_time
        self.due_date = due_date
        self.service_time = service_time

        # if id is 0 then it is a depot
        if self.id == 0:
            self.is_depot = True
        else:
            self.is_depot = False