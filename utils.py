"""
file: utils.py
description: This program consists of helper functions
language: python3
author: Sanish Suwal(ss4657@rit.edu), Jay Nair(an1147@rit.edu), Bhavdeep Khileri(bk2281@rit.edu)
"""
def helper_function(next_index, distance,graph, time):
    """
    Calculates the total time of distance, waiting and service
    :param next_index: index where ant is
    :param distance: distance between nodes
    :return: total time elapsed
    """
    service_time = graph.customers[next_index].service_time
    waiting_time = max(graph.customers[next_index].ready_time - time - distance, 0)
    all_time = distance + waiting_time + service_time
    return all_time

def helper_function_two(index, time, distance, customers):
    """
    Calculates the total time of distance, waiting and service
    :param index: index where ant is
    :param time: time elapsed
    :param distance: distance between nodes
    :return: total time elapsed
    """
    waiting_time = max(customers[index].ready_time - time - distance, 0)
    service_time = customers[index].service_time
    all_time = distance + waiting_time + service_time

    return all_time