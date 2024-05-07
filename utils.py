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