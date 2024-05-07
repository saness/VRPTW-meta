"""
file: localsearch.py
description: This program consists of implementation of local search procedure
language: python3
author: Sanish Suwal(ss4657@rit.edu), Jay Nair(an1147@rit.edu), Bhavdeep Khileri(bk2281@rit.edu)
"""
def apply_local_search(ants, customer_distance_matrix):
    """
    Applies local search procedure to all ants after the tour is complete
    :param ants: ants
    :param customer_distance_matrix: distance matrix
    :return: None
    """
    for ant in ants:
        path = ant.path
        routes = get_routes_from_path(path)
        new_routes = []

        for route in routes:
            # apply two opt algorithm
            new_route, improved_distance = two_opt_move(route, customer_distance_matrix)
            new_routes.append(new_route)

        # combine all routes to path
        ant.path = flatten_routes(new_routes)
        ant.total_distance = calculate_path_distance(ant.path, customer_distance_matrix)


def get_routes_from_path(path):
    """
    Separates the path into individual routes, where each route starts and ends at the depot
    :param path: Path
    :return: a list of routes
    """
    routes = []
    current_route = []
    for node in path:
        if node == 0:
            if current_route:
                routes.append(current_route)
                current_route = []
        else:
            current_route.append(node)
    if current_route:
        routes.append(current_route)
    return routes


def flatten_routes(routes):
    """
    combines all the individual routes into a single path by flattening the list of routes,
    :param routes: routes
    :return: Single path
    """
    flattened_path = []
    for route in routes:
        flattened_path.extend(route)
        flattened_path.append(0)
    return flattened_path[:-1]


def calculate_path_distance(path, customer_distance_matrix):
    """
    Calculates the total distance of the given path by summing the distances between consecutive nodes in the path,
    :param path: path
    :param customer_distance_matrix: distance matrix
    :return:
    """
    total_distance = 0
    for i in range(len(path) - 1):
        node1, node2 = path[i], path[i + 1]
        total_distance += customer_distance_matrix[node1][node2]
    return total_distance

def two_opt_move(route, customer_distance_matrix):
    """
    Tries all 2 opt moves in the route and returns the best route and its corresponding distance
    Removes two non-adjacent edges from a route and reconnects the two resulting sub-routes in the opposite order.
    :param route: route
    :param customer_distance_matrix: the distance matrix
    :return: best route and its corresponding distance
    """
    best_route = route.copy()
    best_distance = calculate_route_distance(best_route, customer_distance_matrix)

    for i in range(1, len(route) - 2):
        for j in range(i + 1, len(route)):
            new_route = route.copy()
            # reverse a route in path
            new_route[i:j] = reversed(new_route[i:j])
            # evaluate new route
            new_distance = calculate_route_distance(new_route, customer_distance_matrix)

            # compare current and new route
            if new_distance < best_distance:
                best_route = new_route
                best_distance = new_distance

    return best_route, best_distance

def calculate_route_distance(route, customer_distance_matrix):
    """
    Calculate the total distance of given route,
    :param route: route
    :param customer_distance_matrix: distance matrix
    :return:total distance
    """
    total_distance = 0
    for i in range(len(route) - 1):
        node1, node2 = route[i], route[i + 1]
        total_distance += customer_distance_matrix[node1][node2]
    return total_distance
