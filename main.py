import numpy as np

from graph import Graph
from Solve import ACO


if __name__ == '__main__':
    file_path = './solomon-100/r105.txt'
    ants = 20
    maximum_iteration = 10000
    beta = 2
    q0 = 0.1
    num_trails = 30
    distance_list = np.zeros(num_trails)
    vehicle_list = np.zeros(num_trails)

    for i in range(num_trails):
        print(f'---------------------Trail {i +1} ---------------------------')
        graph = Graph(file_path)
        algo = ACO(graph, ants=ants, maximum_iteration=maximum_iteration, beta=beta, q0=q0)
        best_distance, best_vehicle_number = algo.ant_colony_optimization()
        distance_list[i] = best_distance
        vehicle_list[i] = best_vehicle_number
        del algo, graph

    mean_distance = distance_list.mean()
    std_distance = distance_list.std()
    mean_vehicles = vehicle_list.mean()
    std_vehicles = vehicle_list.std()

    print("Average best path distance is:{:.7f}".format(mean_distance))
    print("Standard deviation of best path distance is:{:.7f}".format(std_distance))
    print("Average best vehicle number is:{:.7f}".format(mean_vehicles))
    print("Standard deviation of best vehicle numbers is:{:.7f}".format(std_vehicles))





