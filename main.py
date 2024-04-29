from graph import Graph
from Solve import ACO


if __name__ == '__main__':
    file_path = './solomon-100/c201.txt'
    ants = 20
    maximum_iteration = 10000
    beta = 2
    q0 = 0.1
    show_figure = False

    graph = Graph(file_path)
    algo = ACO(graph, ants=ants, maximum_iteration=maximum_iteration, beta=beta, q0=q0)
    algo.ant_colony_optimization()
