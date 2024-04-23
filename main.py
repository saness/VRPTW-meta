from graph import Graph
from Solve import ACO


if __name__ == '__main__':
    file_path = './solomon-100/r205.txt'
    ants = 10
    maximum_iteration = 200
    beta = 2
    q0 = 0.1
    show_figure = False

    graph = Graph(file_path)
    basic_aco = ACO(graph, ants=ants, maximum_iteration=maximum_iteration, beta=beta, q0=q0)

    basic_aco.run_basic_aco()
