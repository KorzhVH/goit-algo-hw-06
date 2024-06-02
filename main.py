import networkx as nx
import matplotlib.pyplot as plt
from collections import deque
from graph import G


#task_1
def graph_research(graph):
    neighbors_of_nodes = {}
    for node in graph.nodes():
        neighbors_of_nodes[node] = list(graph.neighbors(node))
    print(f"Кількість обласних центрів України: {graph.nodes}\n"
          f"Кількість доріг між ними: {len(graph.edges())}\n"
          f"Сусіди областей:\n{neighbors_of_nodes}")

# graph_research(graph.G)
# nx.draw(graph.G)
# plt.show()

#task_2
def dfs_alg(graph, start, visited=None, path=None):
    result = []
    if visited is None:
        visited = set()
    if path is None:
        path = []

    visited.add(start)
    path.append(start)

    paths = []

    for neighbor in graph[start]:
        if neighbor not in visited:
            new_path = dfs_alg(graph, neighbor, visited, path)
            for p in new_path:
                paths.append(p)

    if not paths:
        paths.append(path)

    for i, path in enumerate(paths):
        for j, node in enumerate(path[:-1]):
            result.append((node, path[j + 1]))

    return result


def bfs_alg(graph,start):
    visited = set()
    result = []
    queue = deque([start])
    while queue:
        vertex = queue.popleft()
        if vertex not in visited:
            result.append(vertex)
            visited.add(vertex)
            queue.extend(set(graph[vertex]) - visited)
    return result

# print(funcs.dfs_alg(graph.G, "Київська"))
# print(funcs.bfs_alg(graph.G, "Київська"))

#task_3
adjacency_dict_with_weights = nx.to_dict_of_dicts(G)
graph_with_weights = {}
for vertex, neighbors in adjacency_dict_with_weights.items():
    weighted_neighbors = {neighbor: G[vertex][neighbor]['weight'] for neighbor in neighbors}
    graph_with_weights[vertex] = weighted_neighbors


def dijkstra(graph, start):
    # Ініціалізація відстаней та множини невідвіданих вершин
    distances = {vertex: float('infinity') for vertex in graph}
    distances[start] = 0
    unvisited = list(graph.keys())
    previous_vertices = {vertex: None for vertex in graph}

    while unvisited:
        # Знаходження вершини з найменшою відстанню серед невідвіданих
        current_vertex = min(unvisited, key=lambda vertex: distances[vertex])

        # Якщо поточна відстань є нескінченністю, то ми завершили роботу
        if distances[current_vertex] == float('infinity'):
            break

        for neighbor, weight in graph[current_vertex].items():
            distance = distances[current_vertex] + weight

            # Якщо нова відстань коротша, то оновлюємо найкоротший шлях
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                previous_vertices[neighbor] = current_vertex

        # Видаляємо поточну вершину з множини невідвіданих
        unvisited.remove(current_vertex)

    return distances, previous_vertices


def shortest_path(graph, start, target):
    distances, previous_vertices = dijkstra(graph, start)
    path = []
    current_vertex = target

    while current_vertex is not None:
        path.append(current_vertex)
        current_vertex = previous_vertices[current_vertex]

    path = path[::-1]
    if path[0] == start:
        return path
    else:
        return None
print(shortest_path(graph_with_weights, "Київська", "Кримська"))