import itertools

import numpy as np
from tqdm.auto import tqdm


def preprocess_constraints(constrains_matrix, n):
    visited = np.full(n, False)

    def dfs(start):
        stack, components = [start], []
        while stack:
            vertex = stack.pop()
            if visited[vertex]:
                continue
            components.append(vertex)
            visited[vertex] = True
            for neighbor in np.where(constrains_matrix[vertex] == 1)[0]:
                stack.append(neighbor)
        return components

    neighborhoods = []
    for i in tqdm(np.arange(n), total=n):
        if not visited[i]:
            components = dfs(i)
            constrains_matrix[zip(*itertools.permutations(components, 2))] = 1
            neighborhoods.append(components)

    for (i, j) in tqdm(np.where(constrains_matrix == -1)):
        for x in np.where(constrains_matrix[i] == 1):
            constrains_matrix[((x, j), (j, x))] = -1
        for y in np.where(constrains_matrix[j] == 1):
            constrains_matrix[((i, y), (y, i))] = -1
        for x in np.where(constrains_matrix[i] == 1):
            for y in np.where(constrains_matrix[j] == 1):
                constrains_matrix[((x, y), (y, x))] = -1

    return constrains_matrix, neighborhoods
