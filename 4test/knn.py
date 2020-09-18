from sklearn.neighbors import NearestNeighbors
import numpy as np

n_neigh = 1

test = [[1,2,3,4,5,6],[1,1,1,1,1,1],[6,5,4,3,2,1],[1,2,3,4,5,6],[6,5,4,3,2,1]]

nbrs = NearestNeighbors(n_neighbors=n_neigh).fit(test)

query = [[1,1,1,1,1,1]]

distances, indices = nbrs.kneighbors(query)

print(distances, indices)

print(test[indices[0][0]])


