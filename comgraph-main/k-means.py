# https://www.alanshawn.com/jupyter-nb-show/2019/10/31/spectral-clustering.html


import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.linalg import fractional_matrix_power
from sklearn.cluster import KMeans


with open('graph/karate.paj', 'r') as infile:
    karateLines = list(filter(lambda x: len(x) > 0, infile.read().split('\n')))

edgeInd = karateLines.index('*Edges')
vertexLines = karateLines[1:edgeInd]
edgeLines = karateLines[edgeInd+1:]

print(vertexLines)
print(edgeLines)

# generate adjacency matrix
nVertices = len(vertexLines)
print('num. of vertices:', nVertices)
adMat = np.zeros((nVertices, nVertices), np.float)

for line in edgeLines:
    v1, v2 = list(map(lambda x: int(x) - 1, line.split(' ')))
    adMat[v1, v2] = 1.0
    adMat[v2, v1] = 1.0

print('adjancency matrix:')
print(adMat)
nxGraph = nx.from_numpy_matrix(adMat)
# generate a circular layout
nxPos_x = np.sin(np.linspace(
    0, 2.0 * np.pi, nVertices, endpoint=False)).tolist()
nxPos_y = np.cos(np.linspace(
    0, 2.0 * np.pi, nVertices, endpoint=False)).tolist()
nxPos_tup = zip(nxPos_x, nxPos_y)
nxPos = dict()
for i, loc in enumerate(nxPos_tup):
    nxPos[i] = loc
nx.draw(nxGraph, nxPos)
ax = plt.gca()
ax.set_aspect(1.0)
ax.set_title('Visualization of Karate club')
# plt.show()
# compute the degree matrix
dMat = np.diag(np.sum(adMat, axis=0))
print(dMat)
# compute the lapacian
laplacian_rcut = dMat - adMat
dMat_inv_sqrt = fractional_matrix_power(dMat, -0.5)
laplacian_ncut = np.identity(
    dMat.shape[0]) - dMat_inv_sqrt @ adMat @ dMat_inv_sqrt


def spectral_clustering(laplacian):
    eigVals, eigVecs = np.linalg.eig(laplacian)
    # find the second smallest eigenvalue
    eigValInds = list(zip(range(eigVals.size), eigVals.tolist()))
    eigValInds.sort(key=lambda x: x[1])
    # print(eigValInds)
    eigInd = eigValInds[1][0]
    vec = eigVecs[:, eigInd].reshape((-1, 1))
    # print(vec)
    kmeans = KMeans(n_clusters=2)
    print(vec)
    kmeans.fit(vec)
    assignment = kmeans.predict(vec)
    group1 = np.where(assignment == 0)[0] + 1
    group2 = np.where(assignment == 1)[0] + 1
    print('people in group 1:', group1)
    print('people in group 2:', group2)


spectral_clustering(laplacian_rcut)
