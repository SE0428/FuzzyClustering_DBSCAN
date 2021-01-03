import os
from scipy.io import loadmat
import pandas as pd

from Dataload import fileload
import numpy as np
from math import pow

import matplotlib.pyplot as plt

os.chdir('./Data')
DBSCAN = loadmat('./DBSCAN.mat')
DBSCAN_df=fileload(DBSCAN)

"""

#DB SCAN description 

1. Draw a picture for your cluster results and outliers in each parameter setting in your report. 
For clearly, in each picture, the color of outliers should be BLUE.
2. Add a table to report how many clusters and outliers you find in each parameter setting in your report.

3. Discuss the results of different parameter settings, and report the best setting that you think and write your reason clearly.
4. Note that you are NOT allowed to use any existing DBSCAN library. You need to submit your code.


Parameter Setting
1. Set ε = 5, Minpoints=5.
2. Set ε = 5, Minpoints=10
3. Set ε = 10, Minpoints=5.
4. Set ε = 10, Minpoints=10.

"""


def dbSCAN(Data, eps, minPts):
    Data = list(zip(DBSCAN_df.x, DBSCAN_df.y))
    # initialized all datapoint
    # if label is 0 ,it has been considered
    labels = [0] * len(Data)

    # C is the number of current cluster.
    C = 0

    for P in range(len(Data)):

        if not (labels[P] == 0):
            continue  # the point hasn't been considered

        # get  P's neighboring points.
        Neighbors = RangeQuery(Data, P, eps)

        if len(Neighbors) < minPts:  # Density check
            labels[P] = -1  # Label as Noise == -1


        else:  # if it is not outlier, need to assign cluster
            C += 1
            growCluster(Data, labels, P, Neighbors, C, eps, minPts)

    return labels


def RangeQuery(Data, P, eps):
    neighbors = []

    for Pn in range(len(Data)):

        dist = np.sqrt(pow(Data[P][0] - Data[Pn][0], 2) + pow(Data[P][1] - Data[Pn][1], 2))

        if dist < eps:
            neighbors.append(Pn)

    # print(neighbors)

    return neighbors


def growCluster(Data, labels, P, NeighborPts, C, eps, minPts):
    # Assign the cluster label to the seed point.
    labels[P] = C

    i = 0

    while i < len(NeighborPts):

        # Get the next point from the queue.
        Pn = NeighborPts[i]

        # Change Noise to border point
        if labels[Pn] == -1:
            labels[Pn] = C

        # Otherwise, if Pn isn't already claimed, claim it as part of C.
        elif labels[Pn] == 0:
            # Add Pn to cluster C (Assign cluster label C).
            labels[Pn] = C

            # Find all the neighbors of Pn #Density check
            PnNeighborPts = RangeQuery(Data, Pn, eps)

            if len(PnNeighborPts) >= minPts:
                NeighborPts = NeighborPts + PnNeighborPts  # Add new neighbors to seed set

        i += 1

        # We've finished growing cluster C!


# draw a results

# draw a results

def results(Data, labels):
    Data['labels'] = labels

    cluster = len(np.unique(labels)) - 1
    outlier = labels.count(-1)

    #plt cluster results
    #print("Number of cluster", cluster)  # -1 is outlie
    #print("Number of outliers", outlier)  # -1 is outlier

    #color_map = {-1: 'blue', 0: 'red', 1: 'yellow', 2: "green", 3: "pink", 4: "brown", 5: "orange"}

    #Data['color'] = Data.labels.map(color_map)

    #plt.scatter(x='x', y='y', c='color', data=DBSCAN_df)
    #plt.xlabel("x")
    #plt.ylabel("y")

    return cluster, outlier


## Main##

candidate = [5, 10]

num_id = []
num_cluster = []
num_outlier = []

for i in range(len(candidate)):

    eps = candidate[i]

    for j in range(len(candidate)):
        minPts = candidate[j]

        # print("dbSCAN works eps = ",eps,"minPts =",minPts)
        labels = dbSCAN(DBSCAN_df, eps=eps, minPts=minPts)

        cluster, outlier = results(DBSCAN_df, labels)

        n_id = "(" + str(eps) + "," + str(minPts) + ")"
        num_id.append(n_id)
        num_cluster.append(cluster)
        num_outlier.append(outlier)


result=pd.DataFrame(list(zip(num_id,num_cluster,num_outlier)),columns =['(eps,minPts)', 'num_cluster','num_outlier'])

print(result)





