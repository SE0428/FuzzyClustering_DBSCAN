import os
import pandas as pd
import numpy as np
from scipy.io import loadmat
from Dataload import fileload

from math import pow

import matplotlib.pyplot as plt

os.chdir('./Data')
EM = loadmat('./EM_Points.mat')

#fileload
EM_df=fileload(EM) #x,y,label

"""
# Fuzzy Clustering using EM
1. get the initial center of cluster  #find center 
2. calculate SSE (Sum of Square Eroors) 
3. update the ceter based on SSE 
4. till it converge
5. reassign the cluster number and compare it by drawing 
6. need to discuss in the reports briefly 
"""

#check uniqe value to find the number of cluster
#print(EM_df.label.unique()) #0,1


# E-step
def E_step(EM_df, c0, c1):
    m0 = []
    m1 = []

    # weight for cluster 0 %ppt에서는 1
    for i in range(len(EM_df)):
        temp_x = EM_df.iloc[i][0]
        temp_y = EM_df.iloc[i][1]

        dist_c0_data = pow(temp_x - c0[0], 2) + pow(temp_y - c0[1], 2)
        dist_c1_data = pow(temp_x - c1[0], 2) + pow(temp_y - c1[1], 2)

        weight = dist_c1_data / (dist_c0_data + dist_c1_data)

        m0.append(round(weight, 2))
        m1.append(1 - round(weight, 2))

    M = np.array(list(zip(m0, m1)))

    return M


# M_step + get new center

def M_step(M):
    M_T = M.T

    k = np.square(M_T)

    # cluster 1's re point
    new_x = []
    new_y = []

    for j in range(len(M_T)):
        target_x = 0
        target_y = 0
        # cluster j's re point

        for i in range(len(EM_df)):
            target_x = target_x + k[j][i] * EM_df.x[i]
            target_y = target_y + k[j][i] * EM_df.y[i]

        new_x.append(target_x / np.sum(k[j]))
        new_y.append(target_y / np.sum(k[j]))

    c0 = []
    c1 = []
    c0 = [new_x[0], new_y[0]]  # initial cluser is 1
    c1 = [new_x[1], new_y[1]]  # initial cluser is 0

    return c0, c1


# SSE

def SSE(c0, c1, data, weight):
    score = 0
    # c_x and c_y is the center point

    num_cluster = 2
    num_data = len(data)  # 400

    for j in range(num_cluster):
        if j == 0:
            c = c0
        else:
            c = c1

        for i in range(num_data):
            # print(weight[i][j],c_x,data.x[i],c_y,data.y[i])
            # print( weight[i][j]*(pow(c_x-data.x[i],2)+pow(c_y-data.y[i],2)))
            score = score + weight[i][j] * (pow(c[j] - data.x[i], 2) + pow(c[j] - data.y[i], 2))

    return score


def draw(c0, c1, EM_df):
    # find the center

    color_map = {0: 'red', 1: 'yellow'}

    EM_df['color'] = EM_df.label.map(color_map)

    plt.scatter(x='x', y='y', c='label', data=EM_df)
    plt.xlabel("x")
    plt.xlabel("y")

    label_c0 = str(c0)
    label_c1 = str(c1)

    plt.scatter(c1[0], c1[1], color='green', label=label_c1)  # group 1's center
    plt.scatter(c0[0], c0[1], color='blue', label=label_c0)  # group 0's center

    EM_df.drop(['color'], axis=1)

    # plt.savefig('kmeans_data.png')


def assign_cluster(c0, c1, EM_df):
    from scipy.spatial import distance

    new_label = []

    for i in range(len(EM_df)):

        z = (EM_df.x[i], EM_df.y[i])

        dist_c0 = distance.euclidean(z, c0)
        dist_c1 = distance.euclidean(z, c1)

        if dist_c0 > dist_c1:
            new_label.append(1)

        else:
            new_label.append(0)

    return new_label


def EM(EM_df):
    # Initialized c1 and c0
    c1 = EM_df.iloc[200]  # initial cluser is 1
    c0 = EM_df.iloc[199]  # initial cluser is 0

    c0_list = []
    c1_list = []
    SSE_list = []

    for i in range(10):
        M = E_step(EM_df, c0, c1)
        c0, c1 = M_step(M)

        c0_list.append(c0)
        c1_list.append(c1)

        # print(c0, c1)

        score = SSE(c0, c1, EM_df, M)

        SSE_list.append(score)

        # print(i, "-th SSE", score)

        draw(c0, c1, EM_df)

        # retrun max SSE's center
        if SSE_list[i - 1] > score:
            c0 = c0_list[i - 1]
            c1 = c1_list[i - 1]

            break

    # assign new label
    new_label = assign_cluster(c0, c1, EM_df)

    EM_df['new_label'] = new_label

    # make report

    report = pd.DataFrame(list(zip(c0_list, c1_list, SSE_list)), columns=['c0', 'c1', 'SSE'])

    # print(report)

    print(c0, c1)

    return c0, c1, report


#proceed EM / get the last c0,c1
c0,c1,report= EM(EM_df)

print(report)

#find if cluster is changed
EM_df['same?'] = EM_df['label'] != EM_df['new_label']
sub=EM_df.loc[np.where(EM_df['same?']==True)]

#draw only the changed point
draw(c0,c1,sub)

#draw all data point
draw(c0,c1,EM_df)







