import os
import pandas as pd
from scipy.io import loadmat

def fileload(data):

    # for DBSCAN.mat
    #DBSCAN = loadmat('./DBSCAN.mat')
    num_data = len(data['Points'])
    num_feature = len(data['Points'][0])

    if num_feature ==2:
        columns = ['x', 'y' ]#DB, with no label
    else:
        columns=['x','y','label'] #EM

    #check the form of the data

    #print(DBSCAN.keys()) #Points
    #print(DBSCAN['Points']) #elements of points
    #print(len(DBSCAN['Points']))
    #print(len(DBSCAN['Points'][0]))

    #create dataframe
    df = pd.DataFrame(columns=columns)

    for j in range(num_data):
        to_append = []

        for i in range(num_feature):
            to_append.append(data['Points'][j][i])

        df.loc[j] = to_append


    #print(df)
    return df
