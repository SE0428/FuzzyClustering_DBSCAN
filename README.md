# FuzzyClustering_DBSCAN

# Task1 : Fuzzy Clustering using EM 

### Data Description
The dataset contains 400 2D points totally with 2 clusters. Each point is in the format of [X- coordinate, Y-coordinate, label].

### Implementation
 Manually implement Fuzzy Clustering using the EM algorithm

Initial Setting

* The centers of each cluster referred c0 and c1 to label 0 and label1 respectively.  
The initial settings are  c0=( -0.428721, 1.555075), label = 0 and c1=( 0.749460,0.343558 0) , label = 1

Fuzzy clustering with EM algorithm 

* Fuzzy clustering with EM algorithm implemented in two steps in the program which are E_step and M_step. 
* E_step returns the weights for each data point, while M_step returns a new center of cluster based on weights, which is the result of E_step. 
* When two steps are completed, the updated center is available so that the Sum of Square Error(SSE) can be calculated. 
* The last step is finding the new centers that maximize the SSE. Table1 is the results of two steps iteratively until SSE converges.

### result
Reassigning the clusters to all data points is proceeded, which is decided by comparing the distances to both c0 and c1.

![alternativetext](/Fuzzy Results.png)



