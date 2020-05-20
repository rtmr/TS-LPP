#----------------------------------------------------------------------
# Copyright Ryo Tamura(NIMS), Momo Matsuda(University of Tsukuba), 
# and Yasunori Futamura(University of Tsukuba)
# This program is free software: you can redistribute it and/or modify 
# it under the terms of the MIT licence.
#----------------------------------------------------------------------


import numpy as np
import scipy
import linecache
from sklearn import decomposition
from sklearn.neighbors import KNeighborsClassifier
import random
import scipy.stats

from scipy.spatial import distance

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn import decomposition
from sklearn.neighbors import KNeighborsClassifier
import sklearn.metrics
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score



########################
###### parameters ######
########################

# Reduced dimension in the final step
n_components=3

# Number for neighbor graph in LPP
knn=7

# Number of target clusters
n_clusters=3

########################




###################################
########## Training data ##########
###################################


# Standarization
def new_centering(A, B):
    stdA = np.std(A, 0)
    indexA = np.where(stdA !=0)
    return (B[:,indexA[0]] - np.mean(A[:,indexA[0]],0))/stdA[indexA[0]]

# Traing data
input_file='Training'

num_lines=sum(1 for line in open(input_file+'.csv'))

Xdata=[]

for i in range(num_lines):
    A=linecache.getline(input_file+'.csv',i+1).rstrip('\r\n').split(',')
    length=len(A)

    comp=[]
    for j in range(length):
        comp.append(float(A[j]))

    Xdata.append(comp)

Xdata=np.array(Xdata)


X=new_centering(Xdata,Xdata)

# Number of features
n_features=len(comp)


print('*************')
print('training data')
print('number of data =',len(X))
print('number of features =',n_features)
print('number of clusters =',n_clusters)
print('reduced dimension =',n_components)
print('*************')

#####################################




###########################
########## TSLPP ##########
###########################

# Determinaltion of the grid search
sig_num=20
dim_num=4

sig_list=[]
for i in range(sig_num):
    sig_list.append(5.0*i+5.0)


dim_list=[]
for i in range(dim_num):
    dim_list.append(int(n_features/(dim_num+1)*(i+1)))


PseudoF_max=0.0

print('')
print('Searching optimized hyperparameters...')

# Loop of hyperparameter dm
for dim_index in range(dim_num):

    n_components_middle=dim_list[dim_index]

    # Loop of hyperparameter sigma
    for sig_index in range(sig_num+1):

        if sig_index==sig_num:
            sigma=sigma_max
            n_components_middle=dim_max
        else:
            sigma=sig_list[sig_index]

        ############
        # First LPP

        X1=X

        # Create distance matrix
        dist=distance.cdist(X1,X1,metric='euclidean')

        W=np.exp(-np.power(dist,2)/2.0/sigma/sigma)

        for i in range(len(X1)):
            W[i,i]=0.0


        # Create neighbor graph
        for i in range(len(X1)):

            del_list=np.argsort(W[i])[::-1][knn:]

            W[i,del_list]=0.0


        # Symmetrical of W
        W=np.maximum(W.T,W)

        # Create D
        D=np.diag(np.sum(W, axis=1))

        # Create L
        L=D-W


        # SVD of X1
        delta = 1e-7
        U, Sig, VT = np.linalg.svd(X1, full_matrices=False)
        rk = np.sum(Sig/Sig[0] > delta)
        Sig = np.diag(Sig)
        U1 = U[:,0:rk]
        VT1 = VT[0:rk,:]
        Sig1 = Sig[0:rk,0:rk]
        
        # Positive definite for L
        Lp=np.dot(U1.T,np.dot(L,U1))
        Lp=(Lp+Lp.T)/2

        # Positive definite for D
        Dp=np.dot(U1.T,np.dot(D,U1))
        Dp=(Dp+Dp.T)/2


        # Generalized eigenvalue problem
        eig_val,eig_vec=scipy.linalg.eigh(Lp,Dp)


        # Projection for low dimension
        tmp1 = np.dot(VT1.T, scipy.linalg.solve(Sig1, eig_vec))
        Trans_eig_vec=tmp1.T
        Pro1=Trans_eig_vec[0:n_components_middle]

        X_middle=np.dot(Pro1,X.T).T



        ######
        # Second LPP

        X2=X_middle

        # Create distance matrix
        dist=distance.cdist(X2,X2,metric='euclidean')

        W=np.exp(-np.power(dist,2)/2.0/sigma/sigma)

        for i in range(len(X2)):
            W[i,i]=0.0


        # Create neighbor graph
        for i in range(len(X2)):

            del_list=np.argsort(W[i])[::-1][knn:]

            W[i,del_list]=0.0


        # Symmetrical of W
        W=np.maximum(W.T,W)

        # Create D
        D=np.diag(np.sum(W, axis=1))

        # Create L
        L=D-W

        # SVD of X2
        U, Sig, VT = np.linalg.svd(X2, full_matrices=False)
        rk = np.sum(Sig/Sig[0] > delta)
        Sig = np.diag(Sig)
        U2 = U[:,0:rk]
        VT2 = VT[0:rk,:]
        Sig2 = Sig[0:rk,0:rk]

        # Positive definite for L
        Lp=np.dot(U2.T,np.dot(L,U2))
        Lp=(Lp+Lp.T)/2

        # Positive definite for D
        Dp=np.dot(U2.T,np.dot(D,U2))
        Dp=(Dp+Dp.T)/2


        # Generalized eigenvalue problem
        eig_val,eig_vec=scipy.linalg.eigh(Lp,Dp)


        # Projection for low dimension
        tmp2 = np.dot(VT2.T, scipy.linalg.solve(Sig2, eig_vec))
        Trans_eig_vec=tmp2.T
        Pro2=Trans_eig_vec[0:n_components]


        X_final=np.dot(Pro2,X2.T).T

        #Clustering by K means
        kmeans_model = KMeans(n_clusters=n_clusters, random_state=10).fit(X_final)

        labels=kmeans_model.labels_

        PseudoF=sklearn.metrics.calinski_harabaz_score(X_final,labels)


        if PseudoF > PseudoF_max:
            PseudoF_max=PseudoF
            sigma_max=sigma
            dim_max=n_components_middle

print('')
print('*************')
print('hyperparameter optimization')
print('optimized sigma =',sigma_max)
print('optimizad dm =',dim_max)
print('*************')


##########
# Result


# Output of training result by Kmeans
f1=open('result_training_Kmeans.txt', 'w')

result_str_list1 = []
for i in range(len(X_final)):

    V_list = list(X_final[i])
    V_str_list = ["{0: >30.15f}".format(v) for v in V_list]

    result_str_list1.append(str(int(labels[i])) + "".join(V_str_list) + '\n')

f1.writelines(result_str_list1)

f1.close()

# Plot of training result by true label
# For 2
if n_components==2:

    x_coordinate=[[] for j in range(n_clusters)]
    y_coordinate=[[] for j in range(n_clusters)]

    for i in range(len(X_final)):
    
        x_coordinate[int(labels[i])].append(X_final[i,0])
        y_coordinate[int(labels[i])].append(X_final[i,1])

    
    fig = plt.figure(figsize=(8, 8))

    for i in range(n_clusters):
        plt.plot(x_coordinate[i],y_coordinate[i],'.',alpha=0.7)

    plt.savefig('2D_Kmeans.png', dpi=150)    


# For 3
if n_components==3:

    x_coordinate=[[] for j in range(n_clusters)]
    y_coordinate=[[] for j in range(n_clusters)]
    z_coordinate=[[] for j in range(n_clusters)]

    for i in range(len(X_final)):
    
        x_coordinate[int(labels[i])].append(X_final[i,0])
        y_coordinate[int(labels[i])].append(X_final[i,1])
        z_coordinate[int(labels[i])].append(X_final[i,2])

    
    fig = plt.figure(figsize=(8, 8))
    ax = Axes3D(fig)

    for i in range(n_clusters):
        ax.plot(x_coordinate[i],y_coordinate[i],z_coordinate[i],'.',alpha=0.7)

    ax.view_init(30,60)
    plt.savefig('3D_Kmeans.png', dpi=150)    


# Calculate cluster index
PseudoF=sklearn.metrics.calinski_harabaz_score(X_final,labels)

print('')
print('*************')
print('clustering results')
print('Pseudo F =',PseudoF)
print('*************')

###########################


