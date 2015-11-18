from collections import Counter

import parser
from pandas import *
from numpy import array, isfinite
import operator
from sklearn.cluster import MeanShift, estimate_bandwidth

import numpy as np
import pylab as pl
from itertools import cycle

def plot_meanshift(data, cluster_centers, n_centers_):
    colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
    pl.figure(1)
    pl.clf()

    for k, col in zip(cluster_centers['cluster'], colors):
        cluster_points = data[(data['cluster'] == k)]

        cluster_longitudes = cluster_points['longitude']
        cluster_latitudes = cluster_points['latitude']

        cluster_center = cluster_centers[(cluster_centers['cluster'] == k)]

        pl.plot(cluster_longitudes, cluster_latitudes, col + '.')
        pl.plot(cluster_center['longitude'], cluster_center['latitude'], 'o', markerfacecolor=col,
            markeredgecolor='k', markersize=14)

    pl.title('Result of mean_shift (%d estimated clusters)' % n_centers_)
    pl.show()
def Mean_Shift(path):
    #importer les donnees
    data=pandas.read_csv(filepath_or_buffer=path,delimiter=',',encoding='utf-8')  
    data.drop_duplicates()
    print (data)
    #lire les donnees
    values=data[['latitude', 'longitude']].values
    print("printing values")
    print (values)
    #Mean shift
    print ("Clustering data Meanshift algorithm")
    bandwidth = estimate_bandwidth(values, quantile=0.003, n_samples=None)
    #ms = MeanShift(bandwidth=bandwidth, bin_seeding=True, min_bin_freq=20, cluster_all=False)
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True,min_bin_freq=25,cluster_all=False)
    ms.fit(values)
    data['cluster'] = ms.labels_
    data = data.sort(columns='cluster')
    data = data[(data['cluster'] != -1)]
    print (data['cluster'])
    data['cluster'] = data['cluster'].apply(lambda x:"cluster" +str(x))
    labels_unique = np.unique(ms.labels_).tolist()
    del labels_unique[0]
    # Filtering clusters centers according to data filter
    cluster_centers = DataFrame(ms.cluster_centers_, columns=['latitude', 'longitude'])
    cluster_centers['cluster'] = labels_unique
    print (cluster_centers)
    n_centers_ = len(cluster_centers)
    print("number of clusters is :%d" % n_centers_)
    # print ("Exporting clusters to {}...'.format(clusters_file)")
    data.to_csv(path_or_buf="output/points.csv", cols=['user','latitude','longitude','cluster','picture','datetaken'], encoding='utf-8')
    #print ("Exporting clusters centers to {}...'.format(centers_file)")
    cluster_centers['cluster'] = cluster_centers['cluster'].apply(lambda x:"cluster" +str(x))
    cluster_centers.to_csv(path_or_buf="output/centers.csv", cols=['latitude', 'longitude','cluster'], encoding='utf-8')
    plot_meanshift(data, cluster_centers, n_centers_)
    return 0
Mean_Shift("input/data.csv")