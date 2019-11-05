import os
import sys
import time
import gensim
import gc
import random
import numpy as np
import pickle as pkl
import pandas as pd

from math import ceil
from copy import deepcopy
from collections import defaultdict
from shapely.geometry import Polygon as poly
from shapely.geometry import Point as point
from shapely.geometry import LineString as line
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from collections import Counter

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from encoder import Encoder
from data import Example

########### Position ###########
def mid(n,k):
    mid = np.median(range(n))
    half_k = k/2*1.0
    start_point = mid - half_k
    return range(ceil(start_point),ceil(start_point)+k)

########### Importance ###########
def nearest(vecs, k):
    ##need to transpose!!!
    df = pd.DataFrame(data=vecs).transpose()

    #pearson correlation matrix between all sentences
    corr_matrix = df.corr().values
    corr_sum = np.sum(corr_matrix,axis=0)
    #sort indexs by largest to smallest correlation (importance)
    src_corr_matrix = []
    for i, corr in enumerate(corr_sum):
        src_corr_matrix.append((i,corr))

    src_corr_matrix = sorted(src_corr_matrix, key= lambda x:x[1], reverse=True)
    #return first k indexs
    return sorted([x[0] for x in src_corr_matrix[:k]])

def nearest_np(vecs, k,graph=False):
    corr = np.corrcoef(vecs)
    corr_sum = np.sum(corr, axis=0)
    src_corr_matrix = []

    #In case one
    if type(corr_sum)==np.float64:
        return [0]
    for i, corr in enumerate(corr_sum):
        src_corr_matrix.append((i,corr))

    src_corr_matrix = sorted(src_corr_matrix, key = lambda x:x[1], reverse=True)
    if graph:
        return [x[0] for x in src_corr_matrix[:k]]
    else:
        return sorted([x[0] for x in src_corr_matrix[:k]])


def knn(src_example, n_neighbour, k, graph = False, graph_oracle=[]):
    n = deepcopy(n_neighbour) + 1
    src = deepcopy(src_example)
    vertices = []
    indexs = [x for x in range(len(src_example))]

    if k > len(src):
        return indexs

    for iter in range(k):
        #KNN distance sum
        n = min(n,len(src))
        if n == 0:
            vertices.append(indexs[0])
            continue
        nbrs = NearestNeighbors(n_neighbors=n, algorithm='ball_tree').fit(src)
        distances,_ = nbrs.kneighbors(src)
        dist_sums = np.sum(distances, axis=1)
        #Find a max distance idx --> append

        max_dist_idx = np.argmin(dist_sums)
        vertices.append(indexs[max_dist_idx])
        #Pop all
        src.pop(max_dist_idx)
        indexs.pop(max_dist_idx)

        if graph:
            if set(graph_oracle) <= set(vertices):
                break

    if graph:
        return vertices
    else:
        return sorted(vertices)


########### Diversity ###########
def farthest(pca_src_example,k,graph=False):
    #get centroid
    centroid = np.mean(pca_src_example,axis=0)
    #calculate distance
    src_distance_matrix = []
    for i, pca_src_vec in enumerate(pca_src_example):
        src_distance_matrix.append((i, get_distance(pca_src_vec,centroid)))
    #sort idx by distance (farthest to nearest to centroid, descending order)
    src_distance_matrix = sorted(src_distance_matrix, key= lambda x:x[1], reverse = True)
    #return first k indexs
    if graph:
        return [x[0] for x in src_distance_matrix]
    return sorted([x[0] for x in src_distance_matrix[:k]])

def hard_waterfall(convexhull, pca_src_example, tgt_length, r):
    ch_vertice = convexhull.vertices
    sentence_reduced_wf = [[x, y] for x, y in zip(ch_vertice, pca_src_example[ch_vertice])]
    curr_vertice = ch_vertice.tolist()

    if len(curr_vertice) <= tgt_length:
        return sorted(curr_vertice)

    if tgt_length >= len(pca_src_example):
        return sorted(ch_vertice)

    if tgt_length == 1:
            # Find a center
            centroid = np.mean(pca_src_example, axis=0)
            distance_list = []
            for pca_src_vec in pca_src_example[ch_vertice]:
                distance_list.append(get_distance(pca_src_vec, centroid))

            # Take the nearest one
            return [ch_vertice[np.argmax(distance_list)]]


    for iter in range(len(ch_vertice) - (r+1)):
        src_vol_wf = defaultdict()
        for idx_wf, sentence in enumerate(sentence_reduced_wf):
            # calculate the volume w/o idx_wf
            curr_sentences = sentence_reduced_wf[:idx_wf] + sentence_reduced_wf[idx_wf + 1:]
            src_vol_wf[idx_wf] = PolyVol(np.array([y for x, y in curr_sentences]), r)

        idx_mx = max(src_vol_wf, key=src_vol_wf.get)

        if len(curr_vertice) <= tgt_length:
            return sorted(curr_vertice)

        sentence_reduced_wf = sentence_reduced_wf[:idx_mx] + sentence_reduced_wf[idx_mx + 1:]
        curr_vertice = [x for (x, y) in sentence_reduced_wf]

        if len(curr_vertice) == tgt_length:
            break

    if tgt_length == 2:
        centroid = np.mean(pca_src_example[curr_vertice], axis=0)
        distance_list = []
        for pca_src_vec in pca_src_example[curr_vertice]:
            distance_list.append(get_distance(pca_src_vec, centroid))

        # Take the nearest one
        return sorted(curr_vertice[:np.argmin(distance_list)] + curr_vertice[np.argmin(distance_list) + 1:])

    return sorted(curr_vertice)

def hard_heuristic(convexhull, pca_src_example, tgt_length, graph=False):
    # Hard heuristic (k = tgt_size)
    hs, indexs2 = [], []
    hh_rouge = 0
    #ch_vertice = convexhull.vertices
    hh_rouge_find = False

    if not graph:
       # if len(ch_vertice) <= tgt_length:
       #     return sorted(ch_vertice)

        if tgt_length >= len(pca_src_example):
            return sorted(ch_vertice)

    for sentence in range(tgt_length):
        item_idx, item = HardHeuristic(pca_src_example, indexs2, hs)
        indexs2.append(item_idx)
        hs.append(item)

    if graph:
        return indexs2
    else:
        return sorted(indexs2[:tgt_length])


