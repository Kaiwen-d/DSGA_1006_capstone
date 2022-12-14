import sys
import os
import spacy
from sklearn.cluster import KMeans
import sklearn.metrics as metrics
import os
import numpy as np
import json
from tqdm import tqdm

import pandas as pd

tqdm.pandas()

# def select_date(document):
# start_date = str(input())
# end_date = str(input())
# df1 = df.loc[df["published_datetime"] >= start_date]
# df2 = df1.loc[df["published_datetime"] <= end_date]
# df2

def get_cluster(document):
    embedding = document["embedding"]
    cur = -1
    embedding_l = len(embedding)
    best_label = np.ndarray(embedding_l).reshape(-1, 1)

    if embedding_l >= 6:
        cluster = np.arange(2, 6)
    else:
        cluster = np.arange(2, embedding_l)

    for i in cluster:  # get best num_clusters from silhouette score
        labels = KMeans(n_clusters=i).fit(embedding).labels_  # ,init="k-means++",random_state=200
        score = metrics.silhouette_score(embedding, labels, metric="euclidean")
        if score > cur:
            cur = score
            best_label = labels

    document["labels"] = best_label
    return document


def re_cluster(document):
    labels = np.array(document["labels"])
    clusters, sents_distribution = np.unique(labels, return_counts=True)
    indices = np.where(sents_distribution >= 100)
    embedding = np.array(document["embedding"])

    if len(indices[0]) == 1:
        # case when only one group needs adjustment
        if indices[0][0] == 0:
            sents_ind = np.where(labels == 0)
            new_embedding = embedding[[sents_ind[0]]]
            new_labels = KMeans(n_clusters=2).fit(new_embedding[0]).labels_
            new_labels[new_labels == 1] = 2
            labels[[sents_ind[0]]] = new_labels

        else:
            sents_ind = np.where(labels == 1)
            new_embedding = embedding[[sents_ind[0]]]
            new_labels = KMeans(n_clusters=2).fit(new_embedding[0]).labels_
            new_labels[new_labels == 0] = 2
            labels[[sents_ind[0]]] = new_labels
    # case when two group needs adjustment
    elif len(indices[0]) == 2:
        # process 0 group, the original 0 is still 0, 1 to 3
        sents_ind_0 = np.where(labels == 0)
        new_embedding_0 = embedding[[sents_ind_0[0]]]
        new_labels_0 = KMeans(n_clusters=2).fit(new_embedding_0[0]).labels_
        new_labels_0[new_labels_0 == 1] = 2
        # process 1 group, the original 1 is still 1, 0 to 4
        sents_ind_1 = np.where(labels == 1)
        new_embedding_1 = embedding[[sents_ind_1[0]]]
        new_labels_1 = KMeans(n_clusters=2).fit(new_embedding_1[0]).labels_
        new_labels_1[new_labels_1 == 0] = 3

        labels[[sents_ind_0[0]]] = new_labels_0
        labels[[sents_ind_1[0]]] = new_labels_1

    document["reclustered_labels"] = labels
    return document

dataset_path = "/scratch/kd1860/DSGA_1006_capstone/dataset/moodys/filtered_dataset_with_embedding/shard_"+sys.argv[1]+".jsonl"
save_path = "/scratch/sl6246/DSGA_1006_capstone/dataset/moodys/clustered_dataset/shard_"+sys.argv[1]+".jsonl"
print("read data set")
df = pd.read_json(path_or_buf=dataset_path, lines= False)
# print("change date format")
# df["published_datetime"] = pd.to_datetime(df["published_datetime"])
print("start clustering")
df_cluster = df.progress_apply(get_cluster, axis = 1)
print("start reclustering")
df_recluster = df_cluster.progress_apply(re_cluster, axis = 1)

print("save temp result")
df_recluster.to_json(path_or_buf=save_path)
print("end")

