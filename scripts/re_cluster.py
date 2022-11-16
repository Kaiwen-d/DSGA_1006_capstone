import tqdm
import os
import pandas as pd
import torch
import numpy as np
import datasets
import torch.nn as nn
import transformers
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
import spacy
import string
from sklearn.cluster import KMeans
import sklearn.metrics as metrics

from sentence_transformers import SentenceTransformer



dataset = load_from_disk("/scratch/kd1860/DSGA_1006_capstone/dataset/multi_news_test_embedded/shard_0")


def re_cluster(document):
    labels = np.array(document["labels"])
    clusters, sents_distribution = np.unique(labels, return_counts=True)
    indices = np.where(sents_distribution >= 40)
    doc = np.array(document["document"])
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
    else:

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


#     return new_labels_0, new_labels_1


dataset = dataset.add_column("reclustered_labels", [[0]] * len(dataset))


start = time.time()
print('started')
print("cpu count:", os.cpu_count())
d = dataset.map(cluster_score, num_proc=48)
end = time.time()
print("map ended")
print('save dataset')

d.save_to_disk("/scratch/sl6246/DSGA_1006_capstone/dataset/reclustered_dataset")

print(end - start)