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
from datasets import load_from_disk
import time
import sys


# model = SentenceTransformer('distilbert-base-nli-mean-tokens')

def get_clusters(document):
#     doc = nlp(document["document"]) #use spacy to get the document sentences

    #bert embedding
    embedding = embedding = np.array(document["embedding"])

    
    
    cur = -1
    best_label = np.ndarray(len(embedding), dtype = "int32")
    cluster = np.arange(2,6)
    
    for i in range(2,min(6,len(document['document']))): # get best num_clusters from silhouette score
        labels = KMeans(n_clusters=i).fit(embedding).labels_ #,init="k-means++",random_state=200
        score = metrics.silhouette_score(embedding,labels,metric="euclidean")
        if score > cur:
            cur = score
            best_label = labels

    document["labels"] = best_label 
    # text = np.array(document['document'])
    # clustered_text = []
    # for c in np.unique(document['labels']):
    #     clustered_text.append(list(text[document['labels'] == c]))
        
    # document['document'] = clustered_text
    return document

def cluster(document):
    text = np.array(document['document'])
    clustered_text = []
    for c in np.unique(document['labels']):
        clustered_text.append(list(text[document['labels'] == c]))
        
    document['document'] = clustered_text
    return document

def re_cluster(document):
    labels = np.array(document["labels"])
    clusters, sents_distribution = np.unique(labels, return_counts=True)
    indices = np.where(sents_distribution >= 50)
    embedding = np.array(document["embedding"])
    scores = np.array(document["rouge_scores"])

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

    document["reclustered_labels"] = labels.tolist()

    text = np.array(document['document'])
    clustered_text = []
    clustered_score = []
    for c in np.unique(document['reclustered_labels']):
        clustered_text.append(list(text[document['reclustered_labels'] == c]))
        clustered_score.append(list(scores[document['reclustered_labels'] == c]))
    document['document'] = clustered_text
    document['clustered_scores'] = clustered_score
    return document



def cluster_score(document):
    clusters = max(document["reclustered_labels"])

    scores = [[] for i in range(clusters + 1)]
    for ind, d in enumerate(document["reclustered_labels"]):
        scores[d].append(document["rouge_scores"][ind])

    document["clustered_scores"] = scores
    return document




shard = load_from_disk("/scratch/kd1860/DSGA_1006_capstone/dataset/multi_news_test_embedded/shard_" + sys.argv[1])

print("start")

result = shard.map(get_clusters, num_proc= 8)
print("get cluster finished")
re_clustered = result.map(re_cluster,num_proc= 8)
print("reclustered_finished")
# re_clustered_score = re_clustered.map(cluster_score,num_proc=8)
# print("score cluster finished")

re_clustered.save_to_disk("/scratch/kd1860/DSGA_1006_capstone/dataset/multi_news_test_clustered/shard_" + sys.argv[1])
print("saved")