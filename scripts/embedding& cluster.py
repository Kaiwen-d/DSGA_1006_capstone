import tqdm
import os
import pandas as pd
import torch
import numpy as np
import datasets
import spacy
from sklearn.cluster import KMeans
import sklearn.metrics as metrics

nlp = spacy.load("en_core_web_sm")
dataset = load_from_disk("/scratch/kd1860/DSGA_1006_capstone/dataset/filtered_dataset")
model = SentenceTransformer('distilbert-base-nli-mean-tokens')


def get_clusters(document):
    # doc = nlp(document["document"]) #use spacy to get the document sentences
    doc = document["document"]
    # bert embedding
    embedding = []
    for sentence in doc:
        embedding.append(model.encode(sentence, show_progress_bar=True))
    document["embedding"] = embedding

    cur = -1
    best_label = np.ndarray(len(embedding))
    cluster = np.arange(2, 6)

    for i in range(2, 6):  # get best num_clusters from silhouette score
        labels = KMeans(n_clusters=i).fit(embedding).labels_  # ,init="k-means++",random_state=200
        score = metrics.silhouette_score(embedding, labels, metric="euclidean")
        if score > cur:
            cur = score
            best_label = labels

    document["labels"] = best_label
    return document

dataset = dataset.add_column("embedding", [[0]] * len(dataset))
dataset = dataset.add_column("labels", [[0]] * len(dataset))

start  = time.time()
print('started')
print("cpu count:", os.cpu_count())
d = dataset.map(get_clusters, num_proc = 48)
end = time.time()
print("map ended")
print('save dataset')

d.save_to_disk("/scratch/kd1860/DSGA_1006_capstone/dataset/clustered_dataset")

print (end-start)