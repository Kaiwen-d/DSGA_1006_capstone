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
#load model
model = SentenceTransformer('distilbert-base-nli-mean-tokens')


def get_embedding(document):
    # doc = nlp(document["document"]) #use spacy to get the document sentences
    doc = document["document"]
    # bert embedding
    embedding = []
    for sentence in doc:
        embedding.append(model.encode(sentence, show_progress_bar=True))
    document["embedding"] = embedding

    return document

dataset = dataset.add_column("embedding", [[0]] * len(dataset))

start  = time.time()
print('started')
print("cpu count:", os.cpu_count())
d = dataset.map(get_clusters, num_proc = 48)
end = time.time()
print("map ended")
print('save dataset')

d.save_to_disk("/scratch/kd1860/DSGA_1006_capstone/dataset/embedded_dataset")

print (end-start)