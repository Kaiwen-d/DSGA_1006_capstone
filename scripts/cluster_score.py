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

def cluster_score(document):
    clusters = max(document["labels"])

    scores = [[] for i in range(clusters + 1)]
    for ind, d in enumerate(document["labels"]):
        scores[d].append(document["rouge_scores"][ind])

    document["clustered_scores"] = scores
    return document


dataset = dataset.add_column("clustered_scores", [[0]] * len(dataset))


start = time.time()
print('started')
print("cpu count:", os.cpu_count())
d = dataset.map(cluster_score, num_proc=48)
end = time.time()
print("map ended")
print('save dataset')

d.save_to_disk("/scratch/sl6246/DSGA_1006_capstone/dataset/clustered_scored_dataset")

print(end - start)