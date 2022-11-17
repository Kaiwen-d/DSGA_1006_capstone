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

from summarizer import Summarizer
bert = Summarizer('distilbert-base-uncased')
sbert = SBertSummarizer('paraphrase-MiniLM-L6-v2')


device = "cuda" if torch.cuda.is_available() else "cpu"

# model = PegasusForConditionalGeneration.from_pretrained(model_name).to(device)
print(device)


def get_summary(document):
    # >5 sentence 2-3
    # <=5 sentences select 1
    # labels = np.array(document["reclustered_labels"])

    #     clusters, sents_distribution = np.unique(labels,return_counts = True)
    summary = []
    for i in document["document"]:

        length = len(i)

        if length >= 5:
            cluster_string = "".join(i)

            sub_summary = bert(cluster_string, num_sentences=3).to(device)
            summary.append([sub_summary])

        else:
            cluster_string = "".join(i)
            sub_summary = bert(cluster_string, num_sentences=1).to(device)
            summary.append([sub_summary])
    document["pred"] = summary
    return document

shard = load_from_disk("/scratch/kd1860/DSGA_1006_capstone/dataset/multi_news_test_clustered/shard_" + sys.argv[1])

print("start")
print("device:", device)
result = shard.map(get_summary)
print("end")


result.save_to_disk("/scratch/kd1860/DSGA_1006_capstone/dataset/multi_news_test_bert/shard_"+ sys.argv[1])