import json
from tqdm import tqdm
import numpy as np
import time 
import pandas as pd 
# nltk.download('punkt')
from rouge_score import rouge_scorer
import os
import sys
from sentence_transformers import SentenceTransformer


model = SentenceTransformer('distilbert-base-nli-mean-tokens')

tqdm.pandas()

def embedding(document):
    sent_list = document['content']['filtered_sentences']

    #bert embedding
    embedding = []
    for sentence in sent_list:
        embedding.append(model.encode(sentence, show_progress_bar=False))
    document["embedding"] = embedding

    return document

dataset_path = "/scratch/kd1860/DSGA_1006_capstone/dataset/moodys/filtered_dataset/shard_"+sys.argv[1]+".jsonl"
save_path = "/scratch/kd1860/DSGA_1006_capstone/dataset/moodys/filtered_dataset_with_embedding/shard_"+sys.argv[1]+".jsonl"
print("read data set")
df = pd.read_json(path_or_buf=dataset_path, lines=False)
print("start embedding")
df_emb = df.progress_apply(embedding, axis = 1)
print("save temp result")
df_emb.to_json(path_or_buf=save_path)
print("end")
