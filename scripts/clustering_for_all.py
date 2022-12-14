import sys
import os
from sklearn.cluster import KMeans
import sklearn.metrics as metrics
import numpy as np
import json
from tqdm import tqdm
import pandas as pd

tqdm.pandas()
dataset_path = "/scratch/kd1860/DSGA_1006_capstone/dataset/moodys/filtered_dataset_with_embedding/shard_"+sys.argv[1]+".jsonl"
save_path =  "/scratch/sl6246/DSGA_1006_capstone/dataset/moodys/cluster_for_each/shard_"+sys.argv[1]+".jsonl"

print("read data set")
df = pd.read_json(path_or_buf=dataset_path, lines=False)

# dataset_path_0 = "/scratch/kd1860/DSGA_1006_capstone/dataset/moodys/filtered_dataset_with_embedding/shard_0.jsonl"
# dataset_path_1 = "/scratch/kd1860/DSGA_1006_capstone/dataset/moodys/filtered_dataset_with_embedding/shard_1.jsonl"
# dataset_path_2 = "/scratch/kd1860/DSGA_1006_capstone/dataset/moodys/filtered_dataset_with_embedding/shard_2.jsonl"
# dataset_path_3 = "/scratch/kd1860/DSGA_1006_capstone/dataset/moodys/filtered_dataset_with_embedding/shard_3.jsonl"
# save_path = "/scratch/sl6246/DSGA_1006_capstone/dataset/moodys/clustered_dataset/shard_0.jsonl"
# print("read data set")
# df0 = pd.read_json(path_or_buf=dataset_path_0, lines= False)
# df1 = pd.read_json(path_or_buf=dataset_path_1, lines= False)
# df2 = pd.read_json(path_or_buf=dataset_path_2, lines= False)
# df3 = pd.read_json(path_or_buf=dataset_path_3, lines= False)
# df = pd.concat([df0, df1, df2, df3])

# print("Input start date and end date")
# start_date = str(input())
# end_date = str(input())
start_date = "2019-09-01"
end_date = "2019-09-10"
df1 = df.loc[df["published_datetime"] >= start_date]
df2 = df1.loc[df["published_datetime"] <= end_date]
print("finish select date")

print("get all embedding")
embedding = []
for i in df2["embedding"]:
    for j in i:
        embedding.append(j)

print("get all sentences")
sent = []
for i in df2["content"]:
    for j in i["filtered_sentences"]:
        sent.append(j)

print("get clusters")
cur = -1
embedding_l = len(embedding)
best_label = np.ndarray(embedding_l).reshape(-1, 1)
cluster = np.arange(10, 50, 5)

for i in cluster:  # get best num_clusters from silhouette score
    labels = KMeans(n_clusters=i).fit(embedding).labels_  # ,init="k-means++",random_state=200
    score = metrics.silhouette_score(embedding, labels, metric="euclidean")
    print("Kmeans for k =", i)
    if score > cur:
        cur = score
        best_label = labels

print("save temp result")
df_new = pd.DataFrame()
df_new["sentences"] = sent
df_new["label"] = best_label
df_new.to_json(path_or_buf=save_path)
print("end")

