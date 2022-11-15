import datasets
from datasets import load_from_disk
import numpy as np
import spacy
import time 
import os

nlp = spacy.load("en_core_web_sm")
dataset = load_from_disk("/scratch/kd1860/DSGA_1006_capstone/dataset/scored_dataset_test500")

def filter_sentence(document):
  percentage = 0.7 #percentage to keep
  top = 3 # keep first t sentences 
  sentences = document['document']
  pointers = [i for i in range(len(sentences)) if '|||||' in sentences[i]] #seperate articles
  scores = document['rouge_scores']
  filter = []
  score_splits = [sl.tolist()for sl in np.split(scores, pointers)]
  for a in range(len(score_splits)):
    rest = score_splits[a][top:]
    n_leave = int(len(rest)*percentage) #leave 0.8 percent of the rest of the sentences (except top sentences)
    index_leave = sorted(range(len(rest)), key = lambda sub: rest[sub])[-n_leave:] #index of top n_leave scores
    if len(score_splits[a])>=top:
      filter.append([1]*top+[0]*len(rest))
    else:
      filter.append([1]*(len(score_splits[a])))
    for i in index_leave:
      filter[a][i+top] = 1
  filter = list(map(bool,sum(filter,[])))
  filtered_sentetnces = np.array(sentences)[filter]
  filtered_scores = np.array(np.array(scores))[filter]
  document['document'] = filtered_sentetnces
  document['rouge_scores'] = filtered_scores
  return document


start  = time.time()
print('started')
print("cpu count:", os.cpu_count())
filtered_dataset = dataset.map(filter_sentence, num_proc = 48)
end = time.time()
print("map ended")
print('save dataset')

filtered_dataset.save_to_disk("/scratch/kd1860/DSGA_1006_capstone/dataset/filtered_dataset") 

print (end-start)