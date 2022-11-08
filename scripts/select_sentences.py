import datasets
from datasets import load_from_disk
import numpy as np
import spacy
import time 
import os

nlp = spacy.load("en_core_web_sm")
dataset = load_from_disk("/scratch/kd1860/DSGA_1006_capstone/dataset/scored_dataset_test500")

def filter_sentence(document):
  percentage = 0.8 #percentage to keep
  top = 5 # keep first t sentences 
  sentences = list(nlp(document['document']).sents)
  pointers = [i for i in range(len(sentences)) if '|||||' in sentences[i].text] #seperate articles
  scores = document['rouge_scores']
  filter = []
  score_splits = [sl.tolist()for sl in np.split(scores, pointers)]
  for a in range(len(score_splits)):
    rest = score_splits[a][5:]
    n_leave = int(len(rest)*percentage) #leave 0.8 percent of the rest of the sentences (except top sentences)
    index_leave = sorted(range(len(rest)), key = lambda sub: rest[sub])[-n_leave:] #index of top n_leave scores
    if len(score_splits[a])>=top:
      filter.append([1]*top+[0]*len(rest))
    else:
      filter.append([1]*(len(score_splits[a])))
    for i in index_leave:
      filter[a][i+top] = 1
  filter = sum(filter,[])
  document['filter_sentences'] = filter
  return document

dataset = dataset.add_column("filter_sentences", [[0]] * len(dataset))

start  = time.time()
print('started')
print("cpu count:", os.cpu_count())
d = dataset.map(filter_sentence, num_proc = 48)
end = time.time()
print("map ended")
print('save dataset')

d.save_to_disk("/scratch/kd1860/DSGA_1006_capstone/dataset/filtered_dataset") #实际上只是加了一个column，要存一个新的dataset吗

print (end-start)