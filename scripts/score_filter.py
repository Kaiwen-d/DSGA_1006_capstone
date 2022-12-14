import json
from tqdm import tqdm
import numpy as np
import spacy
import time 
import pandas as pd 
# nltk.download('punkt')
from rouge_score import rouge_scorer
import os
import sys

scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
tqdm.pandas()

def score_sentence(document):
    sent_list = document['content']['sentences']
    rouge_scores = []
    
    for idx in range(len(sent_list)):
        target = sent_list[idx]
        rest_doc = ' '.join(sent_list[:idx] + sent_list[idx+1:])
#         print(target)
        try:
            score = scorer.score(target,
                      rest_doc)['rouge1'][2]
            rouge_scores.append(score)
        except:
            rouge_scores.append(0)
        
  
            
    document['rouge_scores'] = rouge_scores
    
    
    return document

def filter_sentence(document):
    sentences = document['content']['sentences']
    scores = document['rouge_scores']
    percentage = 0.8 #percentage to keep
    top = 5 # keep first t sentences 
    filter = []
    
    rest = sentences[top:]
    n_leave = int(len(rest)*percentage) #leave 0.8 percent of the rest of the sentences (except top sentences)
    index_leave = sorted(range(len(rest)), key = lambda sub: rest[sub])[-n_leave:] #index of top n_leave scores
    if len(sentences)>=top:
        filter.append([1]*top+[0]*len(rest))
    else:
        filter.append([1]*(len(sentences)))
    for i in index_leave:
        filter[0][i+top] = 1
    #print(filter)
    filter = list(map(bool,sum(filter,[]))) #merge lists into one list

    
    filtered_sentences = np.array(sentences)[filter]
    filtered_scores = np.array(np.array(scores))[filter]
    document['content']['filtered_sentences'] = filtered_sentences.tolist()
    document['filtered_rouge_scores'] = filtered_scores.tolist()
    return document

dataset_path = "/scratch/kd1860/DSGA_1006_capstone/dataset/moodys/shard_"+sys.argv[1]+".jsonl"
save_path = "/scratch/kd1860/DSGA_1006_capstone/dataset/moodys/filtered_dataset/shard_"+sys.argv[1]+".jsonl"
print("read data set")
df = pd.read_json(path_or_buf=dataset_path, lines=True)
print("start scoring")
df_score = df.progress_apply(score_sentence, axis = 1)
print("start filtering")
df_filtered = df_score.progress_apply(filter_sentence, axis = 1)

print("save temp result")
df_filtered.to_json(path_or_buf=save_path)
print("end")



