import datasets
from datasets import load_dataset
import re
# from nltk.tokenize import sent_tokenize
# import nltk
# import numpy as np
import tqdm
import spacy
import time 
# nltk.download('punkt')
from rouge_score import rouge_scorer
import os
import sys



scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
nlp = spacy.load("en_core_web_sm")
nlp.max_length = 5000000

dataset = load_dataset('multi_news', split='train')

def score_sentence(document):
    text = document['document']
    doc = nlp(text)
    rouge_scores = []
    entity_counts = []
    
    sent_list = list(str(sent).strip() for sent in doc.sents if str(sent).strip() != '')
    
    
    
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
        

        entity_counts.append(len(nlp(target).ents))   
            
    document['rouge_scores'] = rouge_scores
    document['entity_counts'] = entity_counts
    document['document'] = sent_list
    
#     rank = (-np.array(scores)).argsort()
#     threshold = int(len(rank)*0.8)
#     rank = rank[:threshold]
#     rank.sort()
#     filtered_doc = ' '.join([splited_document[i] for i in rank])
#     doc['document'] = filtered_doc
    return document

# dataset = dataset.select(list(range(100)))

dataset = dataset.add_column("rouge_scores", [[0]] * len(dataset))
dataset = dataset.add_column("entity_counts", [[0]] * len(dataset))


if len(sys.argv) >= 2:
    shard = int(sys.argv[1])
    dataset = dataset.shard(num_shards=40, index=shard)


start  = time.time()
print('started')
print("cpu count:", os.cpu_count())
d = dataset.map(score_sentence, num_proc = 8)
end = time.time()
print("map ended")
print('save dataset')



save_dir = "/scratch/kd1860/DSGA_1006_capstone/dataset/processed_shards/shard_"+str(shard)
print(save_dir)
d.save_to_disk(save_dir)

print (end-start)