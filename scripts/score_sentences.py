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



scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
nlp = spacy.load("en_core_web_sm")

dataset = load_dataset('multi_news', split='train')

def score_sentence(document):
    text = document['document']
    doc = nlp(text)
    rouge_scores = []
    entity_counts = []
    
    
    for s in doc.sents:
        target = text[s.start_char:s.end_char]
        rest_doc = text[:s.start_char] + text[s.end_char:]
#         print(target)
        try:
            score = scorer.score(target,
                      rest_doc)['rouge1'][2]
            rouge_scores.append(score)
        except:
            rouge_scores.append(0)
        

        entity_counts.append(len(nlp(str(s)).ents))   
            
    document['rouge_scores'] = rouge_scores
    document['entity_counts'] = entity_counts
    
#     rank = (-np.array(scores)).argsort()
#     threshold = int(len(rank)*0.8)
#     rank = rank[:threshold]
#     rank.sort()
#     filtered_doc = ' '.join([splited_document[i] for i in rank])
#     doc['document'] = filtered_doc
    return document

dataset = dataset.add_column("rouge_scores", [[0]] * len(dataset))
dataset = dataset.add_column("entity_counts", [[0]] * len(dataset))




start  = time.time()
print('started')
print("cpu count:", os.cpu_count())
d = dataset.map(score_sentence, num_proc = 48)
end = time.time()
print("map ended")
print('save dataset')


d.save_to_disk("/scratch/kd1860/DSGA_1006_capstone/dataset/scored_dataset")

print (end-start)