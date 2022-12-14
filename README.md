# DSGA_1006_capstone
The automatic generation of summaries from multiple news articles is a valuable
tool as the number of online publications grows rapidly. Given a group of
daily news articles on the same topic, we apply preprocessing steps tailored to multi-
document summarization problems, and then summarize these documents into
several sentences using multi-document summarization model, in order to make
information retrieval more efficient. 

This Github includes implementations for the preprocessing steps: filtering and clustering, and summarization with PEGASUS and BERT summarization model. 
These steps are applied on two datasets: Multi-News oublic dataset and Moody's news dataset.

## Step by step python scripts for multi_news dataset:
- Get rouge score for each sentence: score_sentences_shard.py
- Filter sentences that are irrelavent: select_sentences.py
- Get embeddings for each sentence: embedding.py
- Cluster sentence: clulster.py
- Summarization with Pegasus model: pegasus.py

## Code for Moody's news dataset:
- score and filter sentences: score_filter.py
- Sentence embedding: embedding_moodys.py
