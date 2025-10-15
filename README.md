# Tweet Sentiment Transformers
Sentiment analysis on TweetEval using RoBERTa and DistilBERT

This repository contains code for exploring how large pretrained Transformer models can be adapted to
specific downstream tasks. 

The goal is to compare the performance of two distinct models with different fine-tuning strategies:
full fine-tuning and parameter-efficient training (LoRA) of transformer-based language models on the
TweetEval Sentiment dataset (CardiffNLP, 2020).

The goal is to compare the performance and efficiency of RoBERTa and DistilBERT on the TweetEval Sentiment dataset (CardiffNLP, 2020) under two strategies:
- Full fine-tuning (updating all model parameters)
- LoRA fine-tuning (Low-Rank Adaptation)
---

## Repository Structure

```
tweet-sentiment-transformers/
│
├── main.py                 # runs all experiments (both models + both strategies)
├── utils.py                # helper functions: data loading, tokenization, training, metrics
├── requirements.txt        # dependencies
├── .gitignore
├── README.md               # this file
└── results/                # outputs (auto-generated)
```
---
