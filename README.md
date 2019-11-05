# SuGE
Summarization for General Evaluation (SuGE) Benchmarks.
Summarization for Unified General Evaluation (SuGE) Benchmarks.
Evaluate your summarization system across datasets and metrics, and measure general performance toward bias-free, robust summarization system. 

### Leaderboard
Leaderboard.
### Installation
Please run './setup.sh' to download all preprocessed datasets across nine different summarization tasks. Each task has splitted into train, val, and test by following the original paper if possible, otherwise random splits by 0.9, 0.05. and 0.05. The data format is same across all datasets:
```
Dataset format:
<s> I was at home .. </s> <s> It was rainy day ..</s> ... \t <s> Sleeping at home rainy day </s> ..
```
An example python script for loading each dataset is provided here
```
python example/data_load.py --dataset AMI
```
### Summarization Datasets
 - CNNDM
 - NewsRoom
 - DUC
 - GigaWord
 - PeerRead
 - PubMed
 - XSum
 - BookSum (needs additional confirmation from Lada)
 - AMI
 - Reddit
 - MovieScript
### Evaluation Metrics
 - ROUGE with reference abstractive summaries
 - Pyramid with reference abstractive summaries
 - Sentence overlap score with Oracle extractive summaries
 - Volume overlap score with reference abstractive summaries
 
 For each metric, we measure following averaged measures for evaluating general performance of your system:
 - SuGE score: averaged score across dataset
 - Bias score + bias triangle: averaged score across dataset
