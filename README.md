# BiasSum
Bias analysis of your system across different summarization corpora. Evaluate your summarization system across datasets and metrics, and measure general performance toward bias-free, robust summarization system. 


### Installation
Please download the pre-processed nine summarization copora in [task](http://biassum.com/task).
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
 - DUC (will be added)
 - GigaWord (will be added)
 - PeerRead
 - PubMed
 - XSum
 - BookSum (needs additional confirmation from Lada)
 - AMI
 - Reddit
 - MovieScript
 
### Evaluation Metrics
 - ROUGE with reference abstractive summaries
 - Sentence overlap score with Oracle extractive summaries
 - Volume overlap score with reference abstractive summaries
 - SO for each aspect (P/D/I)

### Leaderboard
Please contact to Dongyeop Kang (dongyeok@cs.cmu.edu) if you like to add your system to the leaderboard with your R/SO/VO/PDI scores across corpora.
