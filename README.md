# BiasSum
Bias analysis of your system across different summarization corpora. Evaluate your summarization system across differet domains of datasets and metrics, and measure general performance on robustness against the biases. 


### Installation
Please download the pre-processed nine summarization copora in [task](http://biassum.com/task).
```
Dataset format: [source sentences] \t [target sentences]
<s> I was at home .. </s> <s> It was rainy day ..</s> ... \t <s> Sleeping at home rainy day </s> ..
```
An example python script for loading each dataset is provided here
```
python example/data_load.py --dataset AMI
```
### Summarization Datasets
 - CNNDM
 - NewsRoom
 - PeerRead
 - PubMed
 - XSum
 - BookSum (needs additional confirmation from Lada)
 - AMI
 - Reddit
 - MovieScript
 
### Evaluation Metrics
 - avearged ROUGE with reference abstractive summaries (R)
 - Sentence overlap score with Oracle extractive summaries (SO)
 - Volume overlap score with reference abstractive summaries (VO)
 - the balance across three aspects (P/D/I)

### Leaderboard
 - Please contact to Dongyeop Kang (dongyeok@cs.cmu.edu) if you like to add your system to the leaderboard with your R/SO/VO/PDI scores across corpora.
 - If you like to request for adding a new dataset or a new evaluation metric, please contact to Dongyeop.
