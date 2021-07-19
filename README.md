# BiasSum
Data and code for ["Earlier Isn't Always Better: Sub-aspect Analysis on Corpus and System Biases in Summarization
"](https://arxiv.org/abs/1908.11723) by Taehee Jung*, Dongyeop Kang*, Lucas Mentch and Eduard Hovy (*equal contribution), EMNLP 2019. If you have any questions, please contact to Dongyeop Kang (dongyeok@cs.cmu.edu).

We provide a platform ([BiasSum.com](http://http://dongtae.lti.cs.cmu.edu:3232/)) for bias analysis of your system across different summarization corpora. Please evaluate your summarization system across differet domains of datasets and metrics, and measure general performance on robustness against the biases.  

## Citation

    @inproceedings{jungkang19emnlp_biassum,
        title = {Earlier Isn't Always Better: Sub-aspect Analysis on Corpus and System Biases in Summarization},
        author = {Taehee Jung and Dongyeop Kang and Lucas Mentch and Eduard Hovy},
        booktitle = {Conference on Empirical Methods in Natural Language Processing (EMNLP)},
        address = {Hong Kong},
        month = {November},
        url = {https://arxiv.org/abs/1908.11723},
        year = {2019}
    }




### Note
- Some codes are still under development. We will be refactoring them soon. 
- If you like to add a new dataset or a new evaluation metric, please contact to Dongyeop.


### Installation
Please download the pre-processed nine summarization copora in [task](http://dongtae.lti.cs.cmu.edu:3232//task). Every corpora has the same format of dataset as follow:
```
Dataset format: 
[source sentences] \t [target sentences]
or
<s> I was at home .. </s> <s> It was rainy day ..</s> ... \t <s> Sleeping at home rainy day </s> ..
```
An example python script for loading each dataset is provided here
```
python example/data_load.py --dataset AMI
```
### Summarization Corpora
Please check [task] tab for more details in [BiasSum.com/task](http://dongtae.lti.cs.cmu.edu:3232/task)). If you like to download all the preprocessed dataset at once, please download [here](http://dongtae.lti.cs.cmu.edu/data/biassum_v0.1/biassum_data_v0.1.zip).

NOTE: the links are not available now. Please download the pre-processed datasets [here](https://drive.google.com/drive/folders/16aZ1uME_cUdq0t2Wlq-ftbt_ep2RMsmD?usp=sharing).

| Type |  Name | Preprocessed Dataset | Original |
| :---: | :---: | :---: | :---: |
| News | CNNDM | [link](http://dongtae.lti.cs.cmu.edu/data/biassum_v0.1/cnndm.all.zip)  | [link](https://github.com/abisee/cnn-dailymail) |
| News | NewsRoom | [link](http://dongtae.lti.cs.cmu.edu/data/biassum_v0.1/newsroom.all.zip)  | [link](https://summari.es/) |
| News | XSum | [link](http://dongtae.lti.cs.cmu.edu/data/biassum_v0.1/xsum.all.zip)  | [link](https://github.com/EdinburghNLP/XSum) |
| Papers | PeerRead | [link](http://dongtae.lti.cs.cmu.edu/data/biassum_v0.1/peerread.all.zip)  | [link](https://github.com/allenai/PeerRead) |
| Papers | PubMed | [link](http://dongtae.lti.cs.cmu.edu/data/biassum_v0.1/pubmed.all.zip)  | [link](https://github.com/armancohan/long-summarization) |
| Books | BookSum | - | [link](https://www.aclweb.org/anthology/D07-1040.pdf) |
| Dialogues | AMI | [link](http://dongtae.lti.cs.cmu.edu/data/biassum_v0.1/ami.all.zip)  | [link](http://groups.inf.ed.ac.uk/ami/corpus/) |
| Posts | Reddit | [link](http://dongtae.lti.cs.cmu.edu/data/biassum_v0.1/reddit.all.zip)  | [link](http://www.cs.columbia.edu/~ouyangj/aligned-summarization-data/) |
| Script | MovieScript | [link](http://dongtae.lti.cs.cmu.edu/data/biassum_v0.1/moviescript.all.zip)  | [link](https://github.com/EdinburghNLP/scriptbase) |

 
### Evaluation Metrics
 - avearged ROUGE with reference abstractive summaries (R)
 - Sentence overlap score with Oracle extractive summaries (SO)
 - Volume overlap score with reference abstractive summaries (VO)
 - the balance across three aspects (P/D/I)

### Leaderboard
 - Please contact to Dongyeop if you like to add your system to the leaderboard with your R/SO/VO/PDI scores across corpora.
