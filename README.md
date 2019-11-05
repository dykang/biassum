# BiasSum
Data and code for ["Earlier Isn't Always Better: Sub-aspect Analysis on Corpus and System Biases in Summarization
"](https://arxiv.org/abs/1908.11723) by Taehee Jung*, Dongyeop Kang*, Lucas Mentch and Eduard Hovy (*equal contribution), EMNLP 2019. If you have any questions, please contact to Dongyeop Kang (dongyeok@cs.cmu.edu).

We provide a platform ([BiasSum.com](http://biassum.com)) for bias analysis of your system across different summarization corpora. Please evaluate your summarization system across differet domains of datasets and metrics, and measure general performance on robustness against the biases.  

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
Please download the pre-processed nine summarization copora in [task](http://biassum.com/task). Every corpora has the same format of dataset as follow:
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
(please check [task] tab for more details in [BiasSum.com](http://biassum.com))
 - CNNDM [preprocessed](http://dongtae.lti.cs.cmu.edu/data/biassum_v0.1/cnndm.all.zip) [[[[original]]]](https://github.com/abisee/cnn-dailymail)
 - NewsRoom [[preprocessed]](http://dongtae.lti.cs.cmu.edu/data/biassum_v0.1/newsroom.all.zip) [[original]](https://summari.es/)
 - XSum [[preprocessed]](http://dongtae.lti.cs.cmu.edu/data/biassum_v0.1/xsum.all.zip) [[original]](https://github.com/EdinburghNLP/XSum)
 - PeerRead [[preprocessed]](http://dongtae.lti.cs.cmu.edu/data/biassum_v0.1/peerread.all.zip) [[original]](https://github.com/allenai/PeerRead)
 - PubMed [[preprocessed]](http://dongtae.lti.cs.cmu.edu/data/biassum_v0.1/pubmed.all.zip) [[original]](https://github.com/armancohan/long-summarization)
  - BookSum [[original]](https://www.aclweb.org/anthology/D07-1040.pdf) (needs an additional confirmation from Lada)
 - AMI [[preprocessed]](http://dongtae.lti.cs.cmu.edu/data/biassum_v0.1/ami.all.zip) [[original]](http://groups.inf.ed.ac.uk/ami/corpus/)
 - Reddit [[preprocessed]](http://dongtae.lti.cs.cmu.edu/data/biassum_v0.1/reddit.all.zip) [[original]](http://www.cs.columbia.edu/~ouyangj/aligned-summarization-data/)
 - MovieScript [[preprocessed]](http://dongtae.lti.cs.cmu.edu/data/biassum_v0.1/moviescript.all.zip) [[original]](https://github.com/EdinburghNLP/scriptbase)
 
### Evaluation Metrics
 - avearged ROUGE with reference abstractive summaries (R)
 - Sentence overlap score with Oracle extractive summaries (SO)
 - Volume overlap score with reference abstractive summaries (VO)
 - the balance across three aspects (P/D/I)

### Leaderboard
 - Please contact to Dongyeop if you like to add your system to the leaderboard with your R/SO/VO/PDI scores across corpora.
