import os
import pickle
import gzip
import argparse
import nltk
import gc
import sys
import numpy as np
import itertools
import tarfile
import time
import json

from nltk import sent_tokenize
from collections import defaultdict
from collections import Counter
from pathlib import Path
from typing import NamedTuple, List
from itertools import combinations
from tqdm import tqdm

# sys.path.insert(1, os.path.join(sys.path[0], '..'))

def simple_tokenizer(text: str, lower: bool=False, newline: str=None) -> List[str]:
    """Split an already tokenized input `text`."""
    if lower:
        text = text.lower()
    if newline is not None:    # replace newline by a token
        text = text.replace('\n', ' ' + newline + ' ')
    tokens = text.split()
    return tokens


def split_by_sent_tokens_and_remove_them(tokens):
    sents, sent = [],[]
    for t in tokens:
        if t.lower() == '<s>':
            sent = []
        elif t.lower() == '</s>':
            if len(sent)>0:
                sents.append(sent)
        else:
            sent.append(t)
    return sents


class Example(NamedTuple):
    src_sents: List[List[str]]
    tgt_sents: List[List[str]]


class OracleIndexs(NamedTuple):
    rouge: List[float]
    indexs: List[List[int]]


class Dataset(object):
    def __init__(self, dataset='cnndm', emb='bert', dataset_type='t', path='./', ranges=False):
        #parameters
        self.dataset = dataset
        self.dataset_type = dataset_type
        self.path = path
        self.filename = os.path.join(self.path, '%s.%s.gz'%(self.dataset,self.dataset_type))
        self.ranges = ranges
        self.emb = emb

    def load_dataset(self):
        pairs = []
        if self.filename.endswith('.gz'):
            open = gzip.open

        # count the line number only
        with open(self.filename, 'rt', encoding='utf-8') as f:
            for i, line in enumerate(f):
                continue
        line_num = i
        pbar = tqdm(total=line_num)

        print("Reading dataset %s..." % self.filename, end='\n', flush=True)
        with open(self.filename, 'rt', encoding='utf-8') as f:
            for i, line in enumerate(f):
                pair = line.strip().split('\t')
                pbar.update(1)
                src_sents, tgt_sents = None, None
                if not self.ranges or (self.ranges and self.ranges[0]<=i and i<self.ranges[1]):
                    src_sents = split_by_sent_tokens_and_remove_them(pair[0].replace(' <P>','').split(' '))
                    tgt_sents = split_by_sent_tokens_and_remove_them(pair[1].replace(' <P>','').split(' '))

                    if self.dataset == 'booksum':
                        src_sents = src_sents[:1000]
                        tgt_sents = tgt_sents[:50]
                pairs.append(Example(src_sents, tgt_sents)) #None, None,
        print('Number of sentences:%d' % (len(pairs)))
        #NOTE Don't do this ever never!!!!! you can do the second job once you load pairs, right?
        #return pairs, int(np.mean([len(example.tgt_sents) for example in pairs]))
        return pairs #, int(np.mean([len(example.tgt_sents) for example in pairs]))

    def load_volume_data(self,algorithm):
        volume_dict = dict()
        filename = os.path.join(self.path,'ext', self.dataset, self.dataset_type, self.emb, algorithm, 'volume_overwrap.txt')
        with open(filename,'rt') as f:
            for i,line in enumerate(f):
                pair = line.strip().split('\t')
                if len(pair) != 2:
                    continue
                if pair[0] in volume_dict:
                    continue

                volume_dict[pair[0]] = float(pair[1])
        return volume_dict

    def load_decoded_vertice(self,algorithm):
        pairs = [] #element: [idx,[vertices]]
        filename = os.path.join(self.path,'ext', self.dataset, self.dataset_type, self.emb,algorithm, 'selected_vertices.txt')
        txt_algo_vertice = open(filename,'r')
        for line in txt_algo_vertice:
            idx, vertice = line.split('\t')
            #vertice --> string of list to list
            vertice = json.loads(vertice.replace('\n',''))
            pairs.append([int(idx),vertice])
        return pairs

    #Return Dictionary: key = idx, value = list of decoded sentences with word element
    def load_decoded_file(self,algorithm):
        decoded_dict = dict()
        decode_path = []
        if algorithm=='oracle':
            decode_path.append(os.path.join(self.path,'ext',self.dataset,self.dataset_type,'oracle'))
            replace_words =['','','_oracle.txt']
        elif algorithm=='target':
            if os.path.exists(os.path.join(self.path,'ext',self.dataset,self.dataset_type,self.emb,'target2')):
                  decode_path.append(os.path.join(self.path,'ext',self.dataset,self.dataset_type,self.emb,'target2'))

            decode_path.append(os.path.join(self.path,'ext',self.dataset,self.dataset_type,self.emb,algorithm))
            replace_words = ['_tgt.txt','aaa','_target.txt']
        else:
            if os.path.exists(os.path.join(self.path,'ext',self.dataset,self.dataset_type,self.emb,algorithm,'decode')):
                  decode_path.append(os.path.join(self.path,'ext',self.dataset,self.dataset_type,self.emb,algorithm,'decode'))

            if os.path.exists(os.path.join(self.path,'ext',self.dataset,self.dataset_type,self.emb,algorithm,'decoder')):
                  decode_path.append(os.path.join(self.path,'ext',self.dataset,self.dataset_type,self.emb,algorithm,'decoder'))

            decode_path.append(os.path.join(self.path,'ext',self.dataset,self.dataset_type,self.emb,algorithm,'decoded'))
            replace_words = ['_decoded.txt','_decode.txt','_decoder.txt']
        for path in decode_path:
            sorted_dir_decode = sorted(os.listdir(path))
            for f in sorted_dir_decode:
                if 'ntfs' in f:
                    continue
                if f.replace(replace_words[0],"").replace(replace_words[1],"").replace(replace_words[2],"")=='.,':
                    continue
                idx = int(f.replace(replace_words[0],"").replace(replace_words[1],"").replace(replace_words[2],""))
                #import pdb; pdb.set_trace();
                if idx not in decoded_dict:
                    #import pdb; pdb.se();
                    with open(os.path.join(path,f), 'rt', encoding='utf_8') as decoded_file:
                        decoded_indiv = []
                        for i, line in enumerate(decoded_file):
                            decoded_indiv += line.lower().replace("\n","").split(' ')
                    decoded_dict[idx] = decoded_indiv
        #for target2
        #import pdb; pdb.set_trace();
        return decoded_dict


        print("%d oracle pairs." % len(oracle_indexs))
        return oracle_indexs

    def load_oracle_data(self):
        filename = os.path.join(self.path,'bh', '%s.%s.bh.gz' % (self.dataset, self.dataset_type))
        oracle_indexs = []
        if filename.endswith('.gz'):
            open = gzip.open
        #print("Reading bh dataset %s..." % filename, end=' ', flush=True)
        with open(filename, 'rt', encoding='utf-8') as f:
            for i, line in enumerate(f):
                pair = line.strip().split('\t')
                #import pdb; pdb.set_trace();
                #src_sents = sent_tokenize(pair[0])
                if len(pair) != 4:
                    rouge = [0.0]
                    indexs = [[]]
                else:
                    rouge = [float(x) for x in pair[2].split(',')]
                    indexs = json.loads('[' + pair[3] + ']')

                ##Exception for newsroom, test, 35382 originally 35383
                if i == 35382 and self.dataset == 'newsroom' and self.dataset_type=='test':
                    oracle_indexs.append(OracleIndexs([0.0],[[]]))

                oracle_indexs.append(OracleIndexs(rouge,indexs))

                if len(oracle_indexs) == 347582 and self.dataset =='gigaword' and self.dataset_type =='test':
                    oracle_indexs.append(OracleIndexs(rouge,indexs))

        print("%d oracle pairs." % len(oracle_indexs))
        return oracle_indexs

    def save_oracle_text(self):
        oracle_indexs = self.load_oracle_data()
        #import pdb; pdb.set_trace();
        oracle_path = os.path.join(self.path,'ext',self.dataset,self.dataset_type,'oracle')

        if os.path.exists(oracle_path) and len(os.listdir(oracle_path))>0:
            print("Oracle already exists. Number of Files: %d" %(len(os.listdir(oracle_path))))
            return
        #newsroom test --> 35282 --> +1
        elif not os.path.exists(oracle_path):
            os.makedirs(oracle_path)

        pairs = self.load_dataset()
        diff_length = 0
        for idx in range(len(pairs)):
            idx_pad = str(idx+diff_length).zfill(6)
            best_idx = oracle_indexs[idx].indexs[0]
            #if len(oracle_indexs[idx].src_sents)!=len(pairs[idx+diff_length].src_sents):
            #    diff_length +=1
            if len(best_idx)==0:
                continue
            if idx+diff_length < len(pairs):
                oracle_texts = np.array(pairs[idx+diff_length].src_sents)[best_idx].tolist()
                txt_oracle = open(os.path.join(oracle_path, '%s_oracle.txt' %(idx_pad)), 'w')
                txt_oracle.write('\n'.join([' '.join(x) for x in oracle_texts]))
                txt_oracle.close()

        #import pdb; pdb.set_trace()
        print("Save oracle ! %d docs" %(len(os.listdir(oracle_path))))
