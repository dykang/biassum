import os
import gc
import json
import gzip
import argparse
import gensim
import torch
import sys
import numpy as np
from collections import OrderedDict
from collections import defaultdict
from sklearn import preprocessing
from tqdm import tqdm

from torch.utils.data import TensorDataset
from torch.utils.data import SequentialSampler
from torch.utils.data import DataLoader
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertModel

sys.path.insert(1, os.path.join(sys.path[0], '..'))

class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self, unique_id, sent_id, tokens, input_ids, input_mask):
        self.unique_id = unique_id
        self.sent_id = sent_id
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask

def parse_one_document(ex_index,sid, sent, seq_length, tokenizer):

    tokens_a = []
    #tokenizing for each word in sents --> use for loop!
    for word in sent:
        tokens_a += tokenizer.tokenize(word)

    # Account for [CLS] and [SEP] with "- 2"
    if len(tokens_a) > seq_length - 2:
        tokens_a = tokens_a[0:(seq_length - 2)]

    tokens = []
    tokens.append("[CLS]")
    for token in tokens_a:
        tokens.append(token)
    tokens.append("[SEP]")

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < seq_length:
        input_ids.append(0)
        input_mask.append(0)

    assert len(input_ids) == seq_length
    assert len(input_mask) == seq_length

    return InputFeatures(
        unique_id=ex_index,
        sent_id = sid,
        tokens=tokens,
        input_ids=input_ids,
        input_mask=input_mask)

def convert_examples_to_features(examples, seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    src_features_all, tgt_features_all = [],[]
    num_src_sent, num_tgt_sent = 0, 0
    for (ex_index, example) in enumerate(examples):
        src_features, tgt_features = [], []
        for sid, one_sent in enumerate(example.src_sents):
            src_features.append(parse_one_document(ex_index,sid, one_sent, seq_length, tokenizer))
        for sid, one_sent in enumerate(example.tgt_sents):
            tgt_features.append(parse_one_document(ex_index,sid, one_sent, seq_length, tokenizer))
        num_src_sent += len(src_features)
        num_tgt_sent += len(tgt_features)

        src_features_all.append(src_features)
        tgt_features_all.append(tgt_features)

    print('Number of features: ',len(src_features_all))
    print('Averaged source/target sentence numbers: %.1f/%.1f'%(
        num_src_sent/len(src_features_all)*100.0,num_tgt_sent/len(tgt_features_all)*100.0 ))

    return src_features_all, tgt_features_all


# We can load two different types of the sentence representations : (1)pre-trained BERT, (2) GloVE (or W2V if file exists.)
class Encoder(object):
    def __init__(self, path = '../../../../data/neuralsum', emb='glove', bert_encode_type='last'):
        self.path = path
        self.emb = emb
        self.bert_encode_type = bert_encode_type
        self.length_norm = False

        #emb_name = '{}-{}'.format(self.emb,self.bert_encode_type) if self.emb == 'bert' else self.emb
        #self.data_file = os.path.join(self.path,'data_%s_%s_%s_%s.json' %(dataset, dataset_type, emb_name, self.length_norm))


    def encode(self, pairs):
        """
        Input : Sentence pairs (src, tgt)
        Output: Sentence vector(representation) pairs (src, tgt)
        """
        print('Loading Embeddings from...', self.emb)
        if self.emb == 'bert':
            example_vecs, example_vecs_tgt = \
                    self.sentence_encoding_with_bert(
                            pairs, bert_encode_type='last')
        elif self.emb == 'glove':
            example_vecs, example_vecs_tgt = \
                    self.sentence_encoding_with_w2v(
                            pairs, os.path.join(self.path,'glove.840B.300d.w2v.bin'))

        return example_vecs, example_vecs_tgt

    def sentence_encoding_with_bert(self,
            pairs,
            bert_model='bert-base-uncased',
            bert_encode_type='last'):

        print('Loading bert from %s' % (bert_model))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_gpu = torch.cuda.device_count()
        print("device: {} n_gpu: {}".format(device, n_gpu))

        layer_indexes = [-1] #[-1,-2,-3,-4]

        tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=True)

        src_features, tgt_features = convert_examples_to_features(
            examples=pairs, seq_length=20,tokenizer=tokenizer)

        model = BertModel.from_pretrained(bert_model)
        model.to(device)

        if n_gpu > 1:
            model = torch.nn.DataParallel(model)

        def get_features(features):
            features_flattend = [s for d in features for s in d]
            features_dic = {}
            for id,f in enumerate(features_flattend):
                features_dic[id] = (f.unique_id, f.sent_id)

            all_input_ids = torch.tensor([s.input_ids for d in features for s in d], dtype=torch.long)
            all_input_mask = torch.tensor([s.input_mask for d in features for s in d], dtype=torch.long)
            all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)

            print('Total sentences:',len(features_flattend))

            eval_data = TensorDataset(all_input_ids, all_input_mask, all_example_index)
            eval_sampler = SequentialSampler(eval_data)
            eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=64)

            model.eval()

            doc_features = defaultdict(lambda: defaultdict(list))

            pbar = tqdm(total=len(features_flattend))

            for input_ids, input_mask, example_indices in eval_dataloader:
                input_ids = input_ids.to(device)
                input_mask = input_mask.to(device)

                all_encoder_layers, _ = model(input_ids, token_type_ids=None, attention_mask=input_mask)
                all_encoder_layers = all_encoder_layers

                for b, example_index in enumerate(example_indices):
                    # per sentence
                    feature = features_flattend[example_index.item()]
                    unique_id = int(feature.unique_id)
                    sentence_id = int(feature.sent_id)

                    # print(example_index.item(),len(features_flattend),b,unique_id, sentence_id)
                    all_out_sent_features = []

                    doc_feature = []
                    for (i, token) in enumerate(feature.tokens):
                        all_layers = []
                        for (j, layer_index) in enumerate(layer_indexes):
                            layer_output = all_encoder_layers[int(layer_index)].detach().cpu().numpy()
                            # import pdb; pdb.set_trace()
                            layer_output = layer_output[b]
                            layers = OrderedDict()
                            layers["index"] = layer_index
                            layers["values"] = [
                                round(x.item(), 6) for x in layer_output[i]
                            ]
                            all_layers.append(layers)

                        out_features = OrderedDict()
                        out_features["token"] = token
                        out_features["layers"] = all_layers
                        all_out_sent_features.append(out_features)

                    if bert_encode_type == 'last':
                        tok = all_out_sent_features[-1]['token']
                        doc_feature = all_out_sent_features[-1]['layers'][0]['values']
                    elif bert_encode_type == 'avg':
                        fs = []
                        for sent_feature in all_out_sent_features:
                            #tok = sent_feature['token']
                            fs.append(sent_feature['layers'][0]['values'])
                        doc_feature = np.mean(fs)
                    else:
                        print('Not yet implemented')
                        sys.exit(1)

                    doc_features[unique_id][sentence_id] = doc_feature

                pbar.update(len(example_indices))

            print('Document numbers:',len(doc_features))
            print('Feature dimension:',len(doc_features[0][0]))

            # change dictionary to list
            doc_out_features = []
            for did, doc_feature in doc_features.items():
                out_features = []
                for idx in sorted(list(doc_feature.keys())):
                    out_features.append(doc_feature[idx])
                doc_out_features.append(out_features)
            return doc_out_features

        print ('Processing source documents...')
        example_vecs = get_features(src_features)
        print ('Processing target documents...')
        example_vecs_tgt = get_features(tgt_features)

        return example_vecs, example_vecs_tgt


    def sentence_encoding_with_w2v(self, pairs, w2v_file):
        print('\tloading word2vec from %s' % (w2v_file))
        model = gensim.models.KeyedVectors.load_word2vec_format(w2v_file, binary=True)
        print('\tgenerating w2v projection data')

        # reduced w2v matches
        ws = []
        wt = []
        for example in pairs:
            for sentence in example.src_sents:
                ws += sentence
            for sentence_tg in example.tgt_sents:
                wt += sentence_tg

        print("Num of Vocabs: %d" % (len(ws + wt)))
        words_list = set(list(ws + wt))

        ws = defaultdict()
        for w in words_list:
            if w in model.vocab:
                ws[w] = model[w]

        print("Num of Unique Vocabs: %d" % (len(ws)))

        example_vecs = []
        example_vecs_tgt = []
        w2v_len = len(model['the'])
        for idx, example in enumerate(pairs):
            sentence_vecs = []
            sentence_vecs_tgt = []

            for sentence in example.src_sents:
                word_vecs = []
                for word in sentence:
                    if word in ws.keys():
                        if self.length_norm:
                            word_vecs.append(preprocessing.normalize(
                                ws[word].reshape(1, -1), norm='l2').reshape(-1))
                        else:
                            word_vecs.append(ws[word])

                if len(word_vecs) > 0:
                    sentence_vecs.append(np.mean(word_vecs, 0))

            for sentence_tgt in example.tgt_sents:
                word_vecs_tgt = []
                for word_tgt in sentence_tgt:
                    if word_tgt in ws.keys():
                        if self.length_norm:
                            word_vecs_tgt.append(preprocessing.normalize(
                                ws[word_tgt].reshape(1, -1), norm='l2').reshape(-1))
                        else:
                            word_vecs_tgt.append(ws[word_tgt])

                if len(word_vecs_tgt) > 0:
                    sentence_vecs_tgt.append(np.mean(word_vecs_tgt, 0))

            if len(sentence_vecs_tgt) > 0:
                example_vecs.append(sentence_vecs)
                example_vecs_tgt.append(sentence_vecs_tgt)

            else:
                example_vecs.append([np.zeros(w2v_len)])
                example_vecs_tgt.append([np.zeros(w2v_len)])

        del model
        return example_vecs, example_vecs_tgt


