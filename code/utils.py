import os
import re
import gc
import time
from tempfile import TemporaryDirectory
import subprocess
import numpy as np
import math

from scipy.spatial import Delaunay
from typing import List, Dict, Tuple
import gzip

def rouge(target: List[List[str]], *predictions: List[List[str]]) -> List[Dict[str, float]]:
    """Perform single-reference ROUGE evaluation of one or more systems' predictions."""
    results = [dict() for _ in range(len(predictions))]    # e.g. 0 => 'su4_f' => 0.35

    with TemporaryDirectory() as folder:    # on my server, /tmp is a RAM disk
        # write SPL files
        eval_entries = []
        for i, tgt_tokens in enumerate(target):
            sys_entries = []
            for j, pred_docs in enumerate(predictions):
                sys_file = 'sys%d_%d.spl' % (j, i)
                sys_entries.append('\n        <P ID="%d">%s</P>' % (j, sys_file))
                with open(os.path.join(folder, sys_file), 'wt') as f:
                    f.write(format_tokens(pred_docs[i], for_rouge=True))
            ref_file = 'ref_%d.spl' % i
            with open(os.path.join(folder, ref_file), 'wt') as f:
                f.write(format_tokens(tgt_tokens, for_rouge=True))

            eval_entry = """
<EVAL ID="{1}">
    <PEER-ROOT>{0}</PEER-ROOT>
    <MODEL-ROOT>{0}</MODEL-ROOT>
    <INPUT-FORMAT TYPE="SPL"></INPUT-FORMAT>
    <PEERS>{2}
    </PEERS>
    <MODELS>
        <M ID="A">{3}</M>
    </MODELS>
</EVAL>""".format(folder, i, ''.join(sys_entries), ref_file)
            eval_entries.append(eval_entry)
        # write config file
        xml = '<ROUGE-EVAL version="1.0">{0}\n</ROUGE-EVAL>'.format("".join(eval_entries))
        config_path = os.path.join(folder, 'task.xml')
        with open(config_path, 'wt') as f:
            f.write(xml)
        # run ROUGE
        #a -c 95 -m -n 2 -2 4 -u
        out = subprocess.check_output('./ROUGE-1.5.5.pl -e data -a -c 95 -m -n 2 -2 4 -u ' + config_path,
                                    shell=True, cwd=os.path.join(this_dir, 'data'))
    # parse ROUGE output
    for line in out.split(b'\n'):
        match = rouge_pattern.match(line)
        if match:
            sys_id, metric, rpf, value, low, high = match.groups()
            results[int(sys_id)][(metric + b'_' + rpf).decode('utf-8').lower()] = float(value)
    return results

def clockwise(arr):
    if not type(arr) == list:
        arr2 = np.ndarray.tolist(arr)

    cent = (sum([p[0] for p in arr2])/len(arr2),sum([p[1] for p in arr2])/len(arr2))
    arr2.sort(key=lambda p: math.atan2(p[1]-cent[1],p[0]-cent[0]))

    return np.array(arr2)

def get_distance(v1, v2):
    x = np.asarray(v1 - v2, dtype=np.float64)
    return np.linalg.norm(x)
    # return np.linalg.norm(v1 - v2)


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def ngram_generator(src, n):
    src = [str(x) for x in src]
    out = []
    if len(src) < n:
        out.append('_'.join(src))
    else:
        for idx in range(0,len(src)-n+1):
            out.append('_'.join(src[idx:idx+n]))

    return out


#PCA Result save
def pca_save(pairs, emb, bert_encode_type, path, filename, r, max_sample=10000, max_sents=100):
    start = time.time()
    limit_sent_num_for_pca = 10
    src_sent_vecs = []
    #Set up max sample = 10000
    #Set pu max sents = 100
    if len(pairs) > max_sents:
    random.shuffle(pairs)
    pairs = pairs[:max_sample]
    for i, pair in enumerate(pairs):
        if len(pair.src_sents) > max_sents:
            rand = random.sample(range(len(pair.src_sents)),max_sents)
            pairs[i]=Example(src_sents = np.array(pair.src_sents)[rand].tolist(),tgt_sents = pair.tgt_sents)

    if emb=='glove':
        w2v_file = os.path.join(path,'glove.840B.300d.w2v.bin')
        model = gensim.models.KeyedVectors.load_word2vec_format(w2v_file,
                binary=True)
        w2v_len = len(model['the'])
        pbar=tqdm(len(pairs))
        for i, sample in enumerate(pairs):  # for each sample
            pbar.update(1)
            sample_sent_vecs = []
            for sent in sample.src_sents[:limit_sent_num_for_pca]:
                indiv_sent_vec = []
                for word in sent:
                    if word in model.vocab:
                        indiv_sent_vec.append(model[word])
                if len(indiv_sent_vec) > 0:
                    sample_sent_vecs.append(np.mean(indiv_sent_vec, 0).tolist())
                else:
                    sample_sent_vecs.append(np.zeros(w2v_len).tolist())
            src_sent_vecs.append(sample_sent_vecs)

    elif emb=='bert':
        e = Encoder(path=path, emb=emb, bert_encode_type=bert_encode_type)
        sample_sent_vecs, _ = e.encode(pairs=pairs)
        for i, sample in enumerate(sample_sent_vecs):
            src_sent_vecs.append(sample[:limit_sent_num_for_pca])

    print('Total source sent vectors:', len(src_sent_vecs))
    print('Start PCA... emb:%s'%(emb))
    pca = PCA(n_components=r)
    pca.fit(np.array([x for y in src_sent_vecs for x in y]))
    print('PCA done: %.2f minutes' % ((time.time() - start) / 60))
    pkl.dump(pca, open(filename, 'wb'))

    return pca


def CalculateArea(arr):
    return 0.5 * np.abs(np.dot(arr[:, 0], np.roll(arr[:, 1], 1)) - np.dot(arr[:, 1]    , np.roll(arr[:, 0], 1)))

def PyramidVol(arr):
    def determinant_3x3(m):
        return (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
                m[1][0] * (m[0][1] * m[2][2] - m[0][2] * m[2][1]) +
                m[2][0] * (m[0][1] * m[1][2] - m[0][2] * m[1][1]))

    def subtract(a, b):
        return (a[0] - b[0],
                a[1] - b[1],
                a[2] - b[2])

    return (abs(determinant_3x3(
        (subtract(arr[0], arr[1]),
         subtract(arr[1], arr[2]),
         subtract(arr[2], arr[3]),))) / 6.0)

def PolyVol(arr,r):
    if r == 2:
        polygon = clockwise(arr)
        return CalculateArea(polygon)

    elif r == 3:
        if len(arr) < r + 1:
            return 0
        else:
            delaunay = Delaunay(arr)
            vol = 0
            for i in delaunay.simplices:
                vol += PyramidVol(arr[i])
                return vol
