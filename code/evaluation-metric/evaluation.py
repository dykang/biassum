import os
import sys
import time
import gensim
import gc
import numpy as np

from shapely.geometry import Polygon as poly
from shapely.geometry import Point as point
from shapely.geometry import LineString as line


sys.path.insert(1, os.path.join(sys.path[0], '..'))
from utils import rouge, clockwise
from encoder import Encoder
from data import Example

# Calculate the volume overwrap ratio vs target
# Assume that all volumes are after 2-dimensional PCA.
def volume_overwrap(pca_src_example, vertices, pca_tgt_example):
    if len(vertices) == 0:
        return 0

    elif len(vertices) == 1:
        poly_input = point(pca_src_example[vertices][0])

    elif len(vertices) == 2:
        poly_input = line(pca_src_example[vertices])

    else:
        poly_input = poly(clockwise(pca_src_example[vertices]))

    if not poly_input.is_valid:
        poly_input = poly_input.buffer(0)

    if len(pca_tgt_example) == 0:
        return 0

    elif len(pca_tgt_example) == 1:
        poly_tgt = point(pca_tgt_example[0])
        return int(poly_input.contains(poly_tgt))

    elif len(pca_tgt_example) == 2:
        poly_tgt = line(pca_tgt_example)
        if poly_tgt.length > 0:
            return poly_input.intersection(poly_tgt).length / poly_tgt.length * 1.0
        else:
            return 0

    else:
        poly_tgt = poly(clockwise(pca_tgt_example))
        poly_tgt = poly_tgt.buffer(0)

        if poly_tgt.area > 0:
            return poly_input.intersection(poly_tgt).area / poly_tgt.area * 1.0

        else:
            return 0

def overwrap_ratio_print(bh_vertices, ex_vertices):
    return np.mean([len(set(x).intersection(y)) / len(x) * 1.0 for
                    x, y in zip(bh_vertices, ex_vertices) if len(x) > 0])

def rouge_print(gold_summaries, src_examples, src_vertices):
    src_rouge = [np.array(x)[sorted(y)].tolist() for x, y in zip(src_examples, src_vertices)]
    score = rouge([sum(x, []) for x in gold_summaries], [sum(y, []) for y in src_rouge])[0]
    return score['1_f'] * 100 / 1.0, score['2_f'] * 100 / 1.0, score['l_f'] * 100 / 1.0, score[
        'su4_f'] * 100 / 1.0

def total_rouges(pairs, idx_excepts, algorithm, vertices):

    idxs_include = [item for item in range(len(pairs)) if item not in idx_excepts]

    gold_summaries = np.array([ex.tgt_sents for ex in pairs])[idxs_include].tolist()
    src_examples = np.array([ex.src_sents for ex in pairs])[idxs_include].tolist()

    print("\nROUGES %s" % (algorithm))
    print("%s k: %.4f,%.4f,%.4f,%.4f" % (algorithm, rouge_print(gold_summaries, src_examples, vertices)))

def overwrap_with_oracle(pairs, idx_excepts, algorithm, vertices):

    idxs_include = [item for item in range(len(pairs)) if item not in idx_excepts]
    bh_idxs = np.array([ex.indexs for ex in pairs])[idxs_include].tolist()
    bh_rouges = np.array([ex.rouge for ex in pairs])[idxs_include].tolist()
    max_iter = max([len(x) for x in bh_idxs])

    for i in range(max_iter):
        # avg overwrap ratio (first,last,random,hard-heuristic,hard-waterfall,convex-hull)
        curr_bh_idxs = []
        curr_bh_rouges = []
        for j in range(len(bh_idxs)):
            if len(bh_idxs[j]) > i:
                curr_bh_idxs.append(bh_idxs[j][i])
                curr_bh_rouges.append(bh_rouges[j][i])
            else:
                curr_bh_idxs.append([])
        curr_bh_size = sum([1 for x in curr_bh_idxs if len(x) > 0])
        curr_avg_rouge = np.mean(curr_bh_rouges)[0]
        print("Ave ROUGE (%s/%s),(%s/%s) %.4f"
              % (i, max_iter, curr_bh_size, len(pairs), curr_avg_rouge))
        print("Overwrap %s (%s/%s),(%s/%s) %.4f"
              %(algorithm, i, max_iter, curr_bh_size, len(pairs), overwrap_ratio_print(curr_bh_rouges, vertices)))

def sample_rouges(self, pairs, idx_excepts, hh_vertices_rouge, wf_vertices_rouge):
    idxs_include = [item for item in range(len(pairs)) if item not in idx_excepts]
    gold_summaries = np.array([ex.tgt_sents for ex in pairs])[idxs_include].tolist()
    src_include = np.array([ex.src_words for ex in pairs])[idxs_include].tolist()
    tgt_size = np.array(np.array([len(ex.tgt_sents) for ex in pairs]))[idxs_include].tolist()
    sample_idxs = np.random.choice(len(src_include), size=10, replace=False)
    for idx in sample_idxs:
        first_k = src_include[idx][:tgt_size[idx]]
        last_k = src_include[idx][-tgt_size[idx]:]
        hh = np.array(src_include[idx])[sorted(hh_vertices_rouge[idx])].tolist()
        wf = np.array(src_include[idx])[sorted(wf_vertices_rouge[idx])].tolist()
        summary = gold_summaries[idx]
        # print out
        txt = open(os.path.join('../../figure/sample/', 'sample_%s_%s_%s.txt'
                                % (self.dataset, self.dataset_type, str(idx))), 'a')
        # Ref size: tot
        txt.write('%s\n' % [item for sub in summary for item in sub])
        txt.write('%s\n' % [item for sub in first_k for item in sub])
        txt.write('%s\n' % [item for sub in last_k for item in sub])
        txt.write('%s\n' % [item for sub in hh for item in sub])
        txt.write('%s\n' % [item for sub in wf for item in sub])
        txt.write('Rouge(First,Last,HH,WF): %s,%s,%s,%s'
                  % (
                  rouge([item for sub in summary for item in sub], [item for sub in first_k for item in sub]),
                  rouge([item for sub in summary for item in sub], [item for sub in last_k for item in sub]),
                  rouge([item for sub in summary for item in sub], [item for sub in hh for item in sub]),
                  rouge([item for sub in summary for item in sub], [item for sub in wf for item in sub])))
        txt.write('%s\n' % [item for sub in src_include[idx] for item in sub])
        txt.close()

    print("Done Save!")

def so_ratio(alg,ref):
    #Input (index vertices for algo and ref)
    #Output ratio
    if len(ref)==0:
        return
    return len(list(set(alg).intersection(set(ref))))/len(ref)
