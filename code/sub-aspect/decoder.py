import os
import sys
import gzip,time
import random
import scipy
import argparse
import pickle as pkl
import numpy as np

from scipy.spatial import ConvexHull
from tqdm import tqdm

from method import *
#pca_save, mid, hard_waterfall, hard_heuristic, farthest, nearest_np, knn

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from data import Dataset
from encoder import Encoder
from utils import chunks, get_distance
from evaluation-metric.evaluation import volume_overwrap

class ExtractiveOut():

    def __init__(self,args):

        # parameters
        self.dataset = args.dataset  # 'cnndm'
        self.dataset_type = args.dataset_type  # 'val'
        self.emb = args.emb # 'glove'
        self.bert_encode_type = args.bert_encode_type
        self.path = args.path
        self.save_volume_overlap = args.save_volume_overlap

        self.r = args.pca_components

    def extractive_outputs(self, pairs, data,  org_idxs, algorithm):

        #print('Building volumes...')
        #First, make a PCA components for each pairs for later volume overlap evaluation.
        filename = os.path.join(self.path,'pca','pca_%s_%s_%s.pkl' %(self.dataset, self.dataset_type, self.emb))
        if os.path.isfile(filename):
            pca_fit = pkl.load(open(filename,'rb'))

        else:
            print('PCA file not exist')
            return

        #Final Output here would be vertices & intersect for each extractive summarization algorithms
        #except indexs : exception case exists if no ConvexHull can be generated or no sentence representation exists.
        idx_excepts = []

        print("Start %s" %(algorithm))
        pbar = tqdm(total=len(pairs))
        for idx in range(len(pairs)):
            pbar.update(1)
            org_idx = org_idxs[idx]
            src_example = data[0][idx]
            tgt_example = data[1][idx]

            src_length = len(src_example)
            tgt_length = len(tgt_example)
            src_length_pairs = len(pairs[idx].src_sents)

            #pca_fit
            #Case no sentence embedding exists --> skip
            if src_example==[] or tgt_example==[]:
                idx_excepts.append(org_idx)
                continue

            pca_src_example = pca_fit.transform(src_example)
            pca_tgt_example = pca_fit.transform(tgt_example)

            # Pass example when (1) Load pairs and src length are not equal or
            # (2) src sentences are repeated
            if self.r >= src_length or tgt_length >= src_length:
                algo_vertice = range(src_length)
                if algorithm == 'hard_convex_waterfall':
                    if tgt_length == 1:
                        centroid = np.mean(pca_src_example, axis=0)
                        distance_list = []
                        for pca_src_vec in pca_src_example[range(src_length)]:
                            distance_list.append(get_distance(pca_src_vec, centroid))
                        algo_vertice = [range(src_length)[np.argmax(distance_list)]]

            #Only exception is that no length matching
            elif src_length != src_length_pairs:
                idx_excepts.append(org_idx)
                continue

            else:

                #Importance : n-nearest
                if algorithm == 'nearest':
                    algo_vertice = nearest_np(src_example, tgt_length)

                #Importance : k-nearest
                elif algorithm == 'knn':
                    cluster_size = int(np.sqrt(len(src_example)))
                    algo_vertice = knn(src_example, cluster_size, tgt_length)

                # Position: first-k, last-k, rand-k
                elif algorithm == 'first':
                    algo_vertice = range(tgt_length)

                elif algorithm == 'last':
                    algo_vertice = range(src_length_pairs - tgt_length, src_length_pairs)

                elif algorithm == 'random':
                    algo_vertice = sorted(np.random.choice(src_length_pairs, size=tgt_length, replace=False))

                elif algorithm == 'mid':
                    algo_vertice = mid(src_length_pairs, tgt_length)

                else:
                    # Convex hull -> find # of vertices, area and Volume
                    try:
                        ch_hull = ConvexHull(pca_src_example)
                    except (scipy.spatial.qhull.QhullError, ValueError) as e:
                        idx_excepts.append(idx)
                        continue
                        raise

                    #Diversity: Heuristic, Waterfall
                    if algorithm=='hard_convex': #Entire ConvexHull Vertice
                        algo_vertice = sorted(ch_hull.vertices)
                    elif algorithm == 'hard_convex_waterfall':
                        algo_vertice = hard_waterfall(ch_hull, pca_src_example, tgt_length, self.r)
                    elif algorithm == 'hard_heuristic':
                        algo_vertice = hard_heuristic(ch_hull, pca_src_example, tgt_length)

            #Save (1) Extractive Outputs for each algorithms
            #Location: data -> neuralsum -> ext -> dataset -> dataset_type ->algorithm -> decoded
            #Location: data -> neuralsum -> ext -> dataset -> dataset_type -> target
            #If folder doesn't exist --> make it
            #Set up path
            decoded_path = os.path.join(self.path,'ext',self.dataset,self.dataset_type,self.emb,algorithm,'decoded')
            tgt_path = os.path.join(self.path,'ext',self.dataset,self.dataset_type,self.emb,'target')
            intersect_path = os.path.join(self.path,'ext',self.dataset,self.dataset_type,self.emb,algorithm)
            if not os.path.exists(decoded_path):
                os.makedirs(decoded_path)
            if not os.path.exists(tgt_path):
                os.makedirs(tgt_path)
            if not os.path.exists(intersect_path):
                os.makedirs(intersect_path)

            #Save summaries as separate files
            idx_pad = str(org_idx).zfill(6)

            if os.path.exists(os.path.join(tgt_path,'%s_target.txt' %(idx_pad))) and \
                os.path.exists(os.path.join(decoded_path,'%s_decoded.txt' %(idx_pad))):
                 continue

            txt_tgt = open(os.path.join(tgt_path,'%s_target.txt' %(idx_pad)),'w')
            txt_tgt.write('\n'.join([' '.join(x) for x in pairs[idx].tgt_sents]))
            txt_tgt.close()

            src_decoded = np.array(pairs[idx].src_sents)[algo_vertice].tolist()
            txt_decode= open(os.path.join(decoded_path,'%s_decoded.txt' %(idx_pad)),'w')
            txt_decode.write('\n'.join([' '.join(x) for x in src_decoded]))
            txt_decode.close()

            #Save vertice indexs
            txt_vertice = open(os.path.join(intersect_path, 'selected_vertices.txt'),'a')
            txt_vertice.write('%d\t%s\n'%(org_idx,[x for x in algo_vertice]))
            txt_vertice.close()

            #Save volume_overwrap
            if self.save_volume_overlap:
                try:
                    volume = volume_overwrap(pca_src_example, algo_vertice, pca_tgt_example)
                except ValueError:
                    continue
                    raise
                txt_intersect = open(os.path.join(intersect_path,'volume_overwrap.txt'),'a')
                txt_intersect.write('%d\t%.6f\n'%(org_idx,volume))
                txt_intersect.close()

        print('total pairs',len(pairs),'including', len(pairs) - len(idx_excepts),'except',len(idx_excepts))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    #Data/Encoder
    parser.add_argument("--dataset",  default='peerread')
    parser.add_argument("--dataset_type", default='test')
    parser.add_argument("--emb", default='bert')
    parser.add_argument("--bert_encode_type", default='last')
    parser.add_argument("--path", default='../../../../data/neuralsum')
    parser.add_argument("--encoder_path", default='../../../../data/word2vec')

    #Save Overlap options
    parser.add_argument("--save_volume_overlap", default=True)

    #PCA - Dims
    parser.add_argument("--pca_components", type=int, default=2)

    #Batch
    parser.add_argument("--batch", type=bool, default=True)
    parser.add_argument("--batch_size", type=int, default=25000)
    parser.add_argument("--batch_idx", type=int, default=0)
    # Extractive Algorithms
    parser.add_argument("--algos", nargs='+',
                        default=['first', 'last', 'random', 'mid', # Position
                                 'hard_convex', 'hard_convex_waterfall', 'hard_heuristic', # Diversity
                                 'nearest', 'knn' # Importance
                                ])

    args = parser.parse_args()
    curr_batch_order = args.batch_idx

    d = Dataset(dataset=args.dataset, dataset_type=args.dataset_type, path=args.path)
    pairs = d.load_dataset()

    # #Create PCA if not exists
    filename = os.path.join(args.path, 'pca', 'pca_%s_%s_%s.pkl' % (args.dataset, args.dataset_type, args.emb))
    if not os.path.isfile(filename):
        print('PCA file make!')
        pca_save(pairs, args.emb, args.bert_encode_type, args.encoder_path, filename, args.pca_components)

    if args.batch:
        batches = chunks(pairs,args.batch_size)
        for i in range(curr_batch_order+1):
            batch = next(batches)

    else:
        batch = pairs

    print("batch from %d to %d" %(curr_batch_order*args.batch_size, curr_batch_order*args.batch_size+args.batch_size))

    #Load (or Save) Encoder File
    encoder_file = os.path.join(args.path,'bert','{}.{}.{}.{}.pkl'.format
                    (args.dataset, args.dataset_type, args.emb, args.batch_idx))

    if not os.path.isfile(encoder_file):
        e = Encoder(path=args.encoder_path, emb=args.emb, bert_encode_type=args.bert_encode_type)
        data = e.encode(pairs=batch)
        pkl.dump(data,open(encoder_file,'wb'))

    else:
        start = time.time()
        data = pkl.load(open(encoder_file, 'rb'))
        print("Total Encoding Time: %.2f mins" %(time.time() - start) / 60))

    ##SAVE BERT RESULT
    if args.batch:
        org_idxs=range(curr_batch_order*args.batch_size,curr_batch_order*args.batch_size+args.batch_size)
    else:
        org_idxs=range(len(batch))

    for algo in args.algos:
        exsum = ExtractiveOut(args)
        exsum.extractive_outputs(batch, data, org_idxs, algo)

