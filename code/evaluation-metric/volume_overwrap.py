import gc
import os
import argparse
import numpy as np

#Before this code, we should make a file of intersection ratio using decoder.py
class VolumeOverwrap():

    def __init__(self,args):

        # parameters
        self.dataset = args.dataset  # 'cnndm'
        self.dataset_type = args.dataset_type  # 'val'
        self.emb = args.emb # 'glove'
        self.bert_encode_type = args.bert_encode_type
        self.path = args.path
        self.overlap_type = args.overlap_type

    def volume_overwrap(self,algorithm):
        intersect_path = os.path.join(self.path, 'ext', self.dataset, self.dataset_type,
                                      self.emb, algorithm,'%s_overwrap.txt' %(self.overlap_type))
        intersect_out_path = os.path.join(self.path, 'ext','stats')

        if not os.path.exists(intersect_path):
           print("Error: Calculate %s_overwrap ratio first!" %(self.overlap_type))
           return

        if not os.path.exists(intersect_out_path):
            os.makedirs(intersect_out_path)

        print("Reading %s %s %s %s %s..." %(self.dataset, self.dataset_type, self.emb, algorithm,self.overlap_type),
              end='\n', flush=True)
        with open(intersect_path, 'rt') as f:
            ratios = []
            idxs = {}
            for i, line in enumerate(f):
                intersect = line.strip().split('\t')
                if len(intersect) != 2:
                    continue
                if intersect[0] in idxs:
                    continue

                ratios.append(float(intersect[1]))

        gc.collect()

        # Save volume_overwrap
        avg_overwrap = np.mean(ratios)*100
        txt_intersect = open(os.path.join(intersect_out_path, 'avg_%s_overwrap.txt' %self.overlap_type), 'a')
        txt_intersect.write('%s\t%s\t%s\t%s\t%d\t%.6f\n'
                            %(self.dataset, self.dataset_type, self.emb, algorithm, len(ratios),avg_overwrap))
        txt_intersect.close()

        print("Avg save done!")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    #Data/Encoder
    parser.add_argument("--dataset",  default='cnndm')
    parser.add_argument("--dataset_type", default='test')
    parser.add_argument("--emb", default='glove')
    parser.add_argument("--bert_encode_type", default='last')
    parser.add_argument("--path", default='../../../../data/neuralsum/')
    parser.add_argument("--overlap_type", default='oracle')

    # Extractive Algorithms
    parser.add_argument("--algos", nargs='+',
                        default=['first','last', 'random', 'mid', 'hard_convex', 'hard_convex_waterfall', 'hard_heuristic','farthest','nearest','knn','kmeans','mmr'])

    args = parser.parse_args()

    for algo in args.algos:
        avg_volume = VolumeOverwrap(args)
        avg_volume.volume_overwrap(algo)
