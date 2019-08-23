import copy
import glob
import Levenshtein
import mlx as ml
import multiprocessing
import numpy as np
import os
import pickle
from scipy.stats import wasserstein_distance
import tqdm


def Levenshtein_matrix(sequences):
	'''
	Simple utility for generating an n x n matrix of
	Levenshtein distances where n is the number of sequences.
	Note that this matrix is lower triangular.
	'''
    matrix = np.zeros((len(sequences),len(sequences)))
    
    for row in tqdm.tqdm(range(0,len(sequences))):
        for column in range(0, len(sequences)):
            if row >= column:
                matrix[row][column] = Levenshtein.distance(sequences[row],sequences[column])
            
    return matrix


def getFmap(PATH_TO_TREES):
	'''
	Takes the glob path to the qnet decision trees and returns a dictionary
	of potential class for each position and of the trees themselves.
	'''
    F={}
    TREE={}
    TREES=glob.glob(PATH_TO_TREES)
    for filename in TREES:
        with open(filename,'rb') as f:
            TR = pickle.load(f)
        f.close()
        index=os.path.splitext(os.path.basename(filename))[0].split('_')[-1]

        F[index]=[]
        TREE[index]=TR
        for key,value in TR.feature.iteritems():
            if not TR.TREE_LEAF[key]:
                F[index]=np.append(F[index],value)
    return F,TREE


def qDistance_seq_seq(seq0,seq1,F0,TREES0,F1,TREES1,overall_dist0,overall_dist1,\
                      distance_func=wasserstein_distance):
    '''
     computing genomic distance using qnets
    '''
    D0=getDistribution(seq0,F0,TREES0)
    D1=getDistribution(seq1,F1,TREES1)
    overall_dist0.update(D0)
    overall_dist1.update(D1)

    S=0.0
    nCount=0
    for key0 in overall_dist0.keys():
        p1,p2 = harmonize_dists(overall_dist0[key0],overall_dist1[key0]) #Make sure dists are of same length.
        S=S+distance_func(p1,p2)
        nCount=nCount+1
        
    if nCount == 0:
        nCount=1
    return S/(nCount+0.0)


def qDistance_seq_seq_parallel(args):
	'''
	A utility function which does the same as qDistance_seq_seq but in different format.
	This is intended for use in parallel.
	'''

    row, column, seq0, seq1, F0, TREES0, F1, TREES1, overall_dist0, overall_dist1,distance_func = args

    D0=getDistribution(seq0,F0,TREES0)
    D1=getDistribution(seq1,F1,TREES1)
    overall_dist0.update(D0)
    overall_dist1.update(D1)

    S=0.0
    nCount=0
    for key0 in overall_dist0.keys():

        p1,p2 = harmonize_dists(overall_dist0[key0],overall_dist1[key0]) #Make sure dists are of same length.
        S=S+wasserstein_distance(p1,p2)
        nCount=nCount+1        
    if nCount == 0:
        nCount=1

    return (row,column,S/(nCount+0.0))


def Qdistance_matrix_parallel(sequences,F,TREES,overall_copy0,overall_copy1,distance_func=wasserstein_distance):
	'''
	Generates an n x n matrix of the qdistance between sequences. The F and Trees argument is required.
	Two copies of the overall distribution is also required. Like the Levenshtein matrix,this will 
	also be lower triangular.
	'''
    args = []
    for row in range(0,len(sequences)):
        for column in range(0, len(sequences)):
            if row >= column:
                args.append([row,column, sequences[row],sequences[column], F,TREES,F,TREES,\
                    overall_copy0,overall_copy1,distance_func])
    pool = multiprocessing.Pool(4)
    distances = pool.map(qDistance_seq_seq_parallel, args)
    matrix = np.zeros((len(sequences),len(sequences)))

    for d in distances:
        matrix[d[0]][d[1]] = d[2]
    
    return matrix   


def QMatrix_parrallel_two_trees(sequences,tree_F_dict,distance_func=wasserstein_distance):
	'''
	Like the qDistance_matrix_parallel function, but allows for the use of two different 
	set of trees. Requires a tree_F_dict, which maps the sequences to their respective tree,
	overall distribution, and F map. 
	'''

    args = []
    for row in range(0,len(sequences)):
        for column in range(0, len(sequences)):
            if row >= column:
                seq0 = sequences[row]
                seq1 = sequences[column]
                args.append([row,column, seq0,seq1,\
                    tree_F_dict[seq0][0],tree_F_dict[seq0][1],\
                    tree_F_dict[seq1][0],tree_F_dict[seq1][1],\
                    tree_F_dict[seq0][2],tree_F_dict[seq1][2],distance_func])
    pool = multiprocessing.Pool(4)

    distances = pool.map(qDistance_seq_seq_parallel, args)

    matrix = np.zeros((len(sequences),len(sequences)))
    for d in distances:
        matrix[d[0]][d[1]] = d[2]
    
    return matrix   

def getDistribution(seq,F, TREES):
	'''
	Grabs the conditional probability distribution of the sequence for every
	position.
	'''
    dists = {}
    for KEY in F.keys():
        I=[int(x.replace('P','')) for x in F[KEY]]
        DICT_={'P'+str(i):seq[i] for i in I}
        D=ml.sampleTree(TREES[KEY],DICT_,sample='random',DIST=True)[1]
        dists[KEY]= D
    return dists


def harmonize_dists(dist1,dist2):
	'''
	In the case where two populations are using two different sets of
	trees, we need to ensure that there is an equal number of
	distributions in both distribution set and that each distribution
	corresponds to the same position in each set.
	'''
    for key1 in dist1:
        if key1 not in dist2:
            dist2[key1] = 0.0
            
    for key2 in dist2:
        if key2 not in dist1:
            dist1[key2] = 0.0

    p1 = [dist1[x] for x in sorted(dist1.keys())]
    p2 = [dist2[x] for x in sorted(dist2.keys())]

    return p1,p2


def overall_distribution(sequence_set,length):
	'''
	Using distribution metrics like Wasserstein requires that
	we have two equal length distribution sets and that each position
	in both distribution set corresponds to the same position
	of the qNet. However, sometimes a tree is not generated for
	a certain qNet position. In that case, default to the overall
	distribution at that position.
	'''
	
    distribution = {}
    N = float(len(sequence_set))
    for n in range(length):
        key = 'P' + str(n)
        letter_count = {}
        for seq in sequence_set:
            letter = seq[n]
            if letter not in letter_count:
                letter_count[letter] = 0
            letter_count[letter] += 1
        
        letter_dist = {key:float(letter_count[key] / N) for key in letter_count}
        distribution[key] = letter_dist
        
    return distribution
