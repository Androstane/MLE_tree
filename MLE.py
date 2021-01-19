from funcs import NJ, add_edgeL, felsenstein, felsenstein_m
from parse import parse
import numpy as np
import random
import ete3
from ete3 import Tree
import SPR
import NNI
import sys
from edge import uniform, exponential, lognormal 
import argparse


def predict_move():
    i = random.random()
    #modify orig_height 
    if i < 1/6: 
        return 0
    #modify edge
    if 1/6 <= i and i < 1/2:
        return 1
    #modify topology
    else:
        return 2

#prob from the exponential move
def prob_from_root(M, orig_time):
    log_likelihood = 0
    for site_i in range(len(M)):
        orig_likelihood = 0
        for child_state in range(len(M[0])):
                orig_likelihood += edge_dep_tp(child_state, 2, orig_time) * M[site_i][child_state]
        log_likelihood += np.log(orig_likelihood)
    return log_likelihood


def MLE(path, niter, return_condition, n_tree, orig_time): 
    out, dist_matrix, leaf_name, profile = parse(path)
    NJ_tree = NJ(dist_matrix, leaf_name,out)
    #print(NJ_tree)
    NJ_tree = add_edgeL(NJ_tree)
    tree = NJ_tree
    #start of iteration
    i = 0
    n_states = np.max(list(profile.values())) + 1
    print("how many states:", n_states)
    move_id = predict_move()
    if move_id == 0:
        p, M = felsenstein_m(tree, profile, n_states, orig_time)
    else:
        p = felsenstein(tree, profile, n_states, orig_time) 
        M = False
    count = 0
    spr_thr = 5
    topos = set()
    topos.add(tree.get_topology_id())
    #Tree Move
    while i < niter and count < return_condition: 
        print('iter', i)
        update = False
        next_move = predict_move()
        if move_id == 0: 
            root = tree.get_tree_root()
            orig_time_new = numpy.random.exponential(root.height)
            orig_time_new = np.rint(orig_time_new)
            if M is False:
                pi, M = felsenstein_m(tree, profile, n_states, orig_time)
            else:
                p_i = prob_from_root(M, orig_time_new)
            if p_i > p: 
                orig_time = orig_time_new
                print("Accept tree after change orig_time at ", i, "th iteration", "with probability of ", p)
                update = True

        else if move_id == 1:
            move = np.random.choice([uniform, lognormal])
            suggest_tree = move(tree)
            if next_move == 0:
                p_i, M_i = felsenstein_m(suggest_tree, profile, n_states, orig_time)
            else:
                p_i = felsenstein(suggest_tree, profile, n_states, orig_time)
                M_i = False
            if p_i > p: 
                tree = suggest_tree
                p = p_i
                M = M_i
                print("Accept tree after new edge length at ", i, " th iteration", "with probability of ", p)
                update = True
        else: 
            if count < spr_thr:
                tree_list = NNI.Main(tree, n_tree)
                for t_i in tree_list:
                    if not t_i.get_topology_id() in topos:
                        topos.add(t_i.get_topology_id())
                        if next_move == 0:
                            p_i = felsenstein(t_i, profile, n_states, orig_time)
                            M_i = False
                        else:
                            p_i, M_i = felsenstein_m(t_i, profile, n_states, orig_time)
                        if random.random() < p_i/p:
                            tree = t_i
                            p = p_i 
                            print("Accept tree after NNI at ", i, "th iteration", "with probability of ", p)
                            update = True
                            count = 0
            else:
                tree_list = SPR.Main(tree, n_tree, 1)
                for t_i in tree_list:
                    if not t_i.get_topology_id() in topos:
                        topos.add(t_i.get_topology_id())
                        if next_move == 0:
                            p_i = felsenstein(t_i, profile, n_states, orig_time)
                            M_i = False
                        else:
                            p_i, M_i = felsenstein_m(t_i, profile, n_states, orig_time)
                        if random.random() < p_i/p:
                            tree = t_i
                            p = p_i
                            print("Accept tree after SPR at ", i, "th iteration", "with probability of ", p)
                            update = True
                            count = 0
        if not update: 
            count = count + 1
        i = i + 1
        move_id = next_move


    return tree, p, orig_time, leaf_name


if __name__ == "__main__":
    orig_stdout = sys.stdout
    ap = argparse.ArgumentParser()
    ap.add_argument("-in","--path to input",required=True, help="Path to the input folder containing the copy number profiles ")
    ap.add_argument("-niter","--number of iteration",required=False, help="number of iteration, default to 1 ")
    args = vars(ap.parse_args())
    if args['path to input']!=None:
        path = args['path to input']
    if args['number of iteration']!=None:
        niter = int(args['number of iteration'])
    else: 
        niter = 1
    f = open(path + '/out.txt', 'w+')
    sys.stdout = f
    T, prob, orig_time, leaf_name = MLE(path, niter, 5, 1, 1.0)  
    tree_file = path + '/mle_tree.nw'
    T.write(format=9, outfile=tree_file)
    ff = open(tree_file, "w+")
    ff.write("\n")
    ff.write(str(prob))
    ff.write("\n")
    ff.write(str(orig_time))
    ff.write("\n")
    ff.write(leaf_name)
    ff.close()
    print("prob:", prob)
    print("leaf name:", leaf_name)
    print("orig_time", orig_time)
    print(T.write(format = 9))
    sys.stdout = orig_stdout
    f.close()




            