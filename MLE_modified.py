from funcs import NJ, add_edgeL, felsenstein
from parse import parse
import numpy as np
import random
import ete3
from ete3 import Tree
import SPR
import NNI
import sys

orig_stdout = sys.stdout
f = open('out.txt', 'w+')
sys.stdout = f

#path - path to file 
#niter - max iteration for tree search 
#return_condition - assume best tree after found how many iteration no new tree accepted 
#n_tree - keep how many tree from tree move 
def MLE(path, niter, return_condition, n_tree): 
    out, dist_matrix, leaf_name, profile = parse(path)
    NJ_tree = NJ(dist_matrix, leaf_name, out)
    #print(NJ_tree)
    NJ_tree = add_edgeL(NJ_tree)
    tree = NJ_tree
    #start of iteration
    i = 0
    n_states = np.max(list(profile.values())) + 1
    print("how many states:", n_states)
    p = felsenstein(tree, profile, n_states) 
    count = 0
    spr_thr = 5
    topos = set()
    topos.add(tree.get_topology_id())
    while i < niter and count < return_condition: 
        print('iter', i)
        update = False
        if count < spr_thr:
            tree_list = NNI.Main(tree, n_tree)
            for t_i in tree_list:
                if not t_i.get_topology_id() in topos:
                    topos.add(t_i.get_topology_id())
                    p_i = felsenstein(t_i, profile, n_states)
                    if random.random() < p_i/p:
                        tree = t_i
                        p = p_i 
                        update = True
                        count = 0
        else:
            tree_list = SPR.Main(tree, n_tree, 1)
            for t_i in tree_list:
                if not t_i.get_topology_id() in topos:
                    topos.add(t_i.get_topology_id())
                    p_i = felsenstein(t_i, profile, n_states)
                    if random.random() < p_i/p:
                        tree = t_i
                        p = p_i
                        update = True
                        count = 0
        #select tree move:
        # if random.random() < 0.5:
        #     tree_list = SPR.Main(tree, n_tree, n_tree)
        #     for t_i in tree_list:
        #         if not t_i.get_topology_id() in topos:
        #             p_i = felsenstein(t_i, profile, n_states)
        #             if p_i < p:
        #                 tree = t_i
        #                 p = p_i
        #                 update = True
        #                 count = 0
        # else:
        #     tree_list = NNI.Main(tree, n_tree)
        #     for t_i in tree_list:
        #         if not t_i.get_topology_id() in topos:
        #             p_i = felsenstein(t_i, profile, n_states)
        #             if p_i < p:
        #                 tree = t_i
        #                 p = p_i
        #                 update  = True
        #                 count = 0 
        if not update: 
            count = count + 1
        i = i + 1
     
    return tree, p
path = 'rep1' 
T, prob= MLE(path, 2, 5, 1)  
tree_file = path + '/mle_tree.nw'
file = open(tree_file, "w+")
file.write("\n")
file.write(str(prob))
file.close()
T.write(format=9, outfile=tree_file)
print("prob:", prob)
print(T.write(format = 9))
sys.stdout = orig_stdout
f.close()