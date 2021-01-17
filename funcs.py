
from scipy.linalg import expm
import scipy 
import ete3
from ete3 import Tree
import numpy as np
import scipy.special
import pandas as pd


import dendropy as dnd
import SPR
import NNI
import random 


def read_newick(newick_path):
    newick_file = open(newick_path)
    newick = newick_file.read().strip()
    newick_file.close()
    tree = ete3.Tree(newick)
    return tree


def recurse_likelihood(node, site_i, n_states):
    # set the liklihood at leaf 
    if node.is_leaf():
        node.partial_likelihoods.fill(0) # reset the leaf likelihoods
        leaf_state = node.sequence[site_i]
        node.partial_likelihoods[leaf_state] = 1
    # internal nodes 
    else:
        left_child, right_child = node.get_children()
        left_branch = np.float(left_child.dist)
        right_branch = np.float(right_child.dist)
        recurse_likelihood(left_child, site_i, n_states)
        recurse_likelihood(right_child, site_i, n_states)
        for node_state in range(n_states):
            left_partial_likelihood = 0.0
            right_partial_likelihood = 0.0
            for child_state in range(n_states):
                    left_partial_likelihood += edge_dep_tp(child_state, node_state, left_branch) * left_child.partial_likelihoods[child_state]
                    right_partial_likelihood += edge_dep_tp(child_state, node_state, right_branch) * right_child.partial_likelihoods[child_state]
            node.partial_likelihoods[node_state] = left_partial_likelihood * right_partial_likelihood

def transition_probability(child, ancestor, time):
    if time == 0: 
        return 0
    rate = 0.1
    if ancestor == 0:
        if child != 0:
            return 0 
        else: 
            return 1
    if child == 0:
        return np.power((rate * time) / (1 + rate * time), ancestor)
    if ancestor == 1: 
        return np.power(time,child - 1) / np.power((1 + time), child + 1)
    else:
        p = 0.0
        for j in range(1, min(child, ancestor) + 1):
            p_j = (scipy.special.binom(ancestor, j) *
                scipy.special.binom(child - 1, j - 1) *
                np.power(rate * time, -2 * j))
 
            p += p_j
 
        p *= np.power(time * rate / (1 + time * rate), child + ancestor)
 
        return p


def edge_dep_tp(child, ancestor, l):
    i = 1
    current_item = np.inf
    p = 0
    while i < np.inf and current_item > 1e-7 :
        c = np.power(l, i) * np.exp(-1* l)/np.math.factorial(i)
        #print(transition_probability(child, ancestor, np.float(i)))
        current_item = c * transition_probability(child, ancestor, np.float(i))
        p = p + current_item
        i = i + 1
    return p


#tree: ete3 tree object 
#profile: dictionary with key = node name, value = cna profile 
#n_states: how many possible cna states to consider 
#orig_time: time from root to its single child. 
def felsenstein(tree, profile, n_states, orig_time):
    log_likelihood = 0.0
    for node in tree.traverse():
        # initialize a vector of partial likelihoods that we can reuse for each site
        node.partial_likelihoods = np.zeros(n_states)
        if node.is_leaf():
            node.sequence = profile[node.name]
    site_count = len(list(profile.values())[0])
    for site_i in range(site_count):
        recurse_likelihood(tree, site_i, n_states)
        # assume the root is diploid
        #print(tree.partial_likelihoods)
        #log_likelihood += np.log(tree.partial_likelihoods[2])
        orig_likelihood = 0
        for child_state in range(n_states):
                orig_likelihood += edge_dep_tp(child_state, 2, orig_time) * tree.partial_likelihoods[child_state]
        log_likelihood += np.log(orig_likelihood)
    return log_likelihood

def felsenstein_m(tree, profile, n_states, orig_time):
    M = []
    log_likelihood = 0.0
    for node in tree.traverse():
        # initialize a vector of partial likelihoods that we can reuse for each site
        node.partial_likelihoods = np.zeros(n_states)
        if node.is_leaf():
            node.sequence = profile[node.name]
    site_count = len(list(profile.values())[0])
    for site_i in range(site_count):
        recurse_likelihood(tree, site_i, n_states)
        # assume the root is diploid
        #print(tree.partial_likelihoods)
        #log_likelihood += np.log(tree.partial_likelihoods[2])
        orig_likelihood = 0
        M.append(tree.partial_likelihoods)
        for child_state in range(n_states):
                orig_likelihood += edge_dep_tp(child_state, 2, orig_time) * tree.partial_likelihoods[child_state]
        log_likelihood += np.log(orig_likelihood)
    return log_likelihood, M


def add_edgeL(tree):
    for node in tree.iter_descendants("postorder"):
        parent_toleaf = node.up.get_farthest_leaf(topology_only=True)[1] + 1
        if node.is_leaf() == True:
            node_toleaf = 0
        else:
            node_toleaf = node.get_farthest_leaf(topology_only=True)[1] + 1
        node.dist = parent_toleaf - node_toleaf
    return tree
def write_csv(dist_matrix, names_, filename):
    outf = open(filename,"w")
    st = ""
    for name in names_:
        st+=(","+name)
    outf.write(st+"\n") 
    for i in range(len(dist_matrix)):
        st=names_[i]
        for val in dist_matrix[i]:
            st+=(","+str(val))
        outf.write(st+"\n")
    outf.close()

def NJ(dist_matrix, leaf_name, out_name):
    dist_matrix = dist_matrix.astype(int)
    csv_name2 = out_name + '/dnd.csv'
    write_csv(dist_matrix=dist_matrix, names_= leaf_name, filename = csv_name2)
    pdm = dnd.PhylogeneticDistanceMatrix.from_csv(src=open(csv_name2),delimiter=",")
    nj_tree = pdm.nj_tree()
    nj_tree = str(nj_tree) + ';'
    nj_tree = Tree(nj_tree, format = 5)
    return nj_tree
    
    
