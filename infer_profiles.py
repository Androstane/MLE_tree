import numpy as np
from scipy.linalg import expm
import scipy 
import ete3
from ete3 import Tree
import numpy as np
import scipy.special
import pandas as pd
from scipy.spatial.distance import pdist, squareform
import dendropy as dnd
from parse import parse

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
def add_edgeL(tree):
    for node in tree.iter_descendants("postorder"):
        parent_toleaf = node.up.get_farthest_leaf(topology_only=True)[1] + 1
        if node.is_leaf() == True:
            node_toleaf = 0
        else:
            node_toleaf = node.get_farthest_leaf(topology_only=True)[1] + 1
        node.dist = parent_toleaf - node_toleaf
    return tree

def recurse_profile(tree_node, site_i, n_states): 
    #tree.get_tree_root().dist = orig_time
    if tree_node.is_leaf():
        leaf_state = tree_node.sequence[site_i]
        for i in range(0, n_states):
            tree_node.C[i] = leaf_state
            tree_node.L[i] = edge_dep_tp(leaf_state, i, tree_node.dist)
    else:
        left_child, right_child = tree_node.get_children()
        left_branch = np.float(left_child.dist)
        right_branch = np.float(right_child.dist)
        recurse_profile(left_child, site_i, n_states)
        recurse_profile(right_child, site_i, n_states)
        for parent_node_states in range(n_states):
            j = 0
            prob = 0
            for child_node_states in range(n_states):
                #print(tree_node.dist)
                temp = edge_dep_tp(child_node_states, parent_node_states, tree_node.dist)*left_child.L[child_node_states]*right_child.L[child_node_states]
                if temp > prob: 
                    prob = temp
                    j = child_node_states
            tree_node.L[parent_node_states] = prob
            tree_node.C[parent_node_states] = j


def recurse_assign(tree_node, site_i, n_states, orig_time):
    root = tree_node.get_tree_root()
    left_child, right_child = root.get_children()
    root.prof = 0
    l = 0
    for i in range(0, n_states):
        tp = edge_dep_tp(i, 2, orig_time)
        temp = edge_dep_tp(i, 2, orig_time)*left_child.L[i]*right_child.L[i]
        if temp > l:
            l = temp
            root.prof = i
    if tree_node.is_leaf():
        return 0
    
    else:
        left_child, right_child = tree_node.get_children()
        #print(left_child, right_child)
        #print("prof:", tree_node.prof)
        left_child.prof = left_child.C[int(tree_node.prof)]
        #print(left_child.C)
        right_child.prof = right_child.C[int(tree_node.prof)]
        #print(right_child.C)
        recurse_assign(left_child, site_i, n_states, orig_time)
        recurse_assign(right_child, site_i, n_states, orig_time)
         
def initialize_tree(tree, profile, n_states):
    site_count = len(list(profile.values())[0])
    for node in tree.traverse():
        # initialize a vector of partial likelihoods that we can reuse for each site
        node.L = np.zeros(n_states)
        node.C = np.zeros(n_states)
        node.prof = 1e10
        if node.is_root():
            node.sequence = []
            node.prof = 2
        if node.is_leaf():
            node.sequence = profile[node.name]
        else:
            node.sequence = []
    return tree, site_count
    
    
def infer_profiles(tree, profile, n_states):
    tree, site_count = initialize_tree(tree, profile, n_states)
    for node in tree.traverse():
        node.height = node.get_farthest_node(topology_only=False)[1]
    root = tree.get_tree_root()
    left_child, right_child = root.get_children()
    elder_child_height = max(left_child.height, right_child.height)
    orig_time = elder_child_height + np.random.exponential(elder_child_height)
    for site_i in range(site_count):
        #print("current site", site_i)
        recurse_profile(tree, site_i, n_states)
        recurse_assign(tree, site_i, n_states, orig_time)
        for node in tree.traverse():
             if not node.is_leaf():
                    node.sequence.append(node.prof)

def read_newick(newick_path):
    newick_file = open(newick_path)
    newick = newick_file.read().strip()
    newick = newick + ';'
    newick_file.close()
    tree = ete3.Tree(newick, format = 8)
    return tree

if __name__ == "__main__":
    orig_stdout = sys.stdout
    ap = argparse.ArgumentParser()
    ap.add_argument("-in","--path to input",required=True, help="Path to the input folder containing the copy number profiles ")
    args = vars(ap.parse_args())
    if args['path to input']!=None:
        path = args['path to input']
   	f = open(path + '/infer_prof_out.txt', 'w+')
    sys.stdout = f
	out, dist_matrix, leaf_name, profile = parse(path)
	new_profile = {}
	for key in profile:
    	new_key = key.replace('leaf', '')
    	new_profile[new_key] = profile[key]
	tree = read_newick(path + '/gt.newick')
	tree = (tree & 1).detach()
	tree = add_edgeL(tree)
	n_states = np.max(list(new_profile.values())) + 1
	print("start infer profiles")
	infer_profiles(tree, new_profile, n_states)
	df = pd.DataFrame(columns=profile.keys())
	temp_dic = {}
	for node in tree.traverse():
    	[node.name] = node.sequence
	df = pd.DataFrame({key: pd.Series(value) for key, value in temp_dic.items()})
	df.to_csv(path + '/inferred_profile.csv', encoding='utf-8',index = False)
	sys.stdout = orig_stdout
	f.close()

