import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import numpy as np
import anndata
import random
from conditional_independence import hsic_test 
from copy import deepcopy
from itertools import combinations
from causallearn.utils.cit import CIT
import jpype.imports
import argparse
import pydotplus
import os
import igraph
import pydot
from causallearn.search.ConstraintBased.FCI import fci
from causallearn.utils.cit import kci
from causallearn.graph.Node import Node
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge


def _node_hash(self):
    return hash(self.get_name())

def _node_eq(self, other):
    if not isinstance(other, Node):
        return False
    return self.get_name() == other.get_name()

Node.__hash__ = _node_hash
Node.__eq__   = _node_eq

def count_precision_recall_f1(tp, fp, fn):
    # Precision
    if tp + fp == 0:
        precision = None
    else:
        precision = float(tp) / (tp + fp)

    # Recall
    if tp + fn == 0:
        recall = None
    else:
        recall = float(tp) / (tp + fn)

    # F1 score
    if precision is None or recall is None:
        f1 = None
    elif precision == 0 or recall == 0:
        f1 = 0.0
    else:
        f1 = float(2 * precision * recall) / (precision + recall)
    return precision, recall, f1

def count_dag_accuracy(B_bin_true, B_bin_est):
    d = B_bin_true.shape[0]
    # linear index of nonzeros
    pred = np.flatnonzero(B_bin_est)
    cond = np.flatnonzero(B_bin_true)
    cond_reversed = np.flatnonzero(B_bin_true.T)
    cond_skeleton = np.concatenate([cond, cond_reversed])
    # true pos
    true_pos = np.intersect1d(pred, cond, assume_unique=True)
    # false pos
    false_pos = np.setdiff1d(pred, cond_skeleton, assume_unique=True)
    # reverse
    extra = np.setdiff1d(pred, cond, assume_unique=True)
    reverse = np.intersect1d(extra, cond_reversed, assume_unique=True)
    # compute ratio
    pred_size = len(pred)
    cond_neg_size = 0.5 * d * (d - 1) - len(cond)
    if pred_size == 0:
        fdr = None
    else:
        fdr = float(len(reverse) + len(false_pos)) / pred_size
    if len(cond) == 0:
        tpr = None
    else:
        tpr = float(len(true_pos)) / len(cond)
    if cond_neg_size == 0:
        fpr = None
    else:
        fpr = float(len(reverse) + len(false_pos)) / cond_neg_size
    # structural hamming distance
    pred_lower = np.flatnonzero(np.tril(B_bin_est + B_bin_est.T))
    cond_lower = np.flatnonzero(np.tril(B_bin_true + B_bin_true.T))
    extra_lower = np.setdiff1d(pred_lower, cond_lower, assume_unique=True)
    missing_lower = np.setdiff1d(cond_lower, pred_lower, assume_unique=True)
    shd = len(extra_lower) + len(missing_lower) + len(reverse)
    # false neg
    false_neg = np.setdiff1d(cond, true_pos, assume_unique=True)
    precision, recall, f1 = count_precision_recall_f1(tp=len(true_pos),
                                                      fp=len(reverse) + len(false_pos),
                                                      fn=len(false_neg))
    # return {'fdr': fdr, 'tpr': tpr, 'fpr': fpr, 'shd': shd, 'nnz': pred_size, 
    #         'precision': precision, 'recall': recall, 'f1': f1}
    return {'f1': f1,  'precision': precision, 'recall': recall, 'shd': shd}

# def find_nodes_on_paths(matrix, i, j):
#     n = len(matrix)  # Number of nodes
#     result = set()   # To store all unique nodes on paths between i and j

#     def dfs(current_node, visited):
#         # If current node is j, add all visited nodes to the result
#         if current_node == j:
#             result.update(visited)
#             return

#         # Explore all neighbors of the current node
#         for neighbor in range(n):
#             if matrix[current_node][neighbor] > 0 and neighbor not in visited:
#                 dfs(neighbor, visited + [neighbor])  # Recursive DFS

#     dfs(i, [i])  # Start DFS from node i with i as the initial visited node
#     return result
def get_adjSet(i, G, n_node):
    adj = []
    for j in range(n_node):
        if G[i][j] == 1 or G[j][i] == 1:
            adj.append(j)
    return adj
def get_adj_ij(i, j, G, n_node):
    adj = []
    for k in range(n_node):
        if G[i][k] ==1 & G[k][j] == 1:
            adj.append(k)
    return adj
def fisher_z_test(i, j, K, sample, result):
    indep = True
    fisher_z_obj = CIT(sample, "kci")
    Pvalue = fisher_z_obj(i,j,K)
    result.append([f'{i}_{j}_{K}___{Pvalue}'])
    # print(f'{i}_{j}_{K}___{Pvalue}')
    alpha = 0.1
    if Pvalue >= alpha:
        indep = True
    else:
        indep = False
    return indep

def skeleton(n_node, sample):

    C = np.ones((n_node,n_node))

    S = []
    for i in range(n_node):
        S.append([])
        for j in range(n_node):
            S[i].append([])

    pairs = []
    for i in range(n_node):
        for j in range(n_node - i):
            if(i != (n_node - j - 1)):  
                pairs.append((i, (n_node - j - 1)))
            else:
                C[i, i] = 0
    CI_result = []
    l = -1    
    while 1:
        l = l + 1
        flag = True   
        for (i, j) in pairs:

            adj_set = get_adjSet(i, C, n_node)    
            if(C[i][j] == 1) & (len(adj_set) >= l):    
                flag =False   
                adj_set.remove(j)    

                combin_set = combinations(adj_set, l)    
                for K in combin_set:
                    if fisher_z_test(i, j, list(K), sample, CI_result):   
                        C[i][j] = 0
                        C[j][i] = 0

                        S[i][j] = list(K)
                        S[j][i] = list(K)    
                    else:
                        continue
            else:
                continue

        if flag:
            break

    return C, S, CI_result

def all_paths(matrix, start, end):
    res = []
    stack = [(start, [start])]
    n = len(matrix)
    while stack:
        u, path = stack.pop()
        if u == end:
            res.append(path)
            continue
        for v in range(n):
            if matrix[u][v] != 0 and v not in path: 
                stack.append((v, path + [v]))
    return res


from typing import List, Iterable, Tuple, Set

def _to_adj_lists(matrix):
    n = len(matrix)
    adj  = [[] for _ in range(n)]
    radj = [[] for _ in range(n)]
    for u in range(n):
        row = matrix[u]
        for v, w in enumerate(row):
            if w != 0:
                adj[u].append(v)
                radj[v].append(u)
    return adj, radj

def _dfs(start, adj):
    n = len(adj)
    seen = [False]*n
    stack = [start]
    seen[start] = True
    while stack:
        u = stack.pop()
        for v in adj[u]:
            if not seen[v]:
                seen[v] = True
                stack.append(v)
    return seen

def k_simple_paths(matrix, start, end, k=None, max_len=None) -> Iterable[List[int]]:
 
    adj, radj = _to_adj_lists(matrix)

    can_to_end = _dfs(end, radj)

    if start == end:
        yield [start]
        return

    n = len(adj)
    visited = [False]*n
    path = [start]
    visited[start] = True
    out_count = 0

    it_stack = [iter(adj[start])]

    def can_extend() -> bool:
        if max_len is None:
            return True
        return len(path) < max_len

    while it_stack:
        try:
            v = next(it_stack[-1])

            if visited[v]:
                continue
            if not can_to_end[v]:
                continue
            if not can_extend():
                continue

            path.append(v)
            visited[v] = True

            if v == end:
                yield list(path)
                out_count += 1
              
                visited[v] = False
                path.pop()
                if k is not None and out_count >= k:
                    return
            else:
                it_stack.append(iter(adj[v]))

        except StopIteration:
           
            it_stack.pop()
            if path:
                u = path.pop()
                visited[u] = False

def collect_paths(matrix, start, end, k=6, max_len=4, unique=True) -> List[List[int]]:
    paths = list(k_simple_paths(matrix, start, end, k=k, max_len=max_len))
    if unique:
        as_set: Set[Tuple[int, ...]] = set(map(tuple, paths))
        paths = [list(t) for t in as_set]
        paths.sort(key=lambda p: (len(p), p))
    return paths


def find_nodes_on_paths(matrix, i, j):
    paths = collect_paths(matrix, i, j)
    nodes = set()
    for p in paths:
        nodes.update(p)
    return nodes, paths 

# def has_path_length_2(adj_matrix, node_a, node_b):
#     """
#     Check if there exists a path of length 2 between node A and node B.

#     Parameters:
#     - adj_matrix: numpy array (Adjacency matrix of the graph)
#     - node_a: int (Index of node A)
#     - node_b: int (Index of node B)

#     Returns:
#     - True if a path of length 2 exists between A and B, else False
#     """
#     n = adj_matrix.shape[0]
#     middle_node = []
#     for c in range(n):  # Check all possible middle nodes C
#         if adj_matrix[node_a, c] == 1 and adj_matrix[c, node_b] == 1:
#             middle_node.append(c)
#             return True, middle_node  # Found a path of length 2
    
#     return False, middle_node



def has_path_length_2(adj_matrix, node_a, node_b):
    """
    Find all middle nodes C such that there is a path A -> C -> B.

    Parameters:
    - adj_matrix: numpy array (Adjacency matrix of the graph)
    - node_a: int (Index of node A)
    - node_b: int (Index of node B)

    Returns:
    - middle_nodes: list of int (All valid middle nodes; empty if none)
    """
    n = adj_matrix.shape[0]
    middle_nodes = []
    for c in range(n):
        if adj_matrix[node_a, c] != 0 and adj_matrix[c, node_b] != 0:
            middle_nodes.append(c)
    return middle_nodes

def get_adjSet(i, G, n_node):
    adj = []
    for j in range(n_node):
        if G[i][j] == 1 or G[j][i] == 1:
            adj.append(j)
    return adj


def neighbors_anydir(A, u):

    n = len(A)
    return [v for v in range(n) if A[u][v] != 0 or A[v][u] != 0]

def is_arrow_into(A, a, z):
    return A[a][z] > 0 

def find_paths_with_colliders(A, start, end):
    n = len(A)
    paths = []

    stack = [(start, [start])]
    while stack:
        u, path = stack.pop()
        if u == end:
            if len(path) >= 3:  
                paths.append(path[:])
            continue
        for v in neighbors_anydir(A, u):
            if v in path:
                continue
            stack.append((v, path + [v]))

    detailed_paths = []
    all_intermediate_nodes = set()
    all_colliders = set()

    for node_seq in paths:
     
        for i in range(1, len(node_seq) - 1):
            all_intermediate_nodes.add(node_seq[i])

      
        colliders = []
        for i in range(1, len(node_seq) - 1):
            x, z, y = node_seq[i - 1], node_seq[i], node_seq[i + 1]
            if is_arrow_into(A, x, z) and is_arrow_into(A, y, z):
                colliders.append(z)
                all_colliders.add(z)

     
        edge_weights = []
        for i in range(len(node_seq) - 1):
            a, b = node_seq[i], node_seq[i + 1]
            w = A[a][b]
            if w == 0:
                raise ValueError(f"Invalid path step {a}->{b}: A[a][b] == 0")
            edge_weights.append(w)

        detailed_paths.append({
            'node_seq': node_seq,
            'edge_weights': edge_weights,
            'colliders_on_path': colliders
        })

    return detailed_paths, sorted(all_intermediate_nodes), sorted(all_colliders)


def causal_validate_path( path, start_set, end_set, rules):
    """
    Validate a path based on the given rules and return specific indices of weight pairs.

    Parameters:
    - path: List of weights representing the path (e.g., [1, 2, 5, 6, 2])
    - start_set: Set of valid start weights (e.g., {1, 3})
    - end_set: Set of valid end weights (e.g., {2, 4})
    - rules: Dict of valid transitions for weights apart from the first and last one
             (e.g., {-2: {-2, 4, 6}, 2: {2, 6}, 4: {2, 6}, 6: {-2, 2, 4, 6}})

    Returns:
    - is_valid: True if the path satisfies all constraints, False otherwise
    - specific_indices: List of tuples (i, i+1) where the first weight is in (6, -2, 3)
                        and the next weight is in (4, -2)
    """
    # Initialize to store indices for the specific condition
    specific_indices = []

    # Check if the first weight is valid
    if path[0] not in start_set:
        return False, specific_indices

    # Check if the last weight is valid
    if path[-1] not in end_set:
        return False, specific_indices
    
    if len(path == 2):
        current_weight = path[0]
        next_weight = path[1]
        if current_weight ==3 and next_weight == 4:
            specific_indices.append((0,1))
            return True, specific_indices
        else:
            return False, specific_indices
    else:
        # Validate the second weight based on the first weight
        if path[0] == 1 and path[1] not in {2, 6}:
            return False, specific_indices
        if path[0] == 3 and path[1] not in {-2, 2, 4, 6}:
            return False, specific_indices

        # Validate the second-to-last weight based on the last weight
        if path[-1] == 2 and path[-2] not in {-2, 2, 4, 6}:
            return False, specific_indices
        if path[-1] == 4 and path[-2] not in {-2, 6}:
            return False, specific_indices

        # Validate all other weights using the rules
        for i in range(len(path) - 1):
            current_weight = path[i]
            next_weight = path[i + 1]

            # Check if the weights follow the general rules
            if i > 0 and i < len(path)-2:
                if current_weight in rules and next_weight not in rules[current_weight]:
                    return False, specific_indices

            # Check for the specific condition (current in {6, -2, 3}, next in {4, -2})
            if current_weight in {6, -2, 3} and next_weight in {4, -2}:
                specific_indices.append((i, i + 1))
                break

        # If all checks passed, return True and the specific indices
        return True, specific_indices

def selection_validate_path( path, start_set, end_set, rules):
    """
    Validate a path based on the given rules and return specific indices of weight pairs.

    Parameters:
    - path: List of weights representing the path (e.g., [1, 2, 5, 6, 2])
    - start_set: Set of valid start weights (e.g., {1, 3})
    - end_set: Set of valid end weights (e.g., {2, 4})
    - rules: Dict of valid transitions for weights apart from the first and last one
             (e.g., {-2: {-2, 4, 6}, 2: {2, 6}, 4: {2, 6}, 6: {-2, 2, 4, 6}})

    Returns:
    - is_valid: True if the path satisfies all constraints, False otherwise
    - specific_indices: List of tuples (i, i+1) where the first weight is in (6, -2, 3)
                        and the next weight is in (4, -2)
    """
    # Initialize to store indices for the specific condition
    effect_specific_indices = []
    cause_specific_indices = []

    # Check if the first weight is valid
    if path[0] not in start_set:
        return False, effect_specific_indices, cause_specific_indices

    # Check if the last weight is valid
    if path[-1] not in end_set:
        return False, effect_specific_indices, cause_specific_indices
    
    if len(path == 2):
        current_weight = path[0]
        next_weight = path[1]
        if current_weight ==1 and next_weight == -3:
            cause_specific_indices.append((0,1))
            return True, effect_specific_indices, cause_specific_indices
        elif current_weight ==3 and next_weight ==-1:
            effect_specific_indices.append((0,1))
            return True, effect_specific_indices, cause_specific_indices
    else:
        # Validate the second weight based on the first weight
        if path[0] == 1 and path[1] not in {2, 6}:
            return False, effect_specific_indices, cause_specific_indices
        if path[0] == 3 and path[1] not in {-2, 2, 4, 6}:
            return False, effect_specific_indices, cause_specific_indices

        # Validate the second-to-last weight based on the last weigh
        if path[-1] == -1 and path[-2] not in {-2, 6}:
            return False, effect_specific_indices, cause_specific_indices
        if path[-1] == -3 and path[-2] not in {-2,2,4, 6}:
            return False, effect_specific_indices, cause_specific_indices

        # Validate all other weights using the rules
        for i in range(len(path) - 1):
            current_weight = path[i]
            next_weight = path[i + 1]

            # Check if the weights follow the general rules
            if i > 0 and i < len(path)-2:
                if current_weight in rules and next_weight not in rules[current_weight]:
                    return False, effect_specific_indices, cause_specific_indices

            # Check for the specific condition (current in {6, -2, 3}, next in {4, -2})
            if current_weight in {6, -2, 3} and next_weight in {4, -2, -1}:
                effect_specific_indices.append((i, i + 1))
                break
            if current_weight in {1, 2, 4} and next_weight in {2,-3,6}:
                cause_specific_indices.append((i,i+1))
                break

        # If all checks passed, return True and the specific indices
        return True, effect_specific_indices, cause_specific_indices

def correct_inducing_path(matrix, data_final):
    causal_s = {1,3}
    causal_e = {2,4}
    selection_s = {1,3}
    selection_e = {-1,-3}
    rules = {-2: {-2,2, 4, 6}, 2: {2, 6}, 4: {2, 6}, 6: {-2, 2, 4, 6}}
    n_nodes = matrix.shape[0]
    causal = []
    selection = []
    for i in range(n_nodes):
        for j in range(i+1, n_nodes):
            if matrix[i][j] == 1 and matrix[j][i]==-1:
                causal.append([i,j])
            elif matrix[j][i] ==1 and matrix[i][j] ==-1:
                causal.append([j,i])
            elif matrix[i][j] == 5 and matrix[j][i]==5:
                selection.append([i,j])
    for k in range(len(causal)):
        found = False
        paths, all_nodes, all_colliders = find_paths_with_colliders(matrix, causal[k][0], causal[k][1])
        if len(paths) != 0:
            for m, (node_seq, mark_seq, collider_on_path) in enumerate(paths):
                if len(collider_on_path) != 0:
                    continue
                inducing_path, indices = causal_validate_path( mark_seq,causal_s, causal_e, rules)
                if inducing_path:
                    # import pdb
                    # pdb.set_trace()
                    if len(indices) != 0:
                        node_index = node_seq[indices[0][1]]
                    else:
                        break
                    cause = causal[k][0]
                    i = node_index
                    j = causal[k][1]
                    c_set = all_nodes
                    if c_set is not None:
                        for n in node_seq:
                            if n in c_set:
                                c_set.remove(n)
                        for m in all_colliders:
                            if m in c_set:
                                c_set.remove(m)
                        assert cause not in c_set
                        assert j not in c_set
                    c_set = list(c_set)
                    data_i = data_final[f'per_{i}'][:,[i,j,-1]]
                    data_j = data_final[f'per_{j}'][:,[i,j,-1]]
                    data_p_i = np.concatenate((data_i, data_final[f'per_{i}'][:,c_set]), axis=1)
                    # CIT_obj = CIT(data_p_j, "kci", kernelX='Polynomial', kernelY='Polynomial')
                    g_adj = [i for i in range(3,data_p_i.shape[1])]
                    CIT_obi = CIT(data_p_i,"kci", kernelX='Polynomial', kernelY='Polynomial')
                    Upi_value = CIT_obi(1,2, g_adj)
                    found = True
                    if Upi_value > 0.05:
                        print('success')
                        matrix[causal[k][0]][causal[k][1]] = 7
                        matrix[causal[k][1]][causal[k][0]] = -7
                if found:
                    break
    for k in range(len(selection)):
        found = False
        paths, all_nodes, all_colliders = find_paths_with_colliders(matrix, selection[k][0], selection[k][1])
        if len(paths) != 0:
            for m, (node_seq, mark_seq, collider_on_path) in enumerate(paths):
                if len(collider_on_path) != 0:
                    continue
                inducing_path, effect_indices, cause_indices = selection_validate_path(mark_seq, selection_s, selection_e, rules)
                if inducing_path:
                    if len(effect_indices) != 0:
                        node_index = effect_indices[0][1]
                        selection_1 = selection[k][0]
                        i = node_index
                        j = selection[k][1]
                    elif len(cause_indices) != 0:
                        selection_1 = selection[k][1]
                        node_index = cause_indices[0][1]
                        i = node_index
                        j = selection[k][0]
                    else:
                        break
                    c_set = all_nodes
                    if c_set is not None:
                        for n in node_seq:
                            if n in c_set:
                                c_set.remove(n)
                        for m in all_colliders:
                            if m in c_set:
                                c_set.remove(m)
                        assert selection_1 not in c_set
                        assert j not in c_set
                    c_set = list(c_set)
                    data_i = data_final[f'per_{i}'][:,[i,j,-1]]
                    data_j = data_final[f'per_{j}'][:,[i,j,-1]]
                    data_p_i = np.concatenate((data_i, data_final[f'per_{i}'][:,c_set]), axis=1)
                    # CIT_obj = CIT(data_p_j, "kci", kernelX='Polynomial', kernelY='Polynomial')
                    g_adj = [i for i in range(3,data_p_i.shape[1])]
                    CIT_obi = CIT(data_p_i,"kci", kernelX='Polynomial', kernelY='Polynomial')
                    Upi_value = CIT_obi(1,2, g_adj)
                    found = True
                    if Upi_value > 0.05:
                        print('success')
                        matrix[selection[k][0]][selection[k][1]] = 8
                        # if [selection[k][0], selection[k][1]] in result['selection']:
                        #     result['selection'].remove([selection[k][0], selection[k][1]])
                        # if [selection[k][1], selection[k][0]] in result['selection']:
                        #     result['selection'].remove(selection[k][1], selection[k][0])
                if found:
                    break
    return matrix

def dag_for_bk(dag):
    for i in range(dag.shape[0]):
        for j in range(dag.shape[0]):
            if dag[i][j] < 0:
                dag[i][j] = 0
            if dag[i][j] == 2:
                dag[i][j] = 1
            if dag[i][j] == 3:
                dag[i][j] = 0
            if dag[i][j] == 4:
                dag[i][j] = 0
            if dag[i][j] == 5:
                dag[i][j] = 0
            if dag[i][j] == 6:
                dag[i][j] = 0
            if dag[i][j] == 7:
                dag[i][j] = 0
            if dag[i][j] == 8:
                dag[i][j] = 0
            if dag[i][j] == 9:
                dag[i][j] = 0

    return dag

def dag_transformation(dag):
    for i in range(dag.shape[0]):
        for j in range(dag.shape[0]):
            if dag[i][j] < 0:
                dag[i][j] = 0
            if dag[i][j] == 2:
                dag[i][j] = 1
            if dag[i][j] == 3:
                dag[i][j] = 1
            if dag[i][j] == 4:
                dag[i][j] = 0
            if dag[i][j] == 5:
                dag[i][j] = 0
            if dag[i][j] == 6:
                dag[i][j] = 1
            if dag[i][j] == 7:
                dag[i][j] = 0
            if dag[i][j] == 8:
                dag[i][j] = 0
            if dag[i][j] == 9:
                dag[i][j] = 1
            if dag[i][j] == 10:
                dag[i][j] = 1
    return dag

def draw_graph_from_matrix(matrix, labels=None, filename="graph.png"):
    n = len(matrix)
    if labels is None:
        labels = [str(i) for i in range(n)]

    graph = pydot.Dot(graph_type="digraph")

    # Add nodes
    for i in range(n):
        graph.add_node(pydot.Node(labels[i]))

    # Add edges with custom arrowheads
    for i in range(n):
        for j in range(i+1,n):
            if matrix[i][j] != 0:  # if there's some connection
                # Example rule: if matrix[i][j]=1 and matrix[j][i]=2, then i→j
                if matrix[i][j] == 1 and matrix[j][i] == -1:
                    edge = pydot.Edge(labels[i], labels[j], arrowhead="normal")
                    graph.add_edge(edge)
                elif matrix[i][j] == -1 and matrix[j][i] == 1:
                    edge = pydot.Edge(labels[j], labels[i], arrowhead="normal")
                    graph.add_edge(edge)
                elif matrix[i][j] == 2 and matrix[j][i] == -2:
                    edge = pydot.Edge(labels[i], labels[j], arrowtail = 'obox', arrowhead="normal", dir ='both')
                    graph.add_edge(edge)
                elif matrix[i][j] == -2 and matrix[j][i] == 2:
                    edge = pydot.Edge(labels[j], labels[i], arrowtail = 'obox', arrowhead="normal", dir ='both')
                    graph.add_edge(edge)
                elif matrix[i][j] == 3 and matrix[j][i] == -3:
                    edge = pydot.Edge(labels[i], labels[j],arrowhead="obox")
                    graph.add_edge(edge)
                elif matrix[i][j] == -3 and matrix[j][i] == 3:
                    edge = pydot.Edge(labels[j], labels[i],arrowhead="obox")
                    graph.add_edge(edge)
                elif matrix[i][j] == 4 and matrix[j][i] == 4:
                    edge = pydot.Edge(labels[i], labels[j], arrowtail = 'normal', arrowhead="normal", dir ='both')
                    graph.add_edge(edge)
                elif matrix[i][j] == 5 and matrix[j][i] == 5:
                    edge = pydot.Edge(labels[i], labels[j], dir="none")
                    graph.add_edge(edge)
                elif matrix[i][j] == 6 and matrix[j][i] == 6:
                    edge = pydot.Edge(labels[i], labels[j], arrowtail = 'obox', arrowhead="obox", dir ='both')
                    graph.add_edge(edge)
                elif matrix[i][j] == 7 and matrix[j][i] == -7:
                    edge = pydot.Edge(labels[i], labels[j], label = '▸', arrowhead="normal", labelfloat=True)
                    graph.add_edge(edge)
                elif matrix[i][j] == -7 and matrix[j][i] == 7:
                    edge = pydot.Edge(labels[j], labels[i], label = '▸', arrowhead="normal", labelfloat=True)
                    graph.add_edge(edge)
                elif matrix[i][j] == 8 or matrix[j][i] == 8:
                    edge = pydot.Edge(labels[i], labels[j], label = '▸', dir="none", labelfloat=True)
                    graph.add_edge(edge)
                elif matrix[i][j] == 10 and matrix[j][i] == 9:
                    edge = pydot.Edge(labels[i], labels[j], arrowtail = 'obox', arrowhead="odot", dir ='both')
                    graph.add_edge(edge)
                elif matrix[i][j] == 9 and matrix[j][i] == 10:
                    edge = pydot.Edge(labels[j], labels[i], arrowtail = 'obox', arrowhead="odot", dir ='both')
                    graph.add_edge(edge)
                elif matrix[i][j] == 10 and matrix[j][i] == 1:
                    edge = pydot.Edge(labels[i], labels[j], arrowtail = 'normal', arrowhead="odot",dir ='both')
                    graph.add_edge(edge)
                elif matrix[i][j] == 1 and matrix[j][i] == 10:
                    edge = pydot.Edge(labels[j], labels[i], arrowtail = 'normal', arrowhead="odot", dir ='both')
                    graph.add_edge(edge)
                elif matrix[i][j] == 10 and matrix[j][i] == -1:
                    edge = pydot.Edge(labels[i], labels[j], arrowhead="odot")
                    graph.add_edge(edge)
                elif matrix[i][j] == -1 and matrix[j][i] == 10:
                    edge = pydot.Edge(labels[j], labels[i], arrowhead="odot")
                    graph.add_edge(edge)
                elif matrix[i][j] == 10 and matrix[j][i] == 10:
                    edge = pydot.Edge(labels[j], labels[i], arrowtail="odot", arrowhead="odot", dir ='both')
                    graph.add_edge(edge)
                elif matrix[i][j] == 1 and matrix[j][i] == 1:
                    edge = pydot.Edge(labels[j], labels[i], arrowtail="normal", arrowhead="normal", dir ='both')
                    graph.add_edge(edge)
                elif matrix[i][j] == -1 and matrix[j][i] == -1:
                    edge = pydot.Edge(labels[i], labels[j], dir="none")
                    graph.add_edge(edge)
                # You can extend more rules here (e.g., undirected, bidirected, etc.)

    # Save graph
    graph.write_png(filename)
    print(f"Graph saved to {filename}")


def refine_L_C(dag, data, L_C, selection):
    n=dag.shape[0]
    nodes_with_2 = set()
    for r in range(n):
        for c in range(n):
            if dag[r][c] == 2 and dag[c][r] == -2:
                nodes_with_2.add((r,c))
                # nodes_with_2.add(c)

    special_edge_vals: Set[int] = {5, 3, -3, 6}

    # --- Step 2-4: for each candidate i, check outgoing edges and required keys ---
    for item in sorted(nodes_with_2):
        i = item[0]
        # check whether i has any outgoing edge to other nodes with value in {5,3,-3}
        has_special_outgoing = any(
            (j != i) and (dag[i][j] in special_edge_vals)
            for j in range(n)
        )
        if not has_special_outgoing:
            continue

        k1 = f"per_{i}_h1"
        k2 = f"per_{i}_h2"
        if k1 not in data:
            continue
        if k2 not in data:
            continue
        d1 = data[k1]
        d2 = data[k2]
        per1 = np.concatenate((d1, np.zeros((d1.shape[0],1))), axis=1)
        per2 = np.concatenate((d2, np.ones((d2.shape[0],1))), axis=1)
        data_fin = np.concatenate((per1,per2), axis=0)
        data_i = data_fin[:,[i,item[1],-1]]
        if f"{item[0]}_{item[1]}" in L_C:
            given_set = L_C[f"{item[0]}_{item[1]}"]
        else:
            continue
        data_p_i = np.concatenate((data_i, data_fin[:,given_set]))
        g_adj = [k for k in range(3,data_p_i.shape[1])]
        #############kci ####################
        CIT_obj = CIT(data_p_i, "kci", kernelX='Polynomial', kernelY='Polynomial')
        Upi_value = CIT_obj(0,2, g_adj)
        if Upi_value < 0.05:
            dag[item[0]][item[1]] = 4
            dag[item[1]][item[0]] = 4
    
    return dag
        

        


def main(opt):
    sample = []
    acc_selection = []
    acc_latent = []
    acc_dag = []
    f1_dag = []
    recall_dag = []
    shd_dag = []
    L_C = {}
    count = 0
    interven = opt.intervention
    d = opt.num_nodes
    sample_size_select = opt.num_samples
    noise = opt.noise
    for i in range(3):
        save = i
        file_path = f'./{interven}_large_new/v_{d}/{sample_size_select}/sample_{i}'
        if not os.path.exists(file_path):
            print('path error')
            continue
        data = np.load(os.path.join(file_path, f'sample_{interven}.npz'), allow_pickle=True)
        obs = data['obs']
        interven_list = data['interven']
       
        n_sample, n_node = obs.shape[0],obs.shape[1]
        node_name = [str(i) for i in range(n_node)]
        ske,sep_set, CI_ske_result = skeleton(n_node, obs)
      
        # bad_mark = False
        # for pair in data['selection']:
        #     if ske[pair[0]][pair[1]] == 0:
        #         bad_mark = True

        # for pair in data['latent']:
        #     for i in range(len(pair)):
        #         if i < len(pair)-1:
        #             if ske[pair[i]][pair[i+1]] ==0:
        #                 bad_mark = True
        # if bad_mark:
        #     continue
        # else:
        #     sample.append(i)
        
        dag = deepcopy(ske)
        dag = dag.astype(int)
        data_final = {}
        for i in interven_list:
            per = data[f'per_{i}']
            per = np.concatenate((per, np.ones((per.shape[0],1))), axis=1)
            if 'obs' not in data_final:
                # obs_zero = np.sum(data_ctr_all ==0, axis=1)/data_ctr_all.shape[1]
                # data_ctr_all = data_ctr_all[obs_zero < 0.6]
                data_final['obs'] = obs

            data_ctr = np.concatenate((obs, np.zeros((n_sample,1))), axis=1)
            data_final[f'per_{i}'] = np.concatenate((data_ctr, per), axis= 0)
            # print(data_final[f'per_{i}'].shape)
        rows, cols = ske.shape
        threshold = 0.05
        s_indicator = np.zeros([rows, cols])
        s_without_cause = []
        correct_set = []
        result = {}
        CI_result = {}
        condition_set = {}
        result['latent'] = []
        result['selection'] = []
        count =0
        for m in range(len(interven_list)):
            for n in range(m+1,len(interven_list)):
                i = interven_list[m]
                j = interven_list[n]
                if dag[i][j] == 1 and dag[j][i] == 1:
                    correct = False
                    data_i = data_final[f'per_{i}'][:,[i,j,-1]]
                    data_j = data_final[f'per_{j}'][:,[i,j,-1]]
                    ########### KCI ###########################
                    CIT_obj = CIT(data_j, "kci", kernelX='Polynomial', kernelY='Polynomial')
                    Upj_value = CIT_obj(0,2,set([]))
                    Cpj_value = CIT_obj(0,2,set([1]))
                    CIT_obi = CIT(data_i, "kci", kernelX='Polynomial', kernelY='Polynomial')
                    Upi_value = CIT_obi(1,2, set([]))
                    Cpi_value = CIT_obi(1,2, set([0]))
                    #################################################
                    
                    ##################HSIC###########################
                    
                    # print(f'{i}-{j} {Upj_value}, {Cpj_value},{Upi_value}, {Cpi_value}')
                    
                    if (Upj_value > threshold) & (Cpj_value < threshold) & (Upi_value < threshold) & (Cpi_value > threshold):
                        dag[j][i] = -1
                        # result['direct_cause'][f'{i}-{j}'] = Upj_value
                        # result[f'{i}-{j}'] = 'cause'
                    elif (Upi_value > threshold) & (Cpi_value < threshold) & (Upj_value < threshold) & (Cpj_value > threshold):
                        dag[i][j] = -1
                        # result['direct_cause'][f'{j}-{i}'] = Upi_value
                    elif (Upj_value < threshold) & (Cpj_value > threshold) & (Upi_value < threshold) & (Cpi_value > threshold):
                        dag[j][i] = 5
                        dag[i][j] = 5
                        result['selection'].append([i,j])
                    elif (Upj_value < threshold) & (Cpj_value < threshold) & (Upi_value < threshold) & (Cpi_value > threshold):
                        dag[j][i] = -3
                        dag[i][j] = 3
                        correct = True
                        correct_set.append([i,j])
                        result['selection'].append([i,j])
                        condition_set[f'{i}-{j}'] = 'S_C'
                    elif (Upi_value < threshold) & (Cpi_value < threshold) & (Upj_value < threshold) & (Cpj_value > threshold):
                        dag[i][j] = -3
                        dag[j][i] = 3
                        correct = True
                        correct_set.append([j,i])
                        result['selection'].append([j,i])
                        condition_set[f'{j}-{i}'] = 'S_C'
                    elif (Upj_value > threshold) & (Cpj_value < threshold) & (Upi_value < threshold) & (Cpi_value < threshold):
                        dag[j][i] = -2
                        dag[i][j] = 2
                        L_C[f'{i}_{j}'] = []
                        correct = True
                        correct_set.append([i,j])
                        condition_set[f'{i}-{j}'] = 'L_C'
                        result['latent'].append([i,j])
                    elif (Upi_value > threshold) & (Cpi_value < threshold) & (Upj_value < threshold) & (Cpj_value < threshold):
                        dag[i][j] = -2
                        dag[j][i] = 2
                        L_C[f'{j}_{i}'] = []
                        correct = True
                        correct_set.append([j,i])
                        condition_set[f'{j}-{i}'] = 'L_C'
                        result['latent'].append([j,i])
                    elif (Upj_value > threshold) & (Cpj_value < threshold) & (Upi_value > threshold) & (Cpi_value < threshold):
                        dag[j][i] = 4
                        dag[i][j] = 4
                        result['latent'].append([i,j])
                    elif (Upj_value > threshold) & (Cpj_value > threshold) & (Upi_value > threshold) & (Cpi_value > threshold):
                        dag[j][i] = 0
                        dag[i][j] = 0
                    elif (Upj_value < threshold) & (Cpj_value < threshold) & (Upi_value < threshold) & (Cpi_value < threshold):
                        dag[i][j] = 6
                        dag[j][i] = 6
                        correct_set.append([i,j])
                        condition_set[f'{i}-{j}'] ='F_D'
                    else:
                        dag[i][j] = 10
                        dag[j][i] = 10
                        correct = True
                        correct_set.append([i,j])
                        condition_set[f'{i}-{j}'] ='F_D'
                    
                    count +=1
                    
        print(count)

        for ind, pair in enumerate(correct_set):
            i,j = pair[0], pair[1]
            # c_set = given_set(i,j,dag)
            # c_set = set(get_adjSet(i, dag, rows) + get_adjSet(j, dag, rows))
            c_set,_ = find_nodes_on_paths(dag,i,j)
            if c_set is not None:
                if i in c_set:
                    c_set.remove(i)
                if j in c_set:
                    c_set.remove(j)
                assert i not in c_set
                assert j not in c_set
            c_set = list(c_set)
            middle_node = has_path_length_2(dag,i,j)
            if middle_node:
                # if len(middle_node)>0:
                found = False
                others = list(set(c_set) - set(middle_node))
                for k in range(0, len(middle_node)):
                    paris = list(combinations(middle_node,k))
                    for element in paris:
                        given_set = others+list(element)
                        data_i = data_final[f'per_{i}'][:,[i,j,-1]]
                        data_j = data_final[f'per_{j}'][:,[i,j,-1]]
                        data_p_i = np.concatenate((data_i, data_final[f'per_{i}'][:,given_set]), axis=1)
                        data_p_j = np.concatenate((data_j, data_final[f'per_{j}'][:,given_set]), axis=1)
                        g_adj = [k for k in range(3,data_p_i.shape[1])]
                        #############kci ####################
                        CIT_obj = CIT(data_p_j, "kci", kernelX='Polynomial', kernelY='Polynomial')
                        Upj_value = CIT_obj(0,2, g_adj)
                        Cpj_value = CIT_obj(0,2,g_adj+[1])
                        CIT_obi = CIT(data_p_i,"kci", kernelX='Polynomial', kernelY='Polynomial')
                        Upi_value = CIT_obi(1,2, g_adj)
                        Cpi_value = CIT_obi(1,2,g_adj+[0])
                        value = [Upj_value,Cpj_value,Upi_value,Cpi_value]
                        if condition_set[f'{i}-{j}'] != 'F_D':
                            countv = sum(1 for v in value if v < threshold)
                            if countv >2:
                                continue
                            else:
                                if condition_set[f'{i}-{j}'] == 'S_C':
                                    if (Upj_value < threshold) & (Cpj_value > threshold) & (Upi_value < threshold) & (Cpi_value > threshold):
                                        dag[i][j] = 5
                                        dag[j][i] = 5
                                        found = True
                                        break
                                elif condition_set[f'{i}-{j}'] == 'L_C':
                                    if (Upj_value > threshold) & (Cpj_value < threshold) & (Upi_value < threshold) & (Cpi_value > threshold):
                                        dag[j][i] = -1
                                        dag[i][j] = 1
                                        result['latent'].remove([i,j])
                                        found = True
                                        break
                                        # result['direct_cause'][f'{i}-{j}'] = Upj_value
                                    # elif (Upi_value > threshold) & (Cpi_value < threshold) & (Upj_value < threshold) & (Cpj_value > threshold):
                                    #     dag[i][j] = 0
                                    #     dag[j][i] = 1
                                    #     result['latent'].remove([j,i])
                                    #     found = True
                                    #     break
                                    elif (Upj_value > threshold) & (Cpj_value < threshold) & (Upi_value > threshold) & (Cpi_value < threshold):
                                        dag[j][i] = 4
                                        dag[i][j] = 4
                                        found = True
                                        break
                        else:
                            if (Upj_value > threshold) & (Cpj_value < threshold) & (Upi_value < threshold) & (Cpi_value > threshold):
                                dag[j][i] = -1
                                dag[i][j] = 1
                                found = True
                                break
                            elif (Upi_value > threshold) & (Cpi_value < threshold) & (Upj_value < threshold) & (Cpj_value > threshold):
                                dag[i][j] = -1
                                dag[j][i] = 1
                                found = True
                                break
                            elif (Upj_value < threshold) & (Cpj_value > threshold) & (Upi_value < threshold) & (Cpi_value > threshold):
                                dag[j][i] = 5
                                dag[i][j] = 5
                                result['selection'].append([i,j])
                                found = True
                                break
                            elif (Upj_value < threshold) & (Cpj_value < threshold) & (Upi_value < threshold) & (Cpi_value > threshold):
                                dag[j][i] = -3
                                dag[i][j] = 3
                                result['selection'].append([i,j])
                                found = True
                                break
                            elif (Upi_value < threshold) & (Cpi_value < threshold) & (Upj_value < threshold) & (Cpj_value > threshold):
                                dag[i][j] = -3
                                dag[j][i] = 3
                                result['selection'].append([j,i])
                                found = True
                                break
                            elif (Upj_value > threshold) & (Cpj_value < threshold) & (Upi_value < threshold) & (Cpi_value < threshold):
                                dag[j][i] = -2
                                dag[i][j] = 2
                                L_C[f'{i}_{j}'] = g_adj
                                result['latent'].append([i,j])
                                found = True
                                break
                            elif (Upi_value > threshold) & (Cpi_value < threshold) & (Upj_value < threshold) & (Cpj_value < threshold):
                                dag[i][j] = -2
                                dag[j][i] = 2
                                L_C[f'{j}_{i}'] = g_adj
                                result['latent'].append([i,j])
                                found = True
                                break
                            elif (Upj_value > threshold) & (Cpj_value < threshold) & (Upi_value > threshold) & (Cpi_value < threshold):
                                dag[j][i] = 4
                                dag[i][j] = 4
                                result['latent'].append([i,j]) 
                                found = True
                                break
                    if found:
                        break
            else:
                data_i = data_final[f'per_{i}'][:,[i,j,-1]]
                data_j = data_final[f'per_{j}'][:,[i,j,-1]]
                data_p_i = np.concatenate((data_i, data_final[f'per_{i}'][:,c_set]), axis=1)
                data_p_j = np.concatenate((data_j, data_final[f'per_{j}'][:,c_set]), axis=1)
                # CIT_obj = CIT(data_p_j, "kci", kernelX='Polynomial', kernelY='Polynomial')
                g_adj = [i for i in range(3,data_p_i.shape[1])]

                #################kci ######################
                CIT_obj = CIT(data_p_j, "kci", kernelX='Polynomial', kernelY='Polynomial')
                Upj_value = CIT_obj(0,2, g_adj)
                Cpj_value = CIT_obj(0,2,g_adj+[1])
                CIT_obi = CIT(data_p_i,"kci", kernelX='Polynomial', kernelY='Polynomial')
                Upi_value = CIT_obi(1,2, g_adj)
                Cpi_value = CIT_obi(1,2,g_adj+[0])
               
                value = [Upj_value,Cpj_value,Upi_value,Cpi_value]
                if condition_set[f'{i}-{j}'] != 'F_D':
                    countv = sum(1 for v in value if v < threshold)
                    if countv >2:
                        continue
                    else:
                        if condition_set[f'{i}-{j}'] == 'S_C':
                            if (Upj_value < threshold) & (Cpj_value > threshold) & (Upi_value < threshold) & (Cpi_value > threshold):
                                dag[i][j] = 5
                                dag[j][i] = 5
                        elif condition_set[f'{i}-{j}'] == 'L_C':
                            if (Upj_value > threshold) & (Cpj_value < threshold) & (Upi_value < threshold) & (Cpi_value > threshold):
                                dag[j][i] = -1
                                dag[i][j] = 1
                                result['latent'].remove([i,j])
                                # result['direct_cause'][f'{i}-{j}'] = Upj_value
                            # elif (Upi_value > threshold) & (Cpi_value < threshold) & (Upj_value < threshold) & (Cpj_value > threshold):
                            #     dag[i][j] = 0
                            #     dag[j][i] = 1
                            #     result['latent'].remove([j,i])
                            elif (Upj_value > threshold) & (Cpj_value < threshold) & (Upi_value > threshold) & (Cpi_value < threshold):
                                dag[j][i] = 4
                                dag[i][j] = 4
                else:
                    if (Upj_value > threshold) & (Cpj_value < threshold) & (Upi_value < threshold) & (Cpi_value > threshold):
                        dag[j][i] = -1
                        dag[i][j] = 1
                    elif (Upi_value > threshold) & (Cpi_value < threshold) & (Upj_value < threshold) & (Cpj_value > threshold):
                        dag[i][j] = -1
                        dag[j][i] = 1
                    elif (Upj_value < threshold) & (Cpj_value > threshold) & (Upi_value < threshold) & (Cpi_value > threshold):
                        dag[j][i] = 5
                        dag[i][j] = 5
                        result['selection'].append([i,j])
                    elif (Upj_value < threshold) & (Cpj_value < threshold) & (Upi_value < threshold) & (Cpi_value > threshold):
                        dag[j][i] = -3
                        dag[i][j] = 3
                        result['selection'].append([i,j])
                    elif (Upi_value < threshold) & (Cpi_value < threshold) & (Upj_value < threshold) & (Cpj_value > threshold):
                        dag[i][j] = -3
                        dag[j][i] = 3
                        result['selection'].append([j,i])
                    elif (Upj_value > threshold) & (Cpj_value < threshold) & (Upi_value < threshold) & (Cpi_value < threshold):
                        dag[j][i] = -2
                        dag[i][j] = 2
                        L_C[f'{i}_{j}'] = g_adj
                        result['latent'].append([i,j])
                    elif (Upi_value > threshold) & (Cpi_value < threshold) & (Upj_value < threshold) & (Cpj_value < threshold):
                        dag[i][j] = -2
                        dag[j][i] = 2
                        L_C[f'{j}_{i}'] = g_adj
                        result['latent'].append([i,j])
                    elif (Upj_value > threshold) & (Cpj_value < threshold) & (Upi_value > threshold) & (Cpi_value < threshold):
                        dag[j][i] = 4
                        dag[i][j] = 4
                        result['latent'].append([i,j]) 
                    elif (Upj_value > threshold) & (Cpj_value > threshold) & (Upi_value > threshold) & (Cpi_value > threshold):
                        dag[j][i] = 0
                        dag[i][j] = 0

        dag = refine_L_C(dag, data_final, L_C,result['selection'])
        
        dag = correct_inducing_path(dag, data_final)
        
        for i in range(n_node):
            for j in range(n_node):
                if dag[i][j] == 1 and dag[j][i] == 1:
                    if i in interven_list and j not in interven_list:
                        print(f'{i}_{j}')
                        data_i = data_final[f'per_{i}'][:,[i,j,-1]]
                        c_set,_ = find_nodes_on_paths(dag,i,j)
                        if c_set is not None:
                            if i in c_set:
                                c_set.remove(i)
                            if j in c_set:
                                c_set.remove(j)
                            assert i not in c_set
                            assert j not in c_set
                        c_set = list(c_set)
                        middle_node = has_path_length_2(dag,i,j)
                        if middle_node:
                            found = False
                            others = list(set(c_set) - set(middle_node))
                            for k in range(0, len(middle_node)):
                                pairs = [()]
                                pairs += list(combinations(middle_node,k))
                                for element in pairs:
                                    given_set = others+list(element)
                                    data_i = data_final[f'per_{i}'][:,[i,j,-1]]
                                    data_p_i = np.concatenate((data_i, data_final[f'per_{i}'][:,given_set]), axis=1)
                                    g_adj = [k for k in range(3,data_p_i.shape[1])]
                                    CIT_obi = CIT(data_p_i,"kci", kernelX='Polynomial', kernelY='Polynomial')
                                    Upi_value = CIT_obi(1,2, g_adj)
                                    Cpi_value = CIT_obi(1,2,g_adj+[0])
                                    print(Upi_value, Cpi_value)

                                    if Upi_value < threshold and Cpi_value < threshold:
                                        dag[i][j]=10
                                        dag[j][i]=9
                                        continue
                                    elif Upi_value > threshold and Cpi_value < threshold:
                                        dag[i][j]=10
                                        dag[j][i]=1
                                        found = True
                                        break
                                    elif Upi_value < threshold and Cpi_value > threshold:
                                        dag[i][j]=10
                                        dag[j][i]=-1
                                        found = True
                                        break
                                    else:
                                        if (Upi_value -threshold) > (Cpi_value- threshold) and (Cpi_value- threshold) < 0.05:
                                            dag[i][j]=10
                                            dag[j][i]=-1
                                        elif (Cpi_value -threshold) > (Upi_value- threshold) and (Upi_value- threshold)< 0.05:
                                            dag[i][j]=10
                                            dag[j][i]=1
                                        else:
                                            dag[i][j]=10
                                            dag[j][i]=10
                                        continue
                                if found:
                                    break

                        else:
                            data_i = data_final[f'per_{i}'][:,[i,j,-1]]
                            data_p_i = np.concatenate((data_i, data_final[f'per_{i}'][:,c_set]), axis=1)
                            g_adj = [i for i in range(3,data_p_i.shape[1])]
                            CIT_obi = CIT(data_p_i,"kci", kernelX='Polynomial', kernelY='Polynomial')
                            Upi_value = CIT_obi(1,2, g_adj)
                            Cpi_value = CIT_obi(1,2,g_adj+[0])
                            print(f'non-middle: {Upi_value}, {Cpi_value}')
                            if Upi_value < threshold and Cpi_value < threshold:
                                dag[i][j]=10
                                dag[j][i]=9
                            elif Upi_value > threshold and Cpi_value < threshold:
                                dag[i][j]=10
                                dag[j][i]=1
                            elif Upi_value < threshold and Cpi_value > threshold:
                                dag[i][j]=10
                                dag[j][i]=-1
                            else:
                                if 0.05 > (Upi_value -threshold) > (Cpi_value- threshold):
                                    dag[i][j]=10
                                    dag[j][i]=-1
                                elif 0.05 > (Cpi_value -threshold) > (Upi_value- threshold):
                                    dag[i][j]=10
                                    dag[j][i]=1
                                else:
                                    dag[i][j]=10
                                    dag[j][i]=10

        # import pdb
        # pdb.set_trace()
        dag_n = deepcopy(dag)
        bk_dag = dag_for_bk(dag_n) 
        bk = BackgroundKnowledge()
        
        for i in range(n_node):
            for j in range(i+1, n_node):
                if bk_dag[i][j] ==1:
                    u, v = Node(), Node()
                    u.set_name(str(i))
                    v.set_name(str(j))
                    bk.add_required_by_node(u, v)
                    # bk.add_required_by_node(int(i),int(j))
                elif bk_dag[j][i] ==1:
                    # bk.add_required_by_node(int(j),int(i))
                    u, v = Node(), Node()
                    u.set_name(str(j))
                    v.set_name(str(i))
                    bk.add_required_by_node(u, v)

        cg,_ = fci(obs,kci,0.05,background_knowledge = bk)
        FCI_result = cg.graph
        count_ = 0
        for i in range(n_node):
            for j in range(n_node):
                if dag[i][j] in (10,1) and dag[j][i] in (1,10):
                    if i not in interven_list and j not in interven_list:
                        count_+=1
                        if FCI_result[i][j] < 0:
                            dag[i][j] = FCI_result[i][j]
                        elif FCI_result[i][j] == 1:
                            dag[i][j] = FCI_result[i][j]
                        elif FCI_result[i][j] == 2:
                            dag[i][j] = 10
        print(count_)


        
        
        draw_graph_from_matrix(dag, filename=f"graph_{save}.png")
        dag = dag_transformation(dag)


        count_s = 0
        count_l = 0
      
        for s in result['selection']:
            if s in data['selection'] or (s[1], s[0]) in data['selection']:
                count_s += 1
        for l in result['latent']:
            for t in data['latent']:
                if l[0] in t and l[1] in t:
                    count_l += 1

        if len(data['selection']) != 0:
            if len(result['selection']) == 0:
                acc_selection.append(0)
            else:
                acc_selection.append(count_s/len(result['selection']))

        ret_dire = count_dag_accuracy(data['dag'], dag)
        print("Directions 1 by CausalDAG: ", ret_dire)
        if ret_dire['f1'] == None:
            ret_dire['f1'] = 0
        if ret_dire['recall'] == None:
            ret_dire['recall'] = 0
        if ret_dire['precision'] == None:
            ret_dire['precision'] = 0
        f1_dag.append(ret_dire['f1'])
        recall_dag.append(ret_dire['recall'])
        acc_dag.append(ret_dire['precision'])
        shd_dag.append(ret_dire['shd'])
        
    
    print(f'the reuslt of num_node: {d} num_sample: {sample_size_select} intervention: {interven}')
    print(result['selection'])
    print(f'the average accuracy of selection is {np.mean(acc_selection)}, variance is {np.var(acc_selection)}')
    # print(f'the average accuracy of latent is {np.mean(acc_latent)}, variance is {np.var(acc_latent)}')
    print(f'the average accuracy of dag is {np.mean(acc_dag)}, variance is {np.var(acc_dag)}')
    print(f'the average accuracy of recall of dag is {np.mean(recall_dag)}, variance is {np.var(recall_dag)}')
    print(f'the average f1 score of dag is {np.mean(f1_dag)}, variance is {np.var(f1_dag)}')
    print(f'the average shd of dag is {np.mean(shd_dag)}, variance is {np.var(shd_dag)}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_nodes", type=int, default=0)
    parser.add_argument("--num_samples", type=int, default=0)
    parser.add_argument("--intervention", type=str, default='hard')
    parser.add_argument("--noise", type=str, default='high')
    main(parser.parse_args())