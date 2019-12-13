import numpy as np
import pandas as pd
import networkx
from scipy.sparse import lil_matrix, kron,identity
from scipy.sparse.linalg import lsqr
from sklearn.utils import resample

def norm_mat(adj_mat):
    norm = adj_mat.sum(axis=0)
    norm[norm == 0] = 1
    return adj_mat / norm

def my_kernel(X, Y, lmb):

    max_size = int(np.sqrt(X.shape[1]))
    RWK = np.zeros([X.shape[0], Y.shape[0]])
    step = 10

    for i in range(0, X.shape[0]):
        for j in range(0, Y.shape[0]):
            weighted_sum = 0
            am_pg = np.kron(norm_mat(np.reshape(X[i, :], (max_size, max_size))), norm_mat(np.reshape(Y[j, :], (max_size, max_size))))
            for k in range(step):
                weighted_sum += np.dot(lmb ** k, am_pg ** k)
            rwk = weighted_sum.sum()
            RWK[i, j] = rwk

    return RWK

def random_walk_kernel(X, Y, lmb):

    max_size = int(np.sqrt(X.shape[1]))
    kernel_matrix = np.zeros([X.shape[0], Y.shape[0]])
    step = 10

    for i in range(0, X.shape[0]):
        g1 = lil_matrix(norm_mat(np.reshape(X[i, :], (max_size, max_size))))
        for j in range(0, Y.shape[0]):
            g2 = lil_matrix(norm_mat(np.reshape(Y[j, :], (max_size, max_size))))
            weighted_sum = 0
            am_pg = kron(g1, g2)
            for k in range(step):
                weighted_sum += (am_pg ** k).dot(lmb ** k)
            kernel_matrix[i, j] = weighted_sum.sum()

    return kernel_matrix

def random_walk_kernel_1(X, Y, lmb):

    max_size = int(np.sqrt(X.shape[1]))
    kernel_matrix = np.zeros([X.shape[0], Y.shape[0]])

    for i in range(0, X.shape[0]):
        g1 = norm_mat(np.reshape(X[i, :], (max_size, max_size)))
        for j in range(0, Y.shape[0]):
            g2 = norm_mat(np.reshape(Y[j, :], (max_size, max_size)))
            w_prod = kron(lil_matrix(g1), lil_matrix(g2))
            starting_prob = np.ones(w_prod.shape[0]) / (w_prod.shape[0])
            stop_prob = starting_prob
            A = identity(w_prod.shape[0]) - (w_prod * lmb)
            x = lsqr(A, starting_prob)
            kernel_matrix[i, j] = stop_prob.T.dot(x[0])

    return kernel_matrix

def compute_kernel_matrix(X, Y, lmb, type):

    kernel_matrix = np.zeros([len(X), len(Y)])

    for i in range(0, len(X)):
        for j in range(0, len(Y)):
            if type == "RWK":
                kernel_matrix[i, j] = RWK(X[i], Y[j], lmb)
            elif type == "RWK_norm":
                kernel_matrix[i, j] = RWK_norm(X[i], Y[j], lmb)
            elif type == "RWK_1":
                kernel_matrix[i, j] = RWK_1(X[i], Y[j], lmb)
            elif type == "RWK_1_norm":
                kernel_matrix[i, j] = RWK_1_norm(X[i], Y[j], lmb)

    return kernel_matrix

def RWK(X, Y, lmb):

    step = 10
    weighted_sum = 0
    g1 = norm_mat(networkx.adjacency_matrix(X))
    g2 = norm_mat(networkx.adjacency_matrix(Y))
    g_prod = kron(lil_matrix(g1), lil_matrix(g2))

    for n in range(step):
        weighted_sum += (g_prod ** n).dot(lmb ** n)

    k = weighted_sum.sum()

    return k

def RWK_norm(X, Y, lmb):

    step = 10
    weighted_sum = 0
    weighted_sum_1 = 0
    weighted_sum_2 = 0
    g1 = norm_mat(networkx.adjacency_matrix(X))
    g2 = norm_mat(networkx.adjacency_matrix(Y))
    g_prod = kron(lil_matrix(g1), lil_matrix(g2))
    g_prod_1 = kron(lil_matrix(g1), lil_matrix(g1))
    g_prod_2 = kron(lil_matrix(g2), lil_matrix(g2))

    for n in range(step):
        weighted_sum += (g_prod ** n).dot(lmb ** n)
        weighted_sum_1 += (g_prod_1 ** n).dot(lmb ** n)
        weighted_sum_2 += (g_prod_2 ** n).dot(lmb ** n)

    k = weighted_sum.sum()
    k_1 = weighted_sum_1.sum()
    k_2 = weighted_sum_2.sum()
    k_norm = k / np.sqrt(k_1 * k_2)

    return k_norm

def RWK_1(X, Y, lmb):

    g1 = norm_mat(networkx.adjacency_matrix(X))
    g2 = norm_mat(networkx.adjacency_matrix(Y))
    g_prod = kron(lil_matrix(g1), lil_matrix(g2))
    starting_prob = np.ones(g_prod.shape[0]) / (g_prod.shape[0])
    stop_prob = starting_prob
    A = identity(g_prod.shape[0]) - (g_prod * lmb)
    x = lsqr(A, starting_prob)
    k = stop_prob.T.dot(x[0])

    return k

def RWK_1_norm(X, Y, lmb):

    g1 = norm_mat(networkx.adjacency_matrix(X))
    g2 = norm_mat(networkx.adjacency_matrix(Y))
    g_prod = kron(lil_matrix(g1), lil_matrix(g2))
    g_prod_1 = kron(lil_matrix(g1), lil_matrix(g1))
    g_prod_2 = kron(lil_matrix(g2), lil_matrix(g2))
    starting_prob = np.ones(g_prod.shape[0]) / (g_prod.shape[0])
    starting_prob_1 = np.ones(g_prod_1.shape[0]) / (g_prod_1.shape[0])
    starting_prob_2 = np.ones(g_prod_2.shape[0]) / (g_prod_2.shape[0])
    stop_prob = starting_prob
    stop_prob_1 = starting_prob_1
    stop_prob_2 = starting_prob_2
    A = identity(g_prod.shape[0]) - (g_prod * lmb)
    A_1 = identity(g_prod_1.shape[0]) - (g_prod_1 * lmb)
    A_2 = identity(g_prod_2.shape[0]) - (g_prod_2 * lmb)
    x = lsqr(A, starting_prob)
    x_1 = lsqr(A_1, starting_prob_1)
    x_2 = lsqr(A_2, starting_prob_2)
    k = stop_prob.T.dot(x[0])
    k_1 = stop_prob_1.T.dot(x_1[0])
    k_2 = stop_prob_2.T.dot(x_2[0])
    k_norm = k / np.sqrt(k_1 * k_2)

    return k_norm

def balance_data(G, label, random_state):
    graphs = {'CFG': G, 'label': label}
    #print(graphs)
    df = pd.DataFrame(data=graphs)
    
    if len(df[df.label == -1]) > len(df[df.label == 1]):
        df_majority = df[df.label == -1]
        df_minority = df[df.label == 1]
    else:
        df_majority = df[df.label == 1]
        df_minority = df[df.label == -1]

    df_minority_upsampled = resample(df_minority, replace=True, n_samples=len(df_majority), random_state=random_state)
    df_upsampled = pd.concat([df_majority, df_minority_upsampled], ignore_index=True)
    
    data = np.asarray(df['CFG'])
    target = np.asarray(df['label'])

    return data, target

def rename_nodes(node_label):

    n = len(node_label)
    print(node_label)
    labels = [0] * n
    label_lookup = {}
    label_counter = 0

    for i in range(n):
        num_nodes = len(node_label[i])
        temp_lables = list(node_label[i])
        labels[i] = np.zeros(num_nodes, dtype=np.uint64)
        for j in range(num_nodes):
            temp_node_str = str(np.copy(temp_lables[j]))
            if temp_node_str not in label_lookup:
                label_lookup[temp_node_str] = label_counter
                labels[i][j] = label_counter
                label_counter += 1
            else:
                labels[i][j] = label_lookup[temp_node_str]

    L = label_counter
    print('Number of original labels: %d' % L)

    return L, labels
