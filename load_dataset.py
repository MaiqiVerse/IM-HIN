import pickle
import os
import json
import pandas as pd
import numpy as np
import torch
import scipy.sparse as sp
import time

global_index = {}


def load_link_prediction_dataset_folds(dataset_name, dataset_path, ratio=1,
                                       use_structure_feature=True):
    if dataset_name == 'ppi_phenotype':
        return ppi_phenotype_lp_process_folds(dataset_path, ratio, use_structure_feature)
    raise NameError(f'{dataset_name} model is not supported')


def ppi_phenotype_lp_process_folds(dataset_path, ratio=1, use_structure_feature=True):
    if os.path.isdir("./folds_original"):
        with open("./folds_original/networks.pkl", "rb") as f:
            networks = pickle.load(f)
        with open("./folds_original/full_annotation.pkl", "rb") as f:
            full_annotation = pickle.load(f)
        with open("./folds_original/train_idx.pkl", "rb") as f:
            train_idx = pickle.load(f)
        with open("./folds_original/valid_idx.pkl", "rb") as f:
            valid_idx = pickle.load(f)
        with open("./folds_original/test_idx.pkl", "rb") as f:
            test_idx = pickle.load(f)
        with open("./folds_original/protein_list.pkl", "rb") as f:
            protein_list = pickle.load(f)

    else:
        networks, full_annotation, train_idx, valid_idx, test_idx, protein_list = load_original_dataset(dataset_path)
        os.makedirs("./folds_original", exist_ok=True)
        with open("./folds_original/networks.pkl", "wb") as f:
            pickle.dump(networks, f)
        with open("./folds_original/full_annotation.pkl", "wb") as f:
            pickle.dump(full_annotation, f)
        with open("./folds_original/train_idx.pkl", "wb") as f:
            pickle.dump(train_idx, f)
        with open("./folds_original/valid_idx.pkl", "wb") as f:
            pickle.dump(valid_idx, f)
        with open("./folds_original/test_idx.pkl", "wb") as f:
            pickle.dump(test_idx, f)
        with open("./folds_original/protein_list.pkl", "wb") as f:
            pickle.dump(protein_list, f)

    protein_features = generate_protein_features(networks, gtype="ppmi")

    index_11_30, index_31_100, index_101_300, index_more_300 = get_pheotype_index(full_annotation)
    global_index["11_30"] = index_11_30
    global_index["31_100"] = index_31_100
    global_index["101_300"] = index_101_300
    global_index["more_300"] = index_more_300

    data_keys = [
        "train_networks1", "train_networks2", "train_node_pairs", "valid_node_pairs", "valid_labels",
        "valid_constractive_pairs", "test_networks1", "test_networks2", "test_node_pairs", "test_labels",
        "train_node_features", "test_node_features"
    ]

    if use_structure_feature:
        if os.path.exists(f"./folds/data_structure_{ratio}.pkl"):
            data = torch.load(f'./folds/data_structure_{ratio}.pkl')
            print("Load data from file")
            return (data[i] for i in data_keys)
        elif os.path.exists(f"./folds/data_{ratio}.pkl"):
            data = torch.load(f'./folds/data_{ratio}.pkl')
            # generate random features in torch.float32
            data["train_node_features"] = structure_feature_process(len(data["train_node_features"][0]), protein_list)
            data["test_node_features"] = structure_feature_process(len(data["test_node_features"][0]), protein_list)
            torch.save(data, f"./folds/data_structure_{ratio}.pkl")
            print("Load data from file")
            return (data[i] for i in data_keys)

    if os.path.exists(f"./folds/data_{ratio}.pkl"):
        data = torch.load(f'./folds/data_{ratio}.pkl')

        print("Load data from file")

    else:

        data = {key: [] for key in data_keys}

        for fold in range(5):
            print(f"Fold {fold + 1} processing")
            fold_start = time.time()
            train_network_extend, train_node_pairs = construct_networks(full_annotation, train_idx[fold], ratio=0.2,
                                                                        pos_neg_ratio=ratio)
            print("Finish construct train network:", time.time() - fold_start)

            valid_start = time.time()
            valid_node_pairs, valid_labels = construct_valid_pairs(full_annotation, valid_idx[fold])
            valid_constractive_pairs = construct_constractive_pairs(full_annotation, valid_idx[fold],
                                                                    pos_neg_ratio=ratio)
            print("Finish generate valid pairs:", time.time() - valid_start)

            test_network_start = time.time()
            test_network_extend, test_node_pairs, test_labels = construct_test_networks(full_annotation, test_idx[fold],
                                                                                        pos_neg_ratio=ratio)
            print("Finish construct test network:", time.time() - test_network_start)

            encoder1_start = time.time()
            network1 = get_encoder1_network(networks, train_network_extend)
            print("Finish get encoder1 network:", time.time() - encoder1_start)
            encoder2_start = time.time()
            network2 = get_encoder2_network(networks, train_network_extend)
            print("Finish get encoder2 network:", time.time() - encoder2_start)

            test_network1_start = time.time()
            test_network1 = get_encoder1_network(networks, test_network_extend)
            print("Finish get test encoder1 network:", time.time() - test_network1_start)
            test_network2_start = time.time()
            test_network2 = get_encoder2_network(networks, test_network_extend)
            print("Finish get test encoder2 network:", time.time() - test_network2_start)

            term_feature_start = time.time()
            term_features = generate_term_features(train_network_extend, protein_features, gtype="agg")
            test_term_features = generate_term_features(test_network_extend, protein_features, gtype="agg")
            print("Finish generate term features:", time.time() - term_feature_start)

            node_features = generate_node_features(protein_features, term_features)
            test_node_features = generate_node_features(protein_features, test_term_features)

            data["train_networks1"].append(network1)
            data["train_networks2"].append(network2)

            data["train_node_pairs"].append(train_node_pairs)
            data["valid_node_pairs"].append(valid_node_pairs)
            data["valid_labels"].append(valid_labels)
            data["valid_constractive_pairs"].append(valid_constractive_pairs)

            data["test_networks1"].append(test_network1)
            data["test_networks2"].append(test_network2)
            data["test_node_pairs"].append(test_node_pairs)
            data["test_labels"].append(test_labels)

            data["train_node_features"].append(node_features)
            data["test_node_features"].append(test_node_features)

            print(f"Fold {fold + 1} finished:", time.time() - fold_start)

        begin = time.time()
        data = convert2tensor(data)
        print("Finish convert to tensor:", time.time() - begin)

        os.makedirs("./folds", exist_ok=True)
        begin = time.time()
        torch.save(data, f'./folds/data_{ratio}.pkl')
        print("Finish save data:", time.time() - begin)

        if use_structure_feature:
            # generate random features in torch.float32
            data["train_node_features"] = structure_feature_process(len(data["train_node_features"][0]), protein_list)
            data["test_node_features"] = structure_feature_process(len(data["test_node_features"][0]), protein_list)
            torch.save(data, f"./folds/data_structure_{ratio}.pkl")

    return (data[i] for i in data_keys)


def structure_feature_process(num, protein_list):
    file_list = os.listdir("./embeddings/")
    feature_all = []
    for protein in protein_list:
        if protein + ".npy" in file_list:
            feature = np.load(f"./embeddings/{protein}.npy")
            feature = struture_feature_process_step(feature)
            feature_all.append(feature)
        else:
            feature_all.append(struture_feature_process_step())

    for i in range(num - len(protein_list)):
        feature_all.append(struture_feature_process_step())

    # convert to torch tensor
    feature_all = np.concatenate(feature_all, axis=0)
    feature_all = torch.from_numpy(feature_all).float()
    return [feature_all] * 5


def struture_feature_process_step(feature=None):
    dim = 1024  # Feature dimension
    if feature is not None:
        # Ensure feature is a numpy array
        feature = np.array(feature, dtype=np.float32)

    else:
        # Generate random feature from standard normal distribution
        feature = np.random.randn(1, dim).astype(np.float32)

    return feature


def convert2tensor(data):
    if isinstance(data, dict):
        return {key: convert2tensor(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert2tensor(value) for value in data]
    elif isinstance(data, np.ndarray):
        if data.dtype == np.float64:
            return torch.FloatTensor(data)
        return torch.from_numpy(data)
    elif isinstance(data, sp.spmatrix):
        data = data.tocoo()
        return torch.sparse_coo_tensor(
            torch.LongTensor([data.row.tolist(), data.col.tolist()]),
            torch.FloatTensor(data.data.tolist()),
            torch.Size(data.shape)
        )
    return data


def generate_node_features(protein_features, term_features):
    return np.concatenate([protein_features, term_features], axis=0)


def generate_term_features(network, protein_features, gtype="agg"):
    if gtype == "agg":
        # term_features = np.matmul(network.T, protein_features)
        # accelerate the process by using sparse matrix
        network = sp.coo_matrix(network.T)
        term_features = network @ protein_features
    elif gtype == "random":
        term_features = np.random.randn(network.shape[0], protein_features.shape[1])
    else:
        raise NameError(f'{gtype} is not supported')
    return term_features


def get_encoder1_network(networks, train_network_extend):
    # networks are three different PPI networks
    # train_network_extend is the network between proteins and terms
    # Meta Path: PPI1, PPI2, PPI3, P-T
    networks_new = []

    extend = train_network_extend

    # put them to an ajacency matrix which shape is ((num_proteins + num_terms), (num_proteins + num_terms))
    # convert to sparse matrix
    shape = networks[0].shape[0] + train_network_extend.shape[1]
    for net in networks:
        net = sp.coo_matrix(net)

        data = net.data.copy()
        row = net.row.copy()
        col = net.col.copy()

        net = sp.coo_matrix((data, (row, col)), shape=(shape, shape))
        networks_new.append(net)

    # now convert extended network to sparse matrix
    # notice that extend is shape of (num_proteins, num_terms)
    # so term index is from num_proteins to num_proteins + term_index
    extend = sp.coo_matrix(extend)

    data = extend.data.copy()
    row = extend.row.copy()
    col = extend.col.copy()
    col = col + networks[0].shape[0]

    extend = sp.coo_matrix((data, (row, col)), shape=(shape, shape))

    networks_new.append(extend)
    networks_new = [net + net.T for net in networks_new]
    return networks_new


def get_encoder2_network(networks, train_network_extend):
    # this to encoder term
    # Meta Path: T-P
    shape = networks[0].shape[0] + train_network_extend.shape[1]
    extend = train_network_extend.T

    extend = sp.coo_matrix(extend)
    data = extend.data.copy()
    row = extend.row.copy()
    col = extend.col.copy()
    row = row + networks[0].shape[0]

    extend = sp.coo_matrix((data, (row, col)), shape=(shape, shape))

    networks_new = [extend]

    networks_new = [net + net.T for net in networks_new]
    return networks_new


def construct_networks(protein_term_matrix, index, ratio=0.2, pos_neg_ratio=1):
    # create a protein term matrix with only the selected proteins
    protein_term_matrix_selected = np.zeros_like(protein_term_matrix)
    protein_term_matrix_selected[index] = protein_term_matrix[index]

    positive_pairs = np.argwhere(protein_term_matrix_selected)
    np.random.shuffle(positive_pairs)

    num_positive_pairs = positive_pairs.shape[0]
    num_negative_pairs = int(num_positive_pairs * ratio)

    network = np.zeros_like(protein_term_matrix_selected)
    network[positive_pairs[num_negative_pairs:, 0], positive_pairs[num_negative_pairs:, 1]] = 1

    # construct protein term matrix with only the selected proteins
    protein_term_matrix_selected[index] = 1 - protein_term_matrix_selected[index]

    negative_pairs = np.argwhere(protein_term_matrix_selected)
    np.random.shuffle(negative_pairs)

    num_positive_pairs = num_negative_pairs
    num_negative_pairs = int(num_positive_pairs * pos_neg_ratio)

    if num_negative_pairs > negative_pairs.shape[0]:
        num_negative_pairs = negative_pairs.shape[0]

    positive_pairs = positive_pairs[:num_positive_pairs]
    negative_pairs = negative_pairs[:num_negative_pairs]
    node_pairs = np.concatenate([positive_pairs, negative_pairs], axis=0)

    # convert term index to overall index
    node_pairs[:, 1] += protein_term_matrix.shape[0]

    return network, node_pairs


def construct_valid_pairs(protein_term_matrix, index):
    protein_term_matrix_selected = np.zeros_like(protein_term_matrix)
    protein_term_matrix_selected[index] = protein_term_matrix[index]

    positive_pairs = np.argwhere(protein_term_matrix_selected)

    protein_term_matrix_selected[index] = 1 - protein_term_matrix_selected[index]
    negative_pairs = np.argwhere(protein_term_matrix_selected)

    valid_labels = np.zeros(positive_pairs.shape[0] + negative_pairs.shape[0])
    valid_labels[:positive_pairs.shape[0]] = 1

    node_pairs = np.concatenate([positive_pairs, negative_pairs], axis=0)

    # convert term index to overall index
    node_pairs[:, 1] += protein_term_matrix.shape[0]

    return node_pairs, valid_labels


def construct_constractive_pairs(protein_term_matrix, index, pos_neg_ratio=1):
    protein_term_matrix_selected = np.zeros_like(protein_term_matrix)
    protein_term_matrix_selected[index] = protein_term_matrix[index]

    positive_pairs = np.argwhere(protein_term_matrix_selected)
    num_positive_pairs = positive_pairs.shape[0]

    protein_term_matrix_selected[index] = 1 - protein_term_matrix_selected[index]
    negative_pairs = np.argwhere(protein_term_matrix_selected)
    # if num_positive_pairs > negative_pairs.shape[0]:
    #     num_positive_pairs = negative_pairs.shape[0]

    positive_pairs = positive_pairs[:num_positive_pairs]
    num_negative_pairs = int(num_positive_pairs * pos_neg_ratio)
    if num_negative_pairs > negative_pairs.shape[0]:
        num_negative_pairs = negative_pairs.shape[0]
    negative_pairs = negative_pairs[:num_negative_pairs]
    node_pairs = np.concatenate([positive_pairs, negative_pairs], axis=0)

    # convert term index to overall index
    node_pairs[:, 1] += protein_term_matrix.shape[0]
    return node_pairs


def construct_test_networks(protein_term_matrix, index, pos_neg_ratio=1):
    protein_term_matrix_selected = protein_term_matrix.copy()
    protein_term_matrix_selected[index] = 0

    protein_term_matrix_selected_pairs = np.zeros_like(protein_term_matrix)
    protein_term_matrix_selected_pairs[index] = protein_term_matrix[index]
    positive_pairs = np.argwhere(protein_term_matrix_selected_pairs)
    protein_term_matrix_selected_pairs[index] = 1 - protein_term_matrix_selected_pairs[index]
    negative_pairs = np.argwhere(protein_term_matrix_selected_pairs)
    node_pairs = np.concatenate([positive_pairs, negative_pairs], axis=0)
    test_labels = np.zeros(positive_pairs.shape[0] + negative_pairs.shape[0])
    test_labels[:positive_pairs.shape[0]] = 1

    node_pairs[:, 1] += protein_term_matrix.shape[0]

    return protein_term_matrix_selected, node_pairs, test_labels


def generate_protein_features(networks, gtype="ppmi", embedding_dim=300):
    if gtype == "ppmi":
        protein_features = [PPMI_matrix(net) for net in networks]
        protein_features = np.sum(protein_features, axis=0)
    elif gtype == "random":
        protein_features = np.random.randn(networks[0].shape[0], embedding_dim)
    else:
        raise NameError(f'{gtype} is not supported')
    return protein_features


def PPMI_matrix(M):
    """Compute Positive Pointwise Mutual Information Matrix.
    :param M: input similarity matrix
    :return: PPMI matrix of input matrix M
    """
    # normalize the matrix
    import numpy as np
    M = _scaleSimMat(M)
    n = M.shape[0]
    col = np.asarray(M.sum(axis=0), dtype=float)
    col = col.reshape((1, n))
    row = np.asarray(M.sum(axis=1), dtype=float)
    row = row.reshape((n, 1))
    D = np.sum(col)
    # compute PPMI matrix
    np.seterr(all='ignore')
    PPMI = np.log(np.divide(D * M, np.dot(row, col)))
    PPMI[np.isnan(PPMI)] = 0
    PPMI[PPMI < 0] = 0

    return PPMI


def _scaleSimMat(A):
    """Scale rows of similarity matrix.
    :param A: input similarity matrix
    :return: row-normalized matrix
    """
    import numpy as np
    A = A - np.diag(np.diag(A))
    A = A + np.diag(A.sum(axis=0) == 0)
    col = A.sum(axis=0)
    A = A.astype(float) / col[:, None]

    return A


def load_original_dataset(dataset_path):
    with open(os.path.join(dataset_path, 'hp.pkl'), 'rb') as f:
        dataset = pickle.load(f)

    term_freq = dataset["annotation"].sum(axis=0)
    term_list = term_freq[term_freq > 10].index.tolist()
    full_annotation = dataset["annotation"][term_list]
    protein_list = list(full_annotation.index)
    full_annotation = full_annotation.values

    networks = []
    paths = [
        "STRING_v12_filtered.json",
        "genemania_filtered.json",
        "humannet_xn_filtered.json",
    ]
    for path in paths:
        with open(os.path.join(dataset_path, path)) as fp:
            ppi = json.load(fp)

        ppi = pd.DataFrame(ppi).fillna(0).reindex(
            columns=protein_list, index=protein_list, fill_value=0).values
        diag = 1 / np.sqrt(np.sum(ppi, 1))
        diag[diag == np.inf] = 0
        neg_half_power_degree_matrix = np.diag(diag)
        normalized_ppi = np.matmul(np.matmul(neg_half_power_degree_matrix, ppi),
                                   neg_half_power_degree_matrix)
        networks.append(normalized_ppi)

    train_protein_index_list = []
    test_protein_index_list = []
    valid_protein_index_list = []

    for fold in range(5):
        train_protein_index = dataset["mask"][fold]["train"].reindex(
            index=protein_list, columns=term_list, fill_value=0).values.any(1)
        test_protein_index = dataset["mask"][fold]["test"].reindex(
            index=protein_list, columns=term_list, fill_value=0).values.any(1)
        valid_protein_index = dataset["mask"][fold]["valid"].reindex(
            index=protein_list, columns=term_list, fill_value=0).values.any(1)
        train_protein_index = np.where(train_protein_index)[0]
        test_protein_index = np.where(test_protein_index)[0]
        valid_protein_index = np.where(valid_protein_index)[0]
        train_protein_index_list.append(train_protein_index)
        test_protein_index_list.append(test_protein_index)
        valid_protein_index_list.append(valid_protein_index)

    train_idx = []
    valid_idx = []
    test_idx = []
    excluded = []
    for fold in range(5):
        train_protein_index = train_protein_index_list[fold]
        test_protein_index = test_protein_index_list[fold]

        valid_protein_index = valid_protein_index_list[fold]

        train_idx.append(train_protein_index)
        valid_idx.append(valid_protein_index)
        test_idx.append(test_protein_index)

        train_target = full_annotation[train_protein_index]
        test_target = full_annotation[test_protein_index]
        valid_target = full_annotation[valid_protein_index]

        for i in range(test_target.shape[1]):
            if len(np.unique(train_target[:, i])) == 1:
                excluded.append(i)

        for i in range(test_target.shape[1]):
            if len(np.unique(test_target[:, i])) == 1:
                excluded.append(i)

        for i in range(test_target.shape[1]):
            if len(np.unique(valid_target[:, i])) == 1:
                excluded.append(i)

    print("Excluded:", len(set(excluded)))
    excluded = np.array(list(set(excluded)))

    full_annotation = np.delete(full_annotation, excluded, axis=1)
    return networks, full_annotation, train_idx, valid_idx, test_idx, protein_list


def get_pheotype_index(full_annotation: np.ndarray):
    term_freq = full_annotation.sum(axis=0)
    index_11_30 = (term_freq >= 11) & (term_freq <= 30)
    index_31_100 = (term_freq > 30) & (term_freq <= 100)
    index_101_300 = (term_freq > 100) & (term_freq <= 300)
    index_more_300 = term_freq > 300

    protein_num = full_annotation.shape[0]
    # convert boolean to index
    index_11_30 = np.where(index_11_30)[0] + protein_num
    index_31_100 = np.where(index_31_100)[0] + protein_num
    index_101_300 = np.where(index_101_300)[0] + protein_num
    index_more_300 = np.where(index_more_300)[0] + protein_num

    return index_11_30, index_31_100, index_101_300, index_more_300
