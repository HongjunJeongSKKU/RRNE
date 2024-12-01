import os
import pdb
import numpy as np
from scipy.sparse import csc_matrix, coo_matrix
import matplotlib.pyplot as plt


def plot_rel_dist(adj_list, filename):
    rel_count = []
    for adj in adj_list:
        rel_count.append(adj.count_nonzero())

    fig = plt.figure(figsize=(12, 8))
    plt.plot(rel_count)
    fig.savefig(filename, dpi=fig.dpi)


def process_files(files, saved_relation2id=None, saved_attribute2id=None):
    '''
    files: Dictionary map of file paths to read the triplets from.
    saved_relation2id: Saved relation2id (mostly passed from a trained model) which can be used to map relations to pre-defined indices and filter out the unknown ones.
    '''
    entity2id = {}
    relation2id = {} if saved_relation2id is None else saved_relation2id
    attribute2id = {} if saved_attribute2id is None else saved_attribute2id
    
    triplets = {}
    neg_triplets = {}

    ent = 0
    rel = 0
    att = 0

    for file_type, file_path in files.items():

        data = []
        with open(file_path) as f:
            file_data = [line.split() for line in f.read().split('\n')[:-1]]

        if not file_type == 'literals':
            for triplet in file_data:
                if triplet[0] not in entity2id:
                    entity2id[triplet[0]] = ent
                    ent += 1
                if triplet[2] not in entity2id:
                    entity2id[triplet[2]] = ent
                    ent += 1
                if not saved_relation2id and triplet[1] not in relation2id:
                    relation2id[triplet[1]] = rel
                    rel += 1

                # Save the triplets corresponding to only the known relations
                if triplet[1] in relation2id:
                    data.append([entity2id[triplet[0]], entity2id[triplet[2]], relation2id[triplet[1]]])
            if file_type in ['train', 'valid', 'test']:
                triplets[file_type] = np.array(data)
            else:
                neg_triplets[file_type.split('_')[0]] = np.array(data)
        else:
            row, col, data = [], [], []
            for literal in file_data:

                if literal[1] not in attribute2id.keys():
                    attribute2id[literal[1]] = att
                    att += 1
                # if literal[1] in attribute2id:
                row.append(entity2id[literal[0]])
                col.append(attribute2id[literal[1]])
                data.append(float(literal[2]))
            row, col, data = np.array(row), np.array(col), np.array(data)

            num_lit = coo_matrix((data, (row, col)), shape=(len(entity2id), len(attribute2id))).toarray()

            #max_lit, min_lit = np.max(num_lit, axis=0), np.min(num_lit, axis=0)
            #num_lit = (num_lit - min_lit) / (max_lit - min_lit + 1e-8)

    id2entity = {v: k for k, v in entity2id.items()}
    id2relation = {v: k for k, v in relation2id.items()}
    id2attribute = {v: k for k, v in attribute2id.items()}
    # Construct the list of adjacency matrix each corresponding to eeach relation. Note that this is constructed only from the train data.
    adj_list = []
    for i in range(len(relation2id)):
        idx = np.argwhere(triplets['train'][:, 2] == i)
        adj_list.append(csc_matrix((np.ones(len(idx), dtype=np.uint8), (triplets['train'][:, 0][idx].squeeze(1), triplets['train'][:, 1][idx].squeeze(1))), shape=(len(entity2id), len(entity2id))))

    return adj_list, triplets, neg_triplets, num_lit, entity2id, relation2id, attribute2id, id2entity, id2relation, id2attribute


def save_to_file(directory, file_name, triplets, id2entity, id2relation):
    file_path = os.path.join(directory, file_name)
    with open(file_path, "w") as f:
        for s, o, r in triplets:
            f.write('\t'.join([id2entity[s], id2relation[r], id2entity[o]]) + '\n')
