import os
import numpy as np
import torch
import pdb
from sklearn import metrics
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
import time
import json
from dgl import mean_nodes

class Evaluator():
    def __init__(self, params, graph_classifier, data):
        self.params = params
        self.graph_classifier = graph_classifier
        self.data = data

    def eval(self, save=False):
        pos_scores = []
        pos_labels = []
        neg_scores = []
        neg_labels = []
        pos_orders = []
        neg_orders = []
        dataloader = DataLoader(self.data, batch_size=self.params.batch_size, shuffle=False, num_workers=self.params.num_workers, collate_fn=self.params.collate_fn)

        self.graph_classifier.eval()
        with torch.no_grad():
            b_start_time = time.time()
            for b_idx, batch in enumerate(dataloader):

                data_pos, targets_pos, data_neg, targets_neg = self.params.move_batch_to_device(batch, self.params.device)
                score_pos, _, _, _ = self.graph_classifier(data_pos)
                score_neg, _, _, _ = self.graph_classifier(data_neg)


                pos_scores += score_pos.squeeze(1).detach().cpu().tolist()
                neg_scores += score_neg.squeeze(1).detach().cpu().tolist()


                pos_labels += targets_pos.tolist()
                neg_labels += targets_neg.tolist()
                if (b_idx + 1) % 100 == 0:
                    print(f'{b_idx + 1}th batch, time: {time.time() - b_start_time:.2f}')
                    b_start_time = time.time()
        # acc = metrics.accuracy_score(labels, preds)
        auc = metrics.roc_auc_score(pos_labels + neg_labels, pos_scores + neg_scores)
        auc_pr = metrics.average_precision_score(pos_labels + neg_labels, pos_scores + neg_scores)

        if save:
            pos_test_triplets_path = os.path.join(self.params.main_dir, 'data/{}/{}.txt'.format(self.params.dataset, self.data.file_name))
            with open(pos_test_triplets_path) as f:
                pos_triplets = [line.split() for line in f.read().split('\n')[:-1]]
            pos_file_path = os.path.join(self.params.main_dir, 'data/{}/grail_{}_predictions.txt'.format(self.params.dataset, self.data.file_name))
            with open(pos_file_path, "w") as f:
                for ([s, r, o], score) in zip(pos_triplets, pos_scores):
                    f.write('\t'.join([s, r, o, str(score)]) + '\n')

            neg_test_triplets_path = os.path.join(self.params.main_dir, 'data/{}/neg_{}_0.txt'.format(self.params.dataset, self.data.file_name))
            with open(neg_test_triplets_path) as f:
                neg_triplets = [line.split() for line in f.read().split('\n')[:-1]]
            neg_file_path = os.path.join(self.params.main_dir, 'data/{}/grail_neg_{}_{}_predictions.txt'.format(self.params.dataset, self.data.file_name, self.params.constrained_neg_prob))
            with open(neg_file_path, "w") as f:
                for ([s, r, o], score) in zip(neg_triplets, neg_scores):
                    f.write('\t'.join([s, r, o, str(score)]) + '\n')

        return {'auc': auc, 'auc_pr': auc_pr}

class Evaluator_visualize_repr():
    def __init__(self, params, graph_classifier, data):
        self.params = params
        self.graph_classifier = graph_classifier
        self.data = data
        self.sigmoid = nn.Sigmoid()

    def eval(self, save=False):
        pos_scores = {'g_out': [], 'head_embs': [], 'tail_embs': [], 'target_r_embs': []}
        pos_labels = []
        neg_scores = {'g_out': [], 'head_embs': [], 'tail_embs': [], 'target_r_embs': []}
        neg_labels = []
        dataloader = DataLoader(self.data, batch_size=self.params.batch_size, shuffle=False, num_workers=self.params.num_workers, collate_fn=self.params.collate_fn)
        r_label_list = []
        self.graph_classifier.eval()
        with torch.no_grad():
            b_start_time = time.time()
            for b_idx, batch in enumerate(dataloader):

                data_pos, targets_pos, data_neg, targets_neg = self.params.move_batch_to_device(batch, self.params.device)
                # print([self.data.id2relation[r.item()] for r in data_pos[1]])
                # pdb.set_trace()
                g_out_pos, head_pos, tail_pos, rel_pos = self.get_repr(data_pos)
                g_out_neg, head_neg, tail_neg, rel_neg = self.get_repr(data_neg)
                r_label_list += data_pos[1].detach().cpu().tolist()

                pos_scores['g_out'].append(g_out_pos.detach().cpu().numpy())
                pos_scores['head_embs'].append(head_pos.detach().cpu().numpy())
                pos_scores['tail_embs'].append(tail_pos.detach().cpu().numpy())
                pos_scores['target_r_embs'].append(rel_pos.detach().cpu().numpy())
                neg_scores['g_out'].append(g_out_neg.detach().cpu().numpy())
                neg_scores['head_embs'].append(head_neg.detach().cpu().numpy())
                neg_scores['tail_embs'].append(tail_neg.detach().cpu().numpy())
                neg_scores['target_r_embs'].append(rel_neg.detach().cpu().numpy())


                # preds += torch.argmax(logits.detach().cpu(), dim=1).tolist()
                """
                pos_scores += score_pos.squeeze(1).detach().cpu().tolirst()
                neg_scores += score_neg.squeeze(1).detach().cpu().tolist()
                pos_labels += targets_pos.tolist()
                neg_labels += targets_neg.tolist()
                """
                if (b_idx + 1) % 100 == 0:
                    print(f'{b_idx + 1}th batch, time: {time.time() - b_start_time:.2f}')
                    b_start_time = time.time()
        for key, val in pos_scores.items():
            pos_scores[key] = np.concatenate(val, axis = 0)
        for key, val in neg_scores.items():
            neg_scores[key] = np.concatenate(val, axis = 0)
        pos_scores['r_label'] = r_label_list
        neg_scores['r_label'] = r_label_list
        with open(f'experiments/{self.params.experiment_name}/pos_emb.json', 'w') as f:
            json.dump(pos_scores, f, cls=NumpyEncoder)
        with open(f'experiments/{self.params.experiment_name}/neg_emb.json', 'w') as f:
            json.dump(neg_scores, f, cls=NumpyEncoder)
        exit()
        # acc = metrics.accuracy_score(labels, preds)
        auc = metrics.roc_auc_score(pos_labels + neg_labels, pos_scores + neg_scores)
        auc_pr = metrics.average_precision_score(pos_labels + neg_labels, pos_scores + neg_scores)

        if save:
            pos_test_triplets_path = os.path.join(self.params.main_dir, 'data/{}/{}.txt'.format(self.params.dataset, self.data.file_name))
            with open(pos_test_triplets_path) as f:
                pos_triplets = [line.split() for line in f.read().split('\n')[:-1]]
            pos_file_path = os.path.join(self.params.main_dir, 'data/{}/grail_{}_predictions.txt'.format(self.params.dataset, self.data.file_name))
            with open(pos_file_path, "w") as f:
                for ([s, r, o], score) in zip(pos_triplets, pos_scores):
                    f.write('\t'.join([s, r, o, str(score)]) + '\n')

            neg_test_triplets_path = os.path.join(self.params.main_dir, 'data/{}/neg_{}_0.txt'.format(self.params.dataset, self.data.file_name))
            with open(neg_test_triplets_path) as f:
                neg_triplets = [line.split() for line in f.read().split('\n')[:-1]]
            neg_file_path = os.path.join(self.params.main_dir, 'data/{}/grail_neg_{}_{}_predictions.txt'.format(self.params.dataset, self.data.file_name, self.params.constrained_neg_prob))
            with open(neg_file_path, "w") as f:
                for ([s, r, o], score) in zip(neg_triplets, neg_scores):
                    f.write('\t'.join([s, r, o, str(score)]) + '\n')

        return {'auc': auc, 'auc_pr': auc_pr}
    def get_repr(self, data):

        g, rel_labels = data
        head_ids = (g.ndata['id'] == 1).nonzero().squeeze(1)
        tail_ids = (g.ndata['id'] == 2).nonzero().squeeze(1)
        if self.params.input_feature == 'normal':
            pass
        elif self.params.input_feature == 'numeric':
            g.ndata['feat'] = torch.cat([g.ndata['feat'], g.ndata['attribute']], dim = 1)
        elif self.params.input_feature == 'ra':
            g.ndata['a'], _ = self.graph_classifier.ra_encoder(g)
            new_feat = torch.cat([g.ndata['feat'], g.ndata['a']], dim = 1)
            g.ndata['feat'] = self.sigmoid(self.graph_classifier.feat_layer(new_feat))

        elif self.params.input_feature == 'rra':
            g.ndata['a'], _ = self.graph_classifier.ra_encoder(g)
            h_att, t_att = g.ndata['a'][head_ids], g.ndata['a'][tail_ids]
            h_att_list = []
            t_att_list = []

            num_nodes_for_g = g.batch_num_nodes
            for i, num_nodes in enumerate(num_nodes_for_g):
                h_att_list.append(h_att[i].repeat(num_nodes, 1))
                t_att_list.append(t_att[i].repeat(num_nodes, 1))

            h_att_list = torch.cat(h_att_list)
            t_att_list = torch.cat(t_att_list)
            new_feat = torch.cat([g.ndata['feat'], g.ndata['a'] - h_att_list, g.ndata['a'] - t_att_list], dim = 1)
            g.ndata['feat'] = self.sigmoid(self.graph_classifier.feat_layer(new_feat))
        elif self.params.input_feature == 'literalE':
            pass
        else:
            raise NotImplementedError
        #g.ndata['a'], _ = self.graph_classifier.ra_encoder(g)


        g.ndata['h'] = self.graph_classifier.gnn(g)
        if self.params.input_feature == 'literalE':
            g.ndata['repr'] = self.graph_classifier.emb_num_lit(g.ndata['repr'].view(-1, self.graph_classifier.params.num_gcn_layers * self.graph_classifier.params.emb_dim), g.ndata['attribute'])
        g_out = mean_nodes(g, 'repr')
        head_embs = g.ndata['repr'][head_ids]
        tail_embs = g.ndata['repr'][tail_ids]

        return g_out.view(-1, self.graph_classifier.params.num_gcn_layers * self.graph_classifier.params.emb_dim), head_embs.view(-1, self.graph_classifier.params.num_gcn_layers * self.graph_classifier.params.emb_dim), tail_embs.view(-1, self.graph_classifier.params.num_gcn_layers * self.graph_classifier.params.emb_dim), self.graph_classifier.rel_emb(rel_labels)
    
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist() 
        return json.JSONEncoder.default(self, obj)
    

class Evaluator_isolated():
    def __init__(self, params, graph_classifier, data):
        self.params = params
        self.graph_classifier = graph_classifier
        self.data = data

    def eval(self, save=False):
        pos_scores = []
        pos_labels = []
        neg_scores = []
        neg_labels = []
        dataloader = DataLoader(self.data, batch_size=self.params.batch_size, shuffle=False, num_workers=self.params.num_workers, collate_fn=self.params.collate_fn)
        isolated_pos_lst  = []
        isolated_neg_lst  = []
        self.graph_classifier.eval()
        with torch.no_grad():
            b_start_time = time.time()
            for b_idx, batch in enumerate(dataloader):

                data_pos, targets_pos, data_neg, targets_neg = self.params.move_batch_to_device(batch, self.params.device)
                # print([self.data.id2relation[r.item()] for r in data_pos[1]])
                # pdb.set_trace()
                score_pos = self.graph_classifier(data_pos)
                score_neg = self.graph_classifier(data_neg)
                isolated_pos_idx = [True if x == 2 else False for x in data_pos[0].batch_num_nodes]
                isolated_neg_idx = [True if x == 2 else False for x in data_neg[0].batch_num_nodes]
                if isolated_pos_idx != isolated_neg_idx:
                    raise Exception("pos and neg different_1")
                isolated_pos_lst += isolated_pos_idx
                isolated_neg_lst += isolated_neg_idx

                # preds += torch.argmax(logits.detach().cpu(), dim=1).tolist()
                pos_scores += score_pos.squeeze(1).detach().cpu().tolist()
                neg_scores += score_neg.squeeze(1).detach().cpu().tolist()
                pos_labels += targets_pos.tolist()
                neg_labels += targets_neg.tolist()
                if (b_idx + 1) % 100 == 0:
                    print(f'{b_idx + 1}th batch, time: {time.time() - b_start_time:.2f}')
                    b_start_time = time.time()
        isolated_pos_labels = np.array(pos_labels)[isolated_pos_lst].tolist()
        isolated_neg_labels = np.array(neg_labels)[isolated_neg_lst].tolist()
        if len(isolated_pos_labels) != sum(isolated_pos_lst) or len(isolated_neg_labels) != sum(isolated_neg_lst):
            raise Exception("pos and neg different_2")
        isolated_pos_scores = np.array(pos_scores)[isolated_pos_lst].tolist()
        isolated_neg_scores = np.array(neg_scores)[isolated_neg_lst].tolist()

        # acc = metrics.accuracy_score(labels, preds)
        print(isolated_pos_labels + isolated_neg_labels)
        print((isolated_pos_labels + isolated_neg_labels).shape)
        print(isolated_pos_scores + isolated_neg_scores)
        print((isolated_pos_scores + isolated_neg_scores).shape)
        auc = metrics.roc_auc_score(isolated_pos_labels + isolated_neg_labels, isolated_pos_scores + isolated_neg_scores)
        auc_pr = metrics.average_precision_score(isolated_pos_labels + isolated_neg_labels, isolated_pos_scores + isolated_neg_scores)

        """
        if save:
            pos_test_triplets_path = os.path.join(self.params.main_dir, 'data/{}/{}.txt'.format(self.params.dataset, self.data.file_name))
            with open(pos_test_triplets_path) as f:
                pos_triplets = [line.split() for line in f.read().split('\n')[:-1]]
            pos_file_path = os.path.join(self.params.main_dir, 'data/{}/grail_{}_predictions.txt'.format(self.params.dataset, self.data.file_name))
            with open(pos_file_path, "w") as f:
                for ([s, r, o], score) in zip(pos_triplets, pos_scores):
                    f.write('\t'.join([s, r, o, str(score)]) + '\n')

            neg_test_triplets_path = os.path.join(self.params.main_dir, 'data/{}/neg_{}_0.txt'.format(self.params.dataset, self.data.file_name))
            with open(neg_test_triplets_path) as f:
                neg_triplets = [line.split() for line in f.read().split('\n')[:-1]]
            neg_file_path = os.path.join(self.params.main_dir, 'data/{}/grail_neg_{}_{}_predictions.txt'.format(self.params.dataset, self.data.file_name, self.params.constrained_neg_prob))
            with open(neg_file_path, "w") as f:
                for ([s, r, o], score) in zip(neg_triplets, neg_scores):
                    f.write('\t'.join([s, r, o, str(score)]) + '\n')
        """
        return {'auc': auc, 'auc_pr': auc_pr, "num_isolated_triple": sum(isolated_pos_lst)}