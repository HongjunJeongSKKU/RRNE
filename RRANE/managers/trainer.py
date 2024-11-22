import statistics
import timeit
import os
import logging
import pdb
import numpy as np
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sklearn import metrics


class Trainer():
    def __init__(self, params, graph_classifier, train, valid_evaluator=None):
        self.graph_classifier = graph_classifier
        self.valid_evaluator = valid_evaluator
        self.params = params
        self.train_data = train

        self.updates_counter = 0

        model_params = list(self.graph_classifier.parameters())
        logging.info('Total number of parameters: %d' % sum(map(lambda x: x.numel(), model_params)))

        if params.optimizer == "SGD":
            self.optimizer = optim.SGD(model_params, lr=params.lr, momentum=params.momentum, weight_decay=self.params.l2)
        if params.optimizer == "Adam":
            self.optimizer = optim.Adam(model_params, lr=params.lr, weight_decay=self.params.l2)
        if self.params.input_feature == 'rra_exp':
            self.criterion = nn.MarginRankingLoss(self.params.margin, reduction='sum')
        else:
            self.criterion = nn.MarginRankingLoss(self.params.margin, reduction='sum')
        if self.params.use_self != 'not':
            self.self_margin = nn.TripletMarginLoss(margin = self.params.self_margin, p= self.params.self_p, reduction='sum')
            self.self_criterion = nn.MarginRankingLoss(self.params.self_margin, reduction='sum')
        self.reset_training_state()

    def reset_training_state(self):
        self.best_metric = 0
        self.last_metric = 0
        self.not_improved_count = 0

    def train_epoch(self):
        total_loss = 0
        total_d_sub = 0
        total_self = 0
        all_preds = []
        all_labels = []
        all_scores = []

        dataloader = DataLoader(self.train_data, batch_size=self.params.batch_size, shuffle=True, num_workers=self.params.num_workers, collate_fn=self.params.collate_fn)
        self.graph_classifier.train()
        model_params = list(self.graph_classifier.parameters())
        b_start_time = time.time()
        for b_idx, batch in enumerate(dataloader):
            data_pos, targets_pos, data_neg, targets_neg = self.params.move_batch_to_device(batch, self.params.device)
            self.optimizer.zero_grad()
            if self.params.input_feature in ['ra', 'rra']:

    
                if self.params.use_self != 'not':
                    score_pos, head_sub_tail, hard_pos_output, hard_neg_output = self.graph_classifier(data_pos, use_self=True)
                else:
                    score_pos, head_sub_tail, _, _ = self.graph_classifier(data_pos)
                score_neg, _, _, _ = self.graph_classifier(data_neg)


                loss = self.criterion(score_pos, score_neg.view(len(score_pos), -1).mean(dim=1), torch.Tensor([1]).to(device=self.params.device))

                if self.params.order_loss == 'not':
                    pass
                else:
                    if self.params.order_loss == 'l1':
                        d_sub = torch.sum(torch.clamp(self.params.order_margin - head_sub_tail, min = 0))
                    elif self.params.order_loss == 'l2':
                        d_sub = torch.sum(torch.clamp(self.params.order_margin - head_sub_tail, min = 0) ** 2)
                    else:
                        raise NotImplementedError                
                    d_sub = d_sub * self.params.loss_coef
                    loss += d_sub

                if self.params.use_self != 'not':

                    self_loss = self.self_margin(hard_pos_output[0], hard_neg_output[0], torch.Tensor([1]).to(device=self.params.device))
                    self_loss += self.self_margin(hard_pos_output[1], hard_neg_output[1], torch.Tensor([1]).to(device=self.params.device))
                    
                    self_loss += self.self_criterion(score_pos, hard_pos_output[0], hard_neg_output[0])
                    self_loss += self.self_criterion(score_pos, hard_pos_output[1], hard_neg_output[1])
                    """
                    if self.params.order_loss == 'l1':
                        self_loss += torch.sum(torch.clamp(self.params.order_margin - (hard_pos_output[0] - score_pos), min = 0)) + torch.sum(torch.clamp(self.params.order_margin - (score_pos - hard_neg_output[0]), min = 0))
                        self_loss += torch.sum(torch.clamp(self.params.order_margin - (hard_pos_output[1] - score_pos), min = 0)) + torch.sum(torch.clamp(self.params.order_margin - (score_pos - hard_neg_output[1]), min = 0))

                    elif self.params.order_loss == 'l2':
                        self_loss += torch.sum(torch.clamp(self.params.order_margin - (hard_pos_output[0] - score_pos), min = 0) ** 2) + torch.sum(torch.clamp(self.params.order_margin - (score_pos - hard_neg_output[0]), min = 0) ** 2)
                        self_loss += torch.sum(torch.clamp(self.params.order_margin - (hard_pos_output[1] - score_pos), min = 0) ** 2) + torch.sum(torch.clamp(self.params.order_margin - (score_pos - hard_neg_output[1]), min = 0) ** 2)

                    else:
                        raise NotImplementedError
                    """
                    self_loss = self_loss * self.params.self_coef
                    loss += self_loss

                else:
                    pass

            else:

                score_pos, _, _, _ = self.graph_classifier(data_pos)
                score_neg, _, _, _ = self.graph_classifier(data_neg)
                loss = self.criterion(score_pos, score_neg.view(len(score_pos), -1).mean(dim=1), torch.Tensor([1]).to(device=self.params.device))
            loss.backward()
            self.optimizer.step()
            self.updates_counter += 1

            with torch.no_grad():
                all_scores += score_pos.squeeze().detach().cpu().tolist() + score_neg.squeeze().detach().cpu().tolist()
                all_labels += targets_pos.tolist() + targets_neg.tolist()
                total_loss += loss
                if self.params.order_loss != 'not':
                    total_d_sub += d_sub.detach().cpu().item()
                if self.params.use_self != 'not':
                    total_self += self_loss.detach().cpu().item()

            if (b_idx + 1) % 100 == 0:
                if self.params.order_loss != 'not' and self.params.use_self != 'not':
                    print(f'{b_idx + 1}th batch, total_loss: {total_loss}, ranking_loss: {total_loss - total_d_sub - self_loss}, order loss: {total_d_sub}, self loss: {self_loss} time: {time.time() - b_start_time:.2f}')
                elif self.params.order_loss != 'not' and self.params.use_self == 'not':
                    print(f'{b_idx + 1}th batch, total_loss: {total_loss}, ranking_loss: {total_loss - total_d_sub}, order loss: {total_d_sub} time: {time.time() - b_start_time:.2f}')
                #elif self.params.order_loss == 'not' and self.params.use_self != 'not':
                #    print(f'{b_idx + 1}th batch, total_loss: {total_loss}, ranking_loss: {total_loss - total_d_sub - self_loss}, self loss: {self_loss} time: {time.time() - b_start_time:.2f}')
                else:
                    print(f'{b_idx + 1}th batch, total_loss: {total_loss} time: {time.time() - b_start_time:.2f}')
                b_start_time = time.time()


            if self.valid_evaluator and self.params.eval_every_iter and self.updates_counter % self.params.eval_every_iter == 0:
                tic = time.time()
                result = self.valid_evaluator.eval()
                logging.info('\nPerformance:' + str(result) + 'in ' + str(time.time() - tic))

                if result['auc'] >= self.best_metric:
                    self.save_classifier()
                    self.best_metric = result['auc']
                    self.not_improved_count = 0

                else:
                    self.not_improved_count += 1
                    if self.not_improved_count > self.params.early_stop:
                        logging.info(f"Validation performance didn\'t improve for {self.params.early_stop} epochs. Training stops.")
                        break
                self.last_metric = result['auc']

        auc = metrics.roc_auc_score(all_labels, all_scores)
        auc_pr = metrics.average_precision_score(all_labels, all_scores)
        if self.params.order_loss != 'not' and self.params.use_self != 'not':
            total_loss = (total_loss, total_d_sub, total_self)
        elif self.params.order_loss != 'not' and self.params.use_self == 'not':
            total_loss = (total_loss, total_d_sub, None)
        #elif self.params.order_loss == 'not' and self.params.use_self != 'not':
        #    total_loss = (total_loss, None, total_self)
        else:
            total_loss = (total_loss, None, None)
        weight_norm = sum(map(lambda x: torch.norm(x), model_params))



        return total_loss, auc, auc_pr, weight_norm

    def train(self):
        self.reset_training_state()

        for epoch in range(1, self.params.num_epochs + 1):
            time_start = time.time()
            loss, auc, auc_pr, weight_norm = self.train_epoch()
            time_elapsed = time.time() - time_start

            if self.params.order_loss != 'not' and self.params.use_self != 'not':
                logging.info(f'Epoch {epoch} with loss: {loss[0]}, ranking loss: {loss[0] -loss[1] -loss[2]}, order loss: {loss[1]}, self loss: {loss[2]} training auc: {auc}, training auc_pr: {auc_pr}, best validation AUC: {self.best_metric}, weight_norm: {weight_norm} in {time_elapsed}')
            elif self.params.order_loss != 'not' and self.params.use_self == 'not':
                logging.info(f'Epoch {epoch} with loss: {loss[0]}, ranking loss: {loss[0] -loss[1]}, order loss: {loss[1]}, training auc: {auc}, training auc_pr: {auc_pr}, best validation AUC: {self.best_metric}, weight_norm: {weight_norm} in {time_elapsed}')
            #elif self.params.order_loss == 'not' and self.params.use_self != 'not':
            #    logging.info(f'Epoch {epoch} with loss: {loss[0]}, ranking loss: {loss[0] - loss[2]}, self loss: {loss[2]} training auc: {auc}, training auc_pr: {auc_pr}, best validation AUC: {self.best_metric}, weight_norm: {weight_norm} in {time_elapsed}')
            else:
                logging.info(f'Epoch {epoch} with loss: {loss[0]}, training auc: {auc}, training auc_pr: {auc_pr}, best validation AUC: {self.best_metric}, weight_norm: {weight_norm} in {time_elapsed}')

            if epoch % self.params.save_every == 0:
                torch.save(self.graph_classifier, os.path.join(self.params.exp_dir, 'graph_classifier_chk.pth'))

    def save_classifier(self):
        torch.save(self.graph_classifier, os.path.join(self.params.exp_dir, 'best_graph_classifier.pth'))  # Does it overwrite or fuck with the existing file?
        logging.info('Better models found w.r.t accuracy. Saved it!')
