from .rgcn_model import RGCN
from dgl import mean_nodes
import torch.nn as nn
import torch
import torch.nn.init as init
import dgl
"""
File based off of dgl tutorial on RGCN
Source: https://github.com/dmlc/dgl/tree/master/examples/pytorch/rgcn
"""


class GraphClassifier(nn.Module):
    def __init__(self, params, relation2id, attribute2id):  # in_dim, h_dim, rel_emb_dim, out_dim, num_rels, num_bases):
        super().__init__()

        self.params = params
        self.relation2id = relation2id
        self.attribute2id = attribute2id

        self.gnn = RGCN(params)  # in_dim, h_dim, h_dim, num_rels, num_bases)
        self.rel_emb = nn.Embedding(self.params.num_rels, self.params.rel_emb_dim, sparse=False)
        self.activation = nn.ReLU()

        if self.params.add_ht_emb:
            self.fc_layer = nn.Linear(3 * self.params.num_gcn_layers * self.params.emb_dim + self.params.rel_emb_dim, 1)
        else:
            self.fc_layer = nn.Linear(self.params.num_gcn_layers * self.params.emb_dim + self.params.rel_emb_dim, 1)

        if self.params.input_feature in ['ra', 'rra']:
            self.att_emb = nn.Embedding(self.params.num_atts, self.params.rel_emb_dim, sparse=False)
            self.MHA = nn.MultiheadAttention(embed_dim = self.params.rel_emb_dim, num_heads = 1)
            if self.params.order_loss != 'not' or self.params.use_self != 'not':
                self.att_decoding_weight = nn.Parameter(torch.randn(self.params.num_rels, self.params.num_gcn_layers + 2, self.params.rel_emb_dim))
                init.xavier_uniform_(self.att_decoding_weight)

            self.att_W = nn.Linear(1, self.params.rel_emb_dim, bias = False)
            self.att_bias = nn.Parameter(torch.empty(1, self.params.rel_emb_dim))
            self.att_miss_bias = nn.Parameter(torch.empty(1, self.params.rel_emb_dim))
            init.xavier_uniform_(self.att_bias)
            init.xavier_uniform_(self.att_miss_bias)
            if self.params.input_feature == 'rra':
                self.feat_layer = nn.Linear(self.params.rel_emb_dim * 2 + (self.params.hop + 1) * 2, self.params.rel_emb_dim)
            
            elif self.params.input_feature == 'ra':
                self.feat_layer = nn.Linear(self.params.rel_emb_dim + (self.params.hop + 1) * 2, self.params.rel_emb_dim)            
            else:
                raise Exception("Not allowed input feature")
            
            self.sigmoid = nn.Sigmoid()
            self.dropout_num = nn.Dropout(self.params.dropout_num) 
            self.dropout = nn.Dropout(self.params.dropout) 
            
        elif self.params.input_feature == 'literalE':
            self.emb_num_lit = Gate(self.params.emb_dim * self.params.num_gcn_layers + self.params.num_atts, self.params.emb_dim * self.params.num_gcn_layers)
        else:
            pass

    def forward(self, data, use_self = False):

        g, rel_labels = data
        head_ids = (g.ndata['id'] == 1).nonzero().squeeze(1)
        tail_ids = (g.ndata['id'] == 2).nonzero().squeeze(1)

        if self.params.input_feature == 'numeric':
            g.ndata['feat'] = torch.cat([g.ndata['feat'], g.ndata['attribute']], dim = 1)
        elif self.params.input_feature == 'rra':

            g.ndata['a'], _ = self.ra_encoder(g)
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

            g.ndata['feat'] = self.dropout_num(self.sigmoid(self.feat_layer(new_feat)))

        elif self.params.input_feature == 'ra':
            g.ndata['a'], _ = self.ra_encoder(g)
            h_att, t_att = g.ndata['a'][head_ids], g.ndata['a'][tail_ids]
            new_feat = torch.cat([g.ndata['feat'], g.ndata['a']], dim = 1)
            g.ndata['feat'] = self.dropout_num(self.sigmoid(self.feat_layer(new_feat)))
        
        else:
            pass

        g.ndata['h'] = self.gnn(g)
        if self.params.input_feature == 'literalE':
            g.ndata['repr'] = self.emb_num_lit(g.ndata['repr'].view(-1, self.params.num_gcn_layers * self.params.emb_dim), g.ndata['attribute'])

        g_out = mean_nodes(g, 'repr')

        head_embs = g.ndata['repr'][head_ids]

        tail_embs = g.ndata['repr'][tail_ids]


        if self.params.add_ht_emb:
            g_rep = torch.cat([g_out.view(-1, self.params.num_gcn_layers * self.params.emb_dim),
                            head_embs.view(-1, self.params.num_gcn_layers * self.params.emb_dim),
                            tail_embs.view(-1, self.params.num_gcn_layers * self.params.emb_dim),
                            self.rel_emb(rel_labels)], dim=1)
        else:

            g_rep = torch.cat([g_out.view(-1, self.params.num_gcn_layers * self.params.emb_dim), self.rel_emb(rel_labels)], dim=1)

        output = self.fc_layer(g_rep)
        
        if self.params.order_loss != 'not':
                all_embs = torch.cat([(h_att - t_att).unsqueeze(1),
                                    (g.ndata['feat'][head_ids] - g.ndata['feat'][tail_ids]).unsqueeze(1),
                                    head_embs-tail_embs], dim = 1)

                d_sub_loss = (self.att_decoding_weight[rel_labels] * all_embs).sum(dim=-1)
        else:
            d_sub_loss = None

        if use_self:
            if self.params.use_self == 'self_1':
                with torch.no_grad():
                    head_order_score = (self.att_decoding_weight[rel_labels, 2:,:] * head_embs).sum(dim=-1, keepdim=True)
                    batch_order_score = (g.ndata['repr'].unsqueeze(0) * self.att_decoding_weight[rel_labels,2:,:].unsqueeze(1)).sum(dim=-1, keepdim=True)
                    abs_repr_cand = head_order_score.unsqueeze(1) - batch_order_score
                    # hard pos : smaller than real head e.g.) head height: 173 hard pos height : 171
                    hard_pos_mask = abs_repr_cand > 0 
                    hard_pos_cand = torch.where(hard_pos_mask, abs_repr_cand, torch.full_like(abs_repr_cand, float('inf'))).argmin(dim=1).squeeze(-1)
                    j_indices = torch.arange(hard_pos_cand.size(1)).unsqueeze(0).expand_as(hard_pos_cand)
                    hard_pos_tail = g.ndata['repr'][hard_pos_cand, j_indices]
                    # hard neg : larger than real head e.g.) head height: 173 hard neg height : 175
                    hard_neg_mask = abs_repr_cand < 0
                    hard_neg_cand = torch.where(hard_neg_mask, abs_repr_cand, torch.full_like(abs_repr_cand, float('-inf'))).argmax(dim=1).squeeze(-1)
                    hard_neg_tail = g.ndata['repr'][hard_neg_cand, j_indices]


                    tail_order_score = (self.att_decoding_weight[rel_labels, 2:,:] * tail_embs).sum(dim=-1, keepdim=True)
                    abs_repr_cand_for_tail = batch_order_score - tail_order_score.unsqueeze(1)
                    # hard pos for tail : larger than real tail e.g.) tail height: 173 hard pos for tail height : 175
                    hard_pos_mask_for_tail = abs_repr_cand_for_tail > 0
                    hard_pos_cand_for_tail = torch.where(hard_pos_mask_for_tail, abs_repr_cand_for_tail, torch.full_like(abs_repr_cand_for_tail, float('inf'))).argmin(dim=1).squeeze(-1)
                    j_indices = torch.arange(hard_pos_cand_for_tail.size(1)).unsqueeze(0).expand_as(hard_pos_cand_for_tail)
                    hard_pos_head = g.ndata['repr'][hard_pos_cand_for_tail, j_indices]
                    # hard neg for tail : smaller than real tail e.g.) tail height: 173 hard pos for tail height : 171                    
                    hard_neg_mask_for_tail = abs_repr_cand_for_tail < 0
                    hard_neg_cand_for_tail  = torch.where(hard_neg_mask_for_tail , abs_repr_cand_for_tail , torch.full_like(abs_repr_cand_for_tail, float('-inf'))).argmax(dim=1).squeeze(-1)
                    hard_neg_head = g.ndata['repr'][hard_neg_cand_for_tail, j_indices]

                    hard_pos_tail_coef = torch.rand(len(hard_pos_tail),1,1, device=hard_pos_tail.device)
                    hard_pos_sg_coef = torch.rand(len(hard_pos_tail),1,1, device=hard_pos_tail.device)
                    hard_neg_tail_coef = torch.rand(len(hard_neg_tail),1,1, device=hard_pos_tail.device)
                    hard_neg_sg_coef = torch.rand(len(hard_neg_tail),1,1, device=hard_pos_tail.device)



                hard_pos_tail = hard_pos_tail_coef * head_embs + (1-hard_pos_tail_coef) * hard_pos_tail
                hard_neg_tail = hard_neg_tail_coef * head_embs + (1-hard_neg_tail_coef) * hard_neg_tail

                hard_pos_head = hard_pos_tail_coef * tail_embs + (1-hard_pos_tail_coef) * hard_pos_head
                hard_neg_head = hard_neg_tail_coef * tail_embs + (1-hard_neg_tail_coef) * hard_neg_head


                hard_pos_sg = hard_pos_sg_coef * head_embs + (1-hard_pos_sg_coef) * hard_pos_tail
                hard_neg_sg = hard_neg_sg_coef * head_embs + (1-hard_neg_sg_coef) * hard_neg_tail

                hard_pos_sg_2 = hard_pos_sg_coef * tail_embs + (1-hard_pos_sg_coef) * hard_pos_head
                hard_neg_sg_2 = hard_neg_sg_coef * tail_embs + (1-hard_neg_sg_coef) * hard_neg_head


                
                hard_pos_rep = torch.cat([hard_pos_sg.view(-1, self.params.num_gcn_layers * self.params.emb_dim),
                            head_embs.view(-1, self.params.num_gcn_layers * self.params.emb_dim),
                            hard_pos_tail.view(-1, self.params.num_gcn_layers * self.params.emb_dim),
                            self.rel_emb(rel_labels)], dim=1)
                hard_neg_rep = torch.cat([hard_neg_sg.view(-1, self.params.num_gcn_layers * self.params.emb_dim),
                            head_embs.view(-1, self.params.num_gcn_layers * self.params.emb_dim),
                            hard_neg_tail.view(-1, self.params.num_gcn_layers * self.params.emb_dim),
                            self.rel_emb(rel_labels)], dim=1)

                hard_pos_rep_2 = torch.cat([hard_pos_sg_2.view(-1, self.params.num_gcn_layers * self.params.emb_dim),
                            hard_pos_head.view(-1, self.params.num_gcn_layers * self.params.emb_dim),
                            tail_embs.view(-1, self.params.num_gcn_layers * self.params.emb_dim),
                            self.rel_emb(rel_labels)], dim=1)
                hard_neg_rep_2 = torch.cat([hard_neg_sg_2.view(-1, self.params.num_gcn_layers * self.params.emb_dim),
                            hard_neg_head.view(-1, self.params.num_gcn_layers * self.params.emb_dim),
                            tail_embs.view(-1, self.params.num_gcn_layers * self.params.emb_dim),
                            self.rel_emb(rel_labels)], dim=1)

                hard_pos_output = (self.fc_layer(hard_pos_rep), self.fc_layer(hard_pos_rep_2))
                hard_neg_output = (self.fc_layer(hard_neg_rep), self.fc_layer(hard_neg_rep_2))

            elif self.params.use_self == 'self_2':
                    unique_rel_labels, unique_rel_idx = torch.unique(rel_labels, return_inverse = True)
                    mask_for_rel = g.ndata['r_label'].unsqueeze(0) == unique_rel_labels.unsqueeze(1)
                    expanded_tensor_size = (len(mask_for_rel),) + g.ndata['repr'].size()
                    mask_for_rel = mask_for_rel.unsqueeze(-1).unsqueeze(-1).expand(expanded_tensor_size)

                    in_batch_cand = torch.where(mask_for_rel, #mask_for_rel.unsqueeze(-1).unsqueeze(-1).expand(expanded_tensor_size), 
                                                g.ndata['repr'].repeat(expanded_tensor_size[0],1,1,1), 
                                                torch.full(expanded_tensor_size, 0.0).to(g.ndata['repr'].device))

                    head_order_score = (self.att_decoding_weight[rel_labels, 2:,:] * head_embs).sum(dim=-1, keepdim=True)

                    batch_order_score = (in_batch_cand[unique_rel_idx] * self.att_decoding_weight[rel_labels,2:,:].unsqueeze(1)).sum(dim=-1, keepdim=True)
                    batch_order_score = torch.where(torch.any(mask_for_rel[unique_rel_idx], dim=-1, keepdim=True), batch_order_score, torch.full_like(batch_order_score, float('inf')))


                    abs_repr_cand = head_order_score.unsqueeze(1) - batch_order_score

                    hard_pos_mask = abs_repr_cand > 0
                    hard_pos_cand = torch.where(hard_pos_mask, abs_repr_cand, torch.full_like(abs_repr_cand, float('inf'))).argmin(dim=1).squeeze(-1)

                    j_indices = torch.arange(hard_pos_cand.size(1)).unsqueeze(0).expand_as(hard_pos_cand)
                    hard_pos_tail = g.ndata['repr'][hard_pos_cand, j_indices]
                    hard_neg_mask = abs_repr_cand < 0
                    hard_neg_cand = torch.where(hard_neg_mask, abs_repr_cand, torch.full_like(abs_repr_cand, float('-inf'))).argmax(dim=1).squeeze(-1)
                    hard_neg_tail = g.ndata['repr'][hard_neg_cand, j_indices]
                    


                    tail_order_score = (self.att_decoding_weight[rel_labels, 2:,:] * tail_embs).sum(dim=-1, keepdim=True)
                    abs_repr_cand_for_tail = batch_order_score - tail_order_score.unsqueeze(1)
                    hard_pos_mask_for_tail = abs_repr_cand_for_tail > 0
                    hard_pos_cand_for_tail = torch.where(hard_pos_mask_for_tail, abs_repr_cand_for_tail, torch.full_like(abs_repr_cand_for_tail, float('inf'))).argmin(dim=1).squeeze(-1)
                    j_indices = torch.arange(hard_pos_cand_for_tail.size(1)).unsqueeze(0).expand_as(hard_pos_cand_for_tail)
                    hard_pos_head = g.ndata['repr'][hard_pos_cand_for_tail, j_indices]
                    hard_neg_mask_for_tail = abs_repr_cand_for_tail < 0
                    hard_neg_cand_for_tail  = torch.where(hard_neg_mask_for_tail , abs_repr_cand_for_tail , torch.full_like(abs_repr_cand_for_tail, float('-inf'))).argmax(dim=1).squeeze(-1)
                    hard_neg_head = g.ndata['repr'][hard_neg_cand_for_tail, j_indices]

                    
                    
                    hard_pos_tail_coef = torch.rand(len(hard_pos_tail),1,1, device=hard_pos_tail.device)
                    hard_pos_sg_coef = torch.rand(len(hard_pos_tail),1,1, device=hard_pos_tail.device)

                    hard_neg_tail_coef = torch.rand(len(hard_neg_tail),1,1, device=hard_pos_tail.device)
                    hard_neg_sg_coef = torch.rand(len(hard_neg_tail),1,1, device=hard_pos_tail.device)

                    hard_pos_tail = hard_pos_tail_coef * head_embs + (1-hard_pos_tail_coef) * hard_pos_tail
                    hard_neg_tail = hard_neg_tail_coef * head_embs + (1-hard_neg_tail_coef) * hard_neg_tail

                    hard_pos_head = hard_pos_tail_coef * tail_embs + (1-hard_pos_tail_coef) * hard_pos_head
                    hard_neg_head = hard_neg_tail_coef * tail_embs + (1-hard_neg_tail_coef) * hard_neg_head


                    hard_pos_sg = hard_pos_sg_coef * head_embs + (1-hard_pos_sg_coef) * hard_pos_tail
                    hard_neg_sg = hard_neg_sg_coef * head_embs + (1-hard_neg_sg_coef) * hard_neg_tail

                    hard_pos_sg_2 = hard_pos_sg_coef * tail_embs + (1-hard_pos_sg_coef) * hard_pos_head
                    hard_neg_sg_2 = hard_neg_sg_coef * tail_embs + (1-hard_neg_sg_coef) * hard_neg_head


                    
                    hard_pos_rep = torch.cat([hard_pos_sg.view(-1, self.params.num_gcn_layers * self.params.emb_dim),
                                head_embs.view(-1, self.params.num_gcn_layers * self.params.emb_dim),
                                hard_pos_tail.view(-1, self.params.num_gcn_layers * self.params.emb_dim),
                                self.rel_emb(rel_labels)], dim=1)
                    hard_neg_rep = torch.cat([hard_neg_sg.view(-1, self.params.num_gcn_layers * self.params.emb_dim),
                                head_embs.view(-1, self.params.num_gcn_layers * self.params.emb_dim),
                                hard_neg_tail.view(-1, self.params.num_gcn_layers * self.params.emb_dim),
                                self.rel_emb(rel_labels)], dim=1)

                    hard_pos_rep_2 = torch.cat([hard_pos_sg_2.view(-1, self.params.num_gcn_layers * self.params.emb_dim),
                                hard_pos_head.view(-1, self.params.num_gcn_layers * self.params.emb_dim),
                                tail_embs.view(-1, self.params.num_gcn_layers * self.params.emb_dim),
                                self.rel_emb(rel_labels)], dim=1)
                    hard_neg_rep_2 = torch.cat([hard_neg_sg_2.view(-1, self.params.num_gcn_layers * self.params.emb_dim),
                                hard_neg_head.view(-1, self.params.num_gcn_layers * self.params.emb_dim),
                                tail_embs.view(-1, self.params.num_gcn_layers * self.params.emb_dim),
                                self.rel_emb(rel_labels)], dim=1)

                    hard_pos_output = (self.fc_layer(hard_pos_rep), self.fc_layer(hard_pos_rep_2))
                    hard_neg_output = (self.fc_layer(hard_neg_rep), self.fc_layer(hard_neg_rep_2))
            else:
                raise NotImplementedError
        else:
            hard_pos_output = None
            hard_neg_output = None

        return output, d_sub_loss, hard_pos_output, hard_neg_output


    def forward_w_self_term_rel(self, data, use_self = False):
        ((g, rel_labels), all_triples, all_lits, missing_mask, all_rels) = data

        head_ids = (g.ndata['id'] == 1).nonzero().squeeze(1)
        tail_ids = (g.ndata['id'] == 2).nonzero().squeeze(1)


        g.ndata['a'], _ = self.ra_encoder(g)
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

        g.ndata['feat'] = self.dropout_num(self.sigmoid(self.feat_layer(new_feat)))
        #g.ndata['feat'] = self.dropout_num(self.activation(self.feat_layer(new_feat)))
        g.ndata['h'] = self.gnn(g)

        g_out = mean_nodes(g, 'repr')

        head_embs = g.ndata['repr'][head_ids]

        tail_embs = g.ndata['repr'][tail_ids]


        if self.params.add_ht_emb:

            g_rep = torch.cat([g_out.view(-1, self.params.num_gcn_layers * self.params.emb_dim),
                            head_embs.view(-1, self.params.num_gcn_layers * self.params.emb_dim),
                            tail_embs.view(-1, self.params.num_gcn_layers * self.params.emb_dim),
                            self.rel_emb(rel_labels)], dim=1)
        else:

            g_rep = torch.cat([g_out.view(-1, self.params.num_gcn_layers * self.params.emb_dim), self.rel_emb(rel_labels), sem_tail_embs], dim=1)

        output = self.fc_layer(g_rep)

        if self.params.input_feature == 'layer_wise_comp':
            if self.params.order_embs == 'all':
                all_embs = torch.cat([(h_att - t_att).unsqueeze(1),
                                    (g.ndata['feat'][head_ids] - g.ndata['feat'][tail_ids]).unsqueeze(1),
                                    head_embs-tail_embs], dim = 1)

                d_sub_loss = (self.att_decoding_weight[rel_labels] * all_embs).sum(dim=-1)#.sum(dim=-1,keepdim=True)

                if use_self:
                    unique_rel_labels, unique_rel_idx = torch.unique(rel_labels, return_inverse = True)
                    mask_for_rel = g.ndata['r_label'].unsqueeze(0) == unique_rel_labels.unsqueeze(1)
                    expanded_tensor_size = (len(mask_for_rel),) + g.ndata['repr'].size()
                    mask_for_rel = mask_for_rel.unsqueeze(-1).unsqueeze(-1).expand(expanded_tensor_size)

                    in_batch_cand = torch.where(mask_for_rel, #mask_for_rel.unsqueeze(-1).unsqueeze(-1).expand(expanded_tensor_size), 
                                                g.ndata['repr'].repeat(expanded_tensor_size[0],1,1,1), 
                                                torch.full(expanded_tensor_size, 0.0).to(g.ndata['repr'].device))

                    head_order_score = (self.att_decoding_weight[rel_labels, 2:,:] * head_embs).sum(dim=-1, keepdim=True)

                    batch_order_score = (in_batch_cand[unique_rel_idx] * self.att_decoding_weight[rel_labels,2:,:].unsqueeze(1)).sum(dim=-1, keepdim=True)
                    batch_order_score = torch.where(torch.any(mask_for_rel[unique_rel_idx], dim=-1, keepdim=True), batch_order_score, torch.full_like(batch_order_score, float('inf')))


                    abs_repr_cand = head_order_score.unsqueeze(1) - batch_order_score

                    hard_pos_mask = abs_repr_cand > 0
                    hard_pos_cand = torch.where(hard_pos_mask, abs_repr_cand, torch.full_like(abs_repr_cand, float('inf'))).argmin(dim=1).squeeze(-1)

                    j_indices = torch.arange(hard_pos_cand.size(1)).unsqueeze(0).expand_as(hard_pos_cand)
                    hard_pos_tail = g.ndata['repr'][hard_pos_cand, j_indices]
                    hard_neg_mask = abs_repr_cand < 0
                    hard_neg_cand = torch.where(hard_neg_mask, abs_repr_cand, torch.full_like(abs_repr_cand, float('-inf'))).argmax(dim=1).squeeze(-1)
                    hard_neg_tail = g.ndata['repr'][hard_neg_cand, j_indices]
                    


                    tail_order_score = (self.att_decoding_weight[rel_labels, 2:,:] * tail_embs).sum(dim=-1, keepdim=True)
                    abs_repr_cand_for_tail = batch_order_score - tail_order_score.unsqueeze(1)
                    hard_pos_mask_for_tail = abs_repr_cand_for_tail > 0
                    hard_pos_cand_for_tail = torch.where(hard_pos_mask_for_tail, abs_repr_cand_for_tail, torch.full_like(abs_repr_cand_for_tail, float('inf'))).argmin(dim=1).squeeze(-1)
                    j_indices = torch.arange(hard_pos_cand_for_tail.size(1)).unsqueeze(0).expand_as(hard_pos_cand_for_tail)
                    hard_pos_head = g.ndata['repr'][hard_pos_cand_for_tail, j_indices]
                    hard_neg_mask_for_tail = abs_repr_cand_for_tail < 0
                    hard_neg_cand_for_tail  = torch.where(hard_neg_mask_for_tail , abs_repr_cand_for_tail , torch.full_like(abs_repr_cand_for_tail, float('-inf'))).argmax(dim=1).squeeze(-1)
                    hard_neg_head = g.ndata['repr'][hard_neg_cand_for_tail, j_indices]

                    
                    
                    hard_pos_tail_coef = torch.rand(len(hard_pos_tail),1,1, device=hard_pos_tail.device)
                    hard_pos_sg_coef = torch.rand(len(hard_pos_tail),1,1, device=hard_pos_tail.device)

                    hard_neg_tail_coef = torch.rand(len(hard_neg_tail),1,1, device=hard_pos_tail.device)
                    hard_neg_sg_coef = torch.rand(len(hard_neg_tail),1,1, device=hard_pos_tail.device)

                    hard_pos_tail = hard_pos_tail_coef * head_embs + (1-hard_pos_tail_coef) * hard_pos_tail
                    hard_neg_tail = hard_neg_tail_coef * head_embs + (1-hard_neg_tail_coef) * hard_neg_tail

                    hard_pos_head = hard_pos_tail_coef * tail_embs + (1-hard_pos_tail_coef) * hard_pos_head
                    hard_neg_head = hard_neg_tail_coef * tail_embs + (1-hard_neg_tail_coef) * hard_neg_head


                    hard_pos_sg = hard_pos_sg_coef * head_embs + (1-hard_pos_sg_coef) * hard_pos_tail
                    hard_neg_sg = hard_neg_sg_coef * head_embs + (1-hard_neg_sg_coef) * hard_neg_tail

                    hard_pos_sg_2 = hard_pos_sg_coef * tail_embs + (1-hard_pos_sg_coef) * hard_pos_head
                    hard_neg_sg_2 = hard_neg_sg_coef * tail_embs + (1-hard_neg_sg_coef) * hard_neg_head


                    
                    hard_pos_rep = torch.cat([hard_pos_sg.view(-1, self.params.num_gcn_layers * self.params.emb_dim),
                                head_embs.view(-1, self.params.num_gcn_layers * self.params.emb_dim),
                                hard_pos_tail.view(-1, self.params.num_gcn_layers * self.params.emb_dim),
                                self.rel_emb(rel_labels)], dim=1)
                    hard_neg_rep = torch.cat([hard_neg_sg.view(-1, self.params.num_gcn_layers * self.params.emb_dim),
                                head_embs.view(-1, self.params.num_gcn_layers * self.params.emb_dim),
                                hard_neg_tail.view(-1, self.params.num_gcn_layers * self.params.emb_dim),
                                self.rel_emb(rel_labels)], dim=1)

                    hard_pos_rep_2 = torch.cat([hard_pos_sg_2.view(-1, self.params.num_gcn_layers * self.params.emb_dim),
                                hard_pos_head.view(-1, self.params.num_gcn_layers * self.params.emb_dim),
                                tail_embs.view(-1, self.params.num_gcn_layers * self.params.emb_dim),
                                self.rel_emb(rel_labels)], dim=1)
                    hard_neg_rep_2 = torch.cat([hard_neg_sg_2.view(-1, self.params.num_gcn_layers * self.params.emb_dim),
                                hard_neg_head.view(-1, self.params.num_gcn_layers * self.params.emb_dim),
                                tail_embs.view(-1, self.params.num_gcn_layers * self.params.emb_dim),
                                self.rel_emb(rel_labels)], dim=1)



                    #hard_pos_output = self.fc_layer(hard_pos_rep)
                    #hard_neg_output = self.fc_layer(hard_neg_rep)
                    #hard_pos_output_2 = self.fc_layer(hard_pos_rep_2)
                    #hard_neg_output_2 = self.fc_layer(hard_neg_rep_2)
                    hard_pos_output = (self.fc_layer(hard_pos_rep), self.fc_layer(hard_pos_rep_2))
                    hard_neg_output = (self.fc_layer(hard_neg_rep), self.fc_layer(hard_neg_rep_2))
                else:
                    hard_pos_output = None
                    hard_neg_output = None



            elif self.params.order_embs == 'gnn':
                d_sub_loss = (self.att_decoding_weight[rel_labels] * head_embs-tail_embs).sum(dim=-1).sum(dim=-1, keepdim=True)
            else:
                raise Exception("order embs error")

            
        else:
            raise Exception("Not layer wise comp")
        
        return output, d_sub_loss, hard_pos_output, hard_neg_output
    

    def forward_w_self_term(self, data, use_self = False):
        ((g, rel_labels), all_triples, all_lits, missing_mask, all_rels) = data

        head_ids = (g.ndata['id'] == 1).nonzero().squeeze(1)
        tail_ids = (g.ndata['id'] == 2).nonzero().squeeze(1)


        g.ndata['a'], _ = self.ra_encoder(g)
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

        g.ndata['feat'] = self.dropout_num(self.sigmoid(self.feat_layer(new_feat)))
        #g.ndata['feat'] = self.dropout_num(self.activation(self.feat_layer(new_feat)))
        g.ndata['h'] = self.gnn(g)

        g_out = mean_nodes(g, 'repr')

        head_embs = g.ndata['repr'][head_ids]

        tail_embs = g.ndata['repr'][tail_ids]


        if self.params.add_ht_emb:
            g_rep = torch.cat([g_out.view(-1, self.params.num_gcn_layers * self.params.emb_dim),
                            head_embs.view(-1, self.params.num_gcn_layers * self.params.emb_dim),
                            tail_embs.view(-1, self.params.num_gcn_layers * self.params.emb_dim),
                            self.rel_emb(rel_labels)], dim=1)
        else:

            g_rep = torch.cat([g_out.view(-1, self.params.num_gcn_layers * self.params.emb_dim), self.rel_emb(rel_labels)], dim=1)

        output = self.fc_layer(g_rep)

        if self.params.input_feature == 'layer_wise_comp':
            if self.params.order_embs == 'all':
                all_embs = torch.cat([(h_att - t_att).unsqueeze(1),
                                    (g.ndata['feat'][head_ids] - g.ndata['feat'][tail_ids]).unsqueeze(1),
                                    head_embs-tail_embs], dim = 1)

                d_sub_loss = (self.att_decoding_weight[rel_labels] * all_embs).sum(dim=-1)#.sum(dim=-1,keepdim=True)
                if use_self:
                    with torch.no_grad():
                        head_order_score = (self.att_decoding_weight[rel_labels, 2:,:] * head_embs).sum(dim=-1, keepdim=True)
                        batch_order_score = (g.ndata['repr'].unsqueeze(0) * self.att_decoding_weight[rel_labels,2:,:].unsqueeze(1)).sum(dim=-1, keepdim=True)
                        abs_repr_cand = head_order_score.unsqueeze(1) - batch_order_score
                        hard_pos_mask = abs_repr_cand > 0
                        hard_pos_cand = torch.where(hard_pos_mask, abs_repr_cand, torch.full_like(abs_repr_cand, float('inf'))).argmin(dim=1).squeeze(-1)
                        j_indices = torch.arange(hard_pos_cand.size(1)).unsqueeze(0).expand_as(hard_pos_cand)
                        hard_pos_tail = g.ndata['repr'][hard_pos_cand, j_indices]
                        hard_neg_mask = abs_repr_cand < 0
                        hard_neg_cand = torch.where(hard_neg_mask, abs_repr_cand, torch.full_like(abs_repr_cand, float('-inf'))).argmax(dim=1).squeeze(-1)
                        hard_neg_tail = g.ndata['repr'][hard_neg_cand, j_indices]


                        tail_order_score = (self.att_decoding_weight[rel_labels, 2:,:] * tail_embs).sum(dim=-1, keepdim=True)
                        abs_repr_cand_for_tail = batch_order_score - tail_order_score.unsqueeze(1)
                        hard_pos_mask_for_tail = abs_repr_cand_for_tail > 0
                        hard_pos_cand_for_tail = torch.where(hard_pos_mask_for_tail, abs_repr_cand_for_tail, torch.full_like(abs_repr_cand_for_tail, float('inf'))).argmin(dim=1).squeeze(-1)
                        j_indices = torch.arange(hard_pos_cand_for_tail.size(1)).unsqueeze(0).expand_as(hard_pos_cand_for_tail)
                        hard_pos_head = g.ndata['repr'][hard_pos_cand_for_tail, j_indices]
                        hard_neg_mask_for_tail = abs_repr_cand_for_tail < 0
                        hard_neg_cand_for_tail  = torch.where(hard_neg_mask_for_tail , abs_repr_cand_for_tail , torch.full_like(abs_repr_cand_for_tail, float('-inf'))).argmax(dim=1).squeeze(-1)
                        hard_neg_head = g.ndata['repr'][hard_neg_cand_for_tail, j_indices]

                        hard_pos_tail_coef = torch.rand(len(hard_pos_tail),1,1, device=hard_pos_tail.device)
                        hard_pos_sg_coef = torch.rand(len(hard_pos_tail),1,1, device=hard_pos_tail.device)
                        hard_neg_tail_coef = torch.rand(len(hard_neg_tail),1,1, device=hard_pos_tail.device)
                        hard_neg_sg_coef = torch.rand(len(hard_neg_tail),1,1, device=hard_pos_tail.device)



                    hard_pos_tail = hard_pos_tail_coef * head_embs + (1-hard_pos_tail_coef) * hard_pos_tail
                    hard_neg_tail = hard_neg_tail_coef * head_embs + (1-hard_neg_tail_coef) * hard_neg_tail

                    hard_pos_head = hard_pos_tail_coef * tail_embs + (1-hard_pos_tail_coef) * hard_pos_head
                    hard_neg_head = hard_neg_tail_coef * tail_embs + (1-hard_neg_tail_coef) * hard_neg_head


                    hard_pos_sg = hard_pos_sg_coef * head_embs + (1-hard_pos_sg_coef) * hard_pos_tail
                    hard_neg_sg = hard_neg_sg_coef * head_embs + (1-hard_neg_sg_coef) * hard_neg_tail

                    hard_pos_sg_2 = hard_pos_sg_coef * tail_embs + (1-hard_pos_sg_coef) * hard_pos_head
                    hard_neg_sg_2 = hard_neg_sg_coef * tail_embs + (1-hard_neg_sg_coef) * hard_neg_head


                    
                    hard_pos_rep = torch.cat([hard_pos_sg.view(-1, self.params.num_gcn_layers * self.params.emb_dim),
                                head_embs.view(-1, self.params.num_gcn_layers * self.params.emb_dim),
                                hard_pos_tail.view(-1, self.params.num_gcn_layers * self.params.emb_dim),
                                self.rel_emb(rel_labels)], dim=1)
                    hard_neg_rep = torch.cat([hard_neg_sg.view(-1, self.params.num_gcn_layers * self.params.emb_dim),
                                head_embs.view(-1, self.params.num_gcn_layers * self.params.emb_dim),
                                hard_neg_tail.view(-1, self.params.num_gcn_layers * self.params.emb_dim),
                                self.rel_emb(rel_labels)], dim=1)

                    hard_pos_rep_2 = torch.cat([hard_pos_sg_2.view(-1, self.params.num_gcn_layers * self.params.emb_dim),
                                hard_pos_head.view(-1, self.params.num_gcn_layers * self.params.emb_dim),
                                tail_embs.view(-1, self.params.num_gcn_layers * self.params.emb_dim),
                                self.rel_emb(rel_labels)], dim=1)
                    hard_neg_rep_2 = torch.cat([hard_neg_sg_2.view(-1, self.params.num_gcn_layers * self.params.emb_dim),
                                hard_neg_head.view(-1, self.params.num_gcn_layers * self.params.emb_dim),
                                tail_embs.view(-1, self.params.num_gcn_layers * self.params.emb_dim),
                                self.rel_emb(rel_labels)], dim=1)



                    #hard_pos_output = self.fc_layer(hard_pos_rep)
                    #hard_neg_output = self.fc_layer(hard_neg_rep)
                    #hard_pos_output_2 = self.fc_layer(hard_pos_rep_2)
                    #hard_neg_output_2 = self.fc_layer(hard_neg_rep_2)
                    hard_pos_output = (self.fc_layer(hard_pos_rep), self.fc_layer(hard_pos_rep_2))
                    hard_neg_output = (self.fc_layer(hard_neg_rep), self.fc_layer(hard_neg_rep_2))
        
                else:
                    hard_pos_output = None
                    hard_neg_output = None

            elif self.params.order_embs == 'gnn':
                d_sub_loss = (self.att_decoding_weight[rel_labels] * head_embs-tail_embs).sum(dim=-1).sum(dim=-1, keepdim=True)
            else:
                raise Exception("order embs error")

            
        else:
            raise Exception("Not layer wise comp")
        
        return output, d_sub_loss, hard_pos_output, hard_neg_output

    
    def ra_encoder(self, g):

        num_lit = g.ndata['attribute'].unsqueeze(-1)
        num_lit = self.att_W(num_lit)
        #expanded_mask = g.ndata['mask'].unsqueeze(-1).expand(-1, -1, num_lit.size(-1))

        num_lit[g.ndata['mask']] += self.att_bias
        num_lit[g.ndata['mask'] == 0] += self.att_miss_bias

        att_emb = self.sigmoid(num_lit * self.att_emb.weight.unsqueeze(0))
            
            
        target_rels = g.ndata['r_label']  

        target_rels_emb = self.rel_emb(target_rels).unsqueeze(1) 

        #scores = torch.matmul(att_emb, target_rels_emb).squeeze(2) 
        #atts = self.softmax(scores)

        att_sem_feats, atts = self.MHA(query = target_rels_emb.transpose(0,1), key = att_emb.transpose(0,1), value = att_emb.transpose(0,1))

        att_sem_feats = att_sem_feats.transpose(0,1).squeeze(1)

        #att_sem_feats = self.w_rel2ent(torch.matmul(atts.unsqueeze(1), att_emb).squeeze(1))
        #att_sem_feats = self.dropout_num(self.sigmoid(self.w_rel2ent(att_sem_feats)))

        return att_sem_feats, atts
    
class Gate(nn.Module):

    def __init__(self,
                    input_size,
                    output_size,
                    gate_activation=torch.sigmoid):
        super(Gate, self).__init__()
        self.output_size = output_size

        self.gate_activation = gate_activation
        self.g = nn.Linear(input_size, output_size)
        self.g1 = nn.Linear(output_size, output_size, bias=False)
        self.g2 = nn.Linear(input_size-output_size, output_size, bias=False)
        self.gate_bias = nn.Parameter(torch.zeros(output_size))

    def forward(self, x_ent, x_lit):
        x = torch.cat([x_ent, x_lit], x_lit.ndimension()-1)
        g_embedded = torch.tanh(self.g(x))
        gate = self.gate_activation(self.g1(x_ent) + self.g2(x_lit) + self.gate_bias)
        output = (1-gate) * x_ent + gate * g_embedded

        return output
    

def sample_true_indices(bool_tensor, k):
    """
    각 행별로 True인 요소들 중 무작위로 k개의 인덱스를 샘플링하는 함수.

    Parameters:
    bool_tensor (torch.Tensor): n x d 크기의 boolean 텐서
    k (int): 각 행에서 샘플링할 인덱스의 개수

    Returns:
    torch.Tensor: n x k 크기의 텐서로, 각 행에서 샘플링된 인덱스
    """
    n, d = bool_tensor.shape
    sampled_indices = torch.full((n, k), -1, dtype=torch.long)  # -1로 초기화된 n*k 크기의 텐서

    for i in range(n):
        true_indices = torch.nonzero(bool_tensor[i], as_tuple=True)[0]  # True인 인덱스 찾기
        if len(true_indices) >= k:
            sampled_indices[i] = true_indices[torch.randperm(len(true_indices))[:k]]
        else:
            # True인 요소가 k개보다 적을 경우, 남은 인덱스를 -1로 유지하거나 다른 처리 가능
            sampled_indices[i, :len(true_indices)] = true_indices
            sampled_indices[i, len(true_indices):] = -1  # 예시: 부족한 부분 -1로 채우기

    return sampled_indices