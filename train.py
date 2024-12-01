import os
import argparse
import logging
import torch
from scipy.sparse import SparseEfficiencyWarning

from subgraph_extraction.datasets import SubgraphDataset, generate_subgraph_datasets
from utils.initialization_utils import initialize_experiment, initialize_model
from utils.graph_utils import collate_dgl, move_batch_to_device_dgl

from model.dgl.graph_classifier import GraphClassifier as dgl_model

from managers.evaluator import Evaluator
from managers.trainer import Trainer

from warnings import simplefilter
import numpy as np
import json
import random
import torch.backends.cudnn as cudnn


def main(params):
    simplefilter(action='ignore', category=UserWarning)
    simplefilter(action='ignore', category=SparseEfficiencyWarning)

    params.db_path = os.path.join(params.main_dir, f'data/{params.dataset}/subgraphs_en_{params.enclosing_sub_graph}_neg_{params.num_neg_samples_per_link}_hop_{params.hop}')

    if not os.path.isdir(params.db_path):
        generate_subgraph_datasets(params)

    train = SubgraphDataset(params.db_path, 'train_pos', 'train_neg', params.file_paths,
                            add_traspose_rels=params.add_traspose_rels,
                            num_neg_samples_per_link=params.num_neg_samples_per_link,
                            use_kge_embeddings=params.use_kge_embeddings, dataset=params.dataset,
                            kge_model=params.kge_model, file_name=params.train_file, use_numeric = params.use_numeric)
    valid = SubgraphDataset(params.db_path, 'valid_pos', 'valid_neg', params.file_paths,
                            add_traspose_rels=params.add_traspose_rels,
                            num_neg_samples_per_link=params.num_neg_samples_per_link,
                            use_kge_embeddings=params.use_kge_embeddings, dataset=params.dataset,
                            kge_model=params.kge_model, file_name=params.valid_file, use_numeric = params.use_numeric)

    params.eval_every_iter = int(np.ceil(len(train) / params.batch_size))

    params.num_rels = train.num_rels
    params.num_atts = train.num_atts
    params.aug_num_rels = train.aug_num_rels
    if not params.use_numeric:
        params.inp_dim = train.n_feat_dim

    else:
        if params.input_feature == 'numeric':
            params.inp_dim = params.num_atts + 2 * (params.hop + 1)#train.n_feat_dim
        elif params.input_feature in ['rra', 'ra']:
            params.inp_dim = params.rel_emb_dim
        elif params.input_feature == 'literalE':
            params.inp_dim = 2 * (params.hop + 1) #train.n_feat_dim
        else:
            raise NotImplementedError
        train.attribute_normalization(density = params.attribute_density, main_dir = params.main_dir, experiment_name = params.experiment_name)
        min_max_path = os.path.join(params.main_dir, f'experiments/{params.experiment_name}/min_max_lit.npy')
        missing_path = os.path.join(params.main_dir, f'experiments/{params.experiment_name}/missing_mask.npy')
        with open(min_max_path, 'rb') as f:
            min_max_lit = np.load(f, allow_pickle=True).item()
        with open(missing_path, 'rb') as f:
            missing_mask = np.load(f, allow_pickle=True)
        valid.attribute_normalization(density = params.attribute_density, saved_max_lit = min_max_lit['max_lit'], saved_min_lit = min_max_lit['min_lit'], saved_missing_mask = missing_mask)

    # Log the max label value to save it in the model. This will be used to cap the labels generated on test set.
    params.max_label_value = train.max_n_label

    graph_classifier = initialize_model(params, dgl_model, params.load_model)

    logging.info(f"Device: {params.device}")
    if not params.use_numeric:
        logging.info(f"Input dim : {params.inp_dim}, # Relations : {params.num_rels}, # Augmented relations : {params.aug_num_rels}")
    else:
        logging.info(f"Input dim : {params.inp_dim}, # Relations : {params.num_rels}, # Augmented relations : {params.aug_num_rels}, # Atrributes : {params.num_atts}")
    valid_evaluator = Evaluator(params, graph_classifier, valid)

    trainer = Trainer(params, graph_classifier, train, valid_evaluator)

    logging.info('Starting training with full batch...')

    trainer.train()


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description='TransE model')

    # Experiment setup params
    parser.add_argument("--experiment_name", "-e", type=str, default="default",
                        help="A folder with this name would be created to dump saved models and log files")
    parser.add_argument("--dataset", "-d", type=str,
                        help="Dataset string")
    parser.add_argument("--gpu", type=int, default=0,
                        help="Which GPU to use?")
    parser.add_argument('--disable_cuda', action='store_true',
                        help='Disable CUDA')
    parser.add_argument('--load_model', action='store_true',
                        help='Load existing model?')
    parser.add_argument("--train_file", "-tf", type=str, default="train",
                        help="Name of file containing training triplets")
    parser.add_argument("--valid_file", "-vf", type=str, default="valid",
                        help="Name of file containing validation triplets")
    parser.add_argument("--train_neg_file", "-tnf", type=str, default="train_neg",
                        help="Name of file containing training triplets")
    parser.add_argument("--valid_neg_file", "-vnf", type=str, default="valid_neg",
                        help="Name of file containing validation triplets")
    parser.add_argument("--literals_file", "-lf", type=str, default="literals/numerical_literals",
                        help="Name of file containing literals")

    # Training regime params
    parser.add_argument("--num_epochs", "-ne", type=int, default=70,
                        help="Learning rate of the optimizer")
    parser.add_argument("--eval_every", type=int, default=3,
                        help="Interval of epochs to evaluate the model?")
    parser.add_argument("--eval_every_iter", type=int, default=455,
                        help="Interval of iterations to evaluate the model?")
    parser.add_argument("--save_every", type=int, default=10,
                        help="Interval of epochs to save a checkpoint of the model?")
    parser.add_argument("--early_stop", type=int, default=100,
                        help="Early stopping patience")
    parser.add_argument("--optimizer", type=str, default="Adam",
                        help="Which optimizer to use?")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="Learning rate of the optimizer")
    parser.add_argument("--clip", type=int, default=1000,
                        help="Maximum gradient norm allowed")
    parser.add_argument("--l2", type=float, default=5e-4,
                        help="Regularization constant for GNN weights")
    parser.add_argument("--margin", type=float, default=10,
                        help="The margin between positive and negative samples in the max-margin loss")

    # Data processing pipeline params
    parser.add_argument("--max_links", type=int, default=1000000,
                        help="Set maximum number of train links (to fit into memory)")
    parser.add_argument("--hop", type=int, default=3,
                        help="Enclosing subgraph hop number")
    parser.add_argument("--max_nodes_per_hop", "-max_h", type=int, default=None,
                        help="if > 0, upper bound the # nodes per hop by subsampling")
    parser.add_argument("--use_kge_embeddings", "-kge", type=bool, default=False,
                        help='whether to use pretrained KGE embeddings')
    parser.add_argument("--kge_model", type=str, default="TransE",
                        help="Which KGE model to load entity embeddings from")
    parser.add_argument('--model_type', '-m', type=str, choices=['ssp', 'dgl'], default='dgl',
                        help='what format to store subgraphs in for model')
    parser.add_argument('--constrained_neg_prob', '-cn', type=float, default=0.0,
                        help='with what probability to sample constrained heads/tails while neg sampling')
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size")
    parser.add_argument("--num_neg_samples_per_link", '-neg', type=int, default=1,
                        help="Number of negative examples to sample per positive link")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="Number of dataloading processes")
    parser.add_argument('--add_traspose_rels', '-tr', type=bool, default=False,
                        help='whether to append adj matrix list with symmetric relations')
    parser.add_argument('--enclosing_sub_graph', '-en', type=bool, default=True,
                        help='whether to only consider enclosing subgraph')

    # Model params
    parser.add_argument("--rel_emb_dim", "-r_dim", type=int, default=32,
                        help="Relation embedding size")
    parser.add_argument("--attn_rel_emb_dim", "-ar_dim", type=int, default=32,
                        help="Relation embedding size for attention")
    parser.add_argument("--emb_dim", "-dim", type=int, default=32,
                        help="Entity embedding size")
    parser.add_argument("--num_gcn_layers", "-l", type=int, default=3,
                        help="Number of GCN layers")
    parser.add_argument("--num_bases", "-b", type=int, default=4,
                        help="Number of basis functions to use for GCN weights")
    parser.add_argument("--dropout", type=float, default=0,
                        help="Dropout rate in GNN layers")
    parser.add_argument("--edge_dropout", type=float, default=0.5,
                        help="Dropout rate in edges of the subgraphs")
    parser.add_argument('--gnn_agg_type', '-a', type=str, choices=['sum', 'mlp', 'gru'], default='sum',
                        help='what type of aggregation to do in gnn msg passing')
    parser.add_argument('--add_ht_emb', '-ht', type=bool, default=True,
                        help='whether to concatenate head/tail embedding with pooled graph representation')
    parser.add_argument('--has_attn', '-attn', type=bool, default=True,
                        help='whether to have attn in model or not')
    parser.add_argument('--use_numeric', action='store_true',
                        help='whether to use numeric features')
    parser.add_argument('--attribute_density', type=float, default=0.8,
                        help='attribute density')
    parser.add_argument('--seed', dest='seed', default=0,
                        type=int, help='Seed for randomization')
    parser.add_argument('--input_feature', type=str, choices=['numeric', 'ra', 'rra', 'normal', 'literalE'], default='rra')
    parser.add_argument("--dropout_num", type=float, default=0.1,
                        help="Dropout rate for num ")
    parser.add_argument('--loss_coef', type=float, default=1.0,
                        help='attribute density')
    parser.add_argument('--order_margin', type=float, default=0.0,
                        help='attribute density')
    parser.add_argument('--order_loss', type=str, choices=['not', 'l1', 'l2'], default='l2')
    #parser.add_argument('--order_embs', type=str, choices=['all'], default='all')
    parser.add_argument('--self_coef', type=float, default=1.0,
                        help='attribute density')
    parser.add_argument('--self_p', type=int, default=2,
                        help='attribute density')
    parser.add_argument('--self_margin', type=float, default=0.0,
                        help='attribute density')
    parser.add_argument('--use_self', type=str, choices=['not', 'self_1', 'self_2'], default='not')

    params = parser.parse_args()
    if params.input_feature in ['numeric', 'ra', 'rra', 'literalE'] and params.use_numeric == False:
        raise Exception('conflicts(numeric)')
    if params.input_feature in ['normal'] and params.use_numeric == True:
        raise Exception('conflicts(normal)')
    if params.use_self != 'not' and params.order_loss == 'not':
        raise Exception('conflicts(self and order)')

    initialize_experiment(params, __file__)

    deterministic = True
    torch.manual_seed(params.seed)
    torch.cuda.manual_seed(params.seed)
    torch.cuda.manual_seed_all(params.seed)
    np.random.seed(params.seed)
    if deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(params.seed)

    params.file_paths = {
        'train': os.path.join(params.main_dir, 'data/{}/{}.txt'.format(params.dataset, params.train_file)),
        'valid': os.path.join(params.main_dir, 'data/{}/{}.txt'.format(params.dataset, params.valid_file)),
        'train_neg': os.path.join(params.main_dir, 'data/{}/{}.txt'.format(params.dataset, params.train_neg_file)),
        'valid_neg': os.path.join(params.main_dir, 'data/{}/{}.txt'.format(params.dataset, params.valid_neg_file)),
        'literals': os.path.join(params.main_dir, 'data/{}/{}.txt'.format(params.dataset, params.literals_file))
    }

    if not params.disable_cuda and torch.cuda.is_available():
        params.device = torch.device('cuda:%d' % params.gpu)
    else:
        params.device = torch.device('cpu')

    params.collate_fn = collate_dgl
    params.move_batch_to_device = move_batch_to_device_dgl

    main(params)
