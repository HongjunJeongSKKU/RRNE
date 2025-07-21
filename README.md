# RRNE - Relation-aware Relative Numeric Encoding
This repository contains official implementation of paper "Enhancing Inductive Numerical Reasoning in Knowledge Graphs with Relation-Aware Relative Numeric Encoding" ([paper](https://link.springer.com/chapter/10.1007/978-981-96-8173-0_14)).

This repository is based on [GraIL](https://github.com/kkteru/grail).

## Requiremetns

All the required packages can be installed by running `pip install -r requirements.txt`.

## Data statistics

Our data statistics are as follows:
![statistics](data_statistics.png)
# Train
For traning of RRNE on Credit or Spotify, please run below code

	python train.py -d $dataset -e $exp_name --input_feature rra --order_loss l2 --use_self self_1 --use_numric --self_margin 1.0 --hop $hop
 
 For traning of RRNE on US-cities please run below code

 	python train.py -d USA_sparse_f -e $exp_name --input_feature rra --order_loss l2 --use_self self_1 --use_numric --self_margin 1.0 --hop 3 --self_coef 0.25

  	python train.py -d USA_sparse_f -e $exp_name --input_feature rra --order_loss l2 --use_self self_1 --use_numric --self_margin 1.0 --hop 4 --batch_size 4 --lr 2e-3 --self_coef 0.25


# Test
For test of RRNE, please run below code

	python train.pyd -d $dataset_ind -e $exp_name --use_numeric --hop $hop

# Citation
If this repository is helpful for you, please cite this paper.

    @InProceedings{10.1007/978-981-96-8173-0_14,
    author="Jeong, Hongjun and Jung, Heesoo and Kim, Gayeong and Kim, Juann and Kim, Ko Keun and Park, Hogun",
    title="Enhancing Inductive Numerical Reasoning in Knowledge Graphs with Relation-Aware Relative Numeric Encoding",
    booktitle="Advances in Knowledge Discovery and Data Mining",
    year="2025",
    pages="173--186",
    }

