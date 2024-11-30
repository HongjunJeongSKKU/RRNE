# RARNE - Relation-Aware Relative Numeric Encoding

This code is based on [GraIL](https://github.com/kkteru/grail).

## Requiremetns

All the required packages can be installed by running `pip install -r requirements.txt`.

# Train
For traning of RRANE, please run below code

	python train.py -d $dataset -e $exp_name --input_feature rra --order_loss l2 --use_self self_1 --use_numric --self_margin 1.0

# Test
For test of RRANE, please run below code

	python train.pyd -d $dataset_ind -e $exp_name --use_numeric
