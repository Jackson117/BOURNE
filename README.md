# BOURNE
Source Code of ICDE'24 submitted paper "BOURNE: Bootstrapped Self-supervised Learning Framework for Unified Graph Anomaly Detection"

## Dependencies
+ python==3.6.1
+ dgl==0.4.1
+ matplotlib==3.3.4
+ networkx==2.5
+ numpy==1.19.2
+ pyparsing==2.4.7
+ scikit-learn==0.24.1
+ scipy==1.5.2
+ sklearn==0.24.1
+ torch==1.8.1
+ tqdm==4.59.0

## Usage
# Node Anomaly Detection
To train and evaluate on Cora:
```
python train_node.py --dataset cora --layer_sizes 256 --epochs 500 --batch_size 2000 --lr 0.001 --alpha 1.0 --beta 0.4 --eval_rounds 200 --cudaID 0
```
To train and evaluate on Pubmed:
```
python run.py --device cuda:0 --expid 2 --dataset Flickr --runs 5 --auc_test_rounds 256 --alpha 1.0 --beta 0.6
```
To train and evaluate on BlogCatalog:
```
python run.py --device cuda:0 --expid 3 --dataset cora --runs 5 --auc_test_rounds 256 --alpha 1.0 --beta 0.6
```

# Edge Anomaly Detection
To train and evaluate on Cora:
```
python train_edge.py --dataset cora --layer_sizes 256 --epochs 500 --batch_size 2000 --lr 0.001 --alpha 1.0 --beta 0.4 --eval_rounds 200 --cudaID 0
```
To train and evaluate on Pubmed:
```
python run.py --device cuda:0 --expid 2 --dataset Flickr --runs 5 --auc_test_rounds 256 --alpha 1.0 --beta 0.6
```
To train and evaluate on BlogCatalog:
```
python run.py --device cuda:0 --expid 3 --dataset cora --runs 5 --auc_test_rounds 256 --alpha 1.0 --beta 0.6
```
