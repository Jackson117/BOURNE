# BOURNE
Source Code of ICDE'24 submitted paper "BOURNE: Bootstrapped Self-supervised Learning Framework for Unified Graph Anomaly Detection"

## Dependencies
+ python==3.8.13
+ torch_geometric==2.3.0
+ matplotlib==3.5.1
+ networkx==3.1
+ numpy==1.21.6
+ scipy==1.8.0
+ sklearn==0.24.1
+ torch==2.0.0
+ tqdm==4.65.0

## Usage
### Node Anomaly Detection
To train and evaluate on Cora:
```
python train_node.py --dataset cora --layer_sizes 256 --epochs 500 --batch_size 2000 --lr 0.001 --alpha 1.0 --beta 0.4 --eval_rounds 200 --cudaID 0
```

### Edge Anomaly Detection
To train and evaluate on Cora:
```
python train_edge.py --dataset cora --layer_sizes 256 --epochs 500 --batch_size 2000 --lr 0.001 --alpha 1.0 --beta 0.4 --eval_rounds 200 --cudaID 0
```
