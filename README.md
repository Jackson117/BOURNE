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
cd NAD
```
```
python train_node.py --dataset cora --layer_sizes 256 --epochs 500 --batch_size 2000 --lr 0.001 --alpha 1.0 --beta 0.4 --eval_rounds 200 --cudaID 0
```

### Edge Anomaly Detection
To train and evaluate on Cora:
```
cd EAD
```
```
python train_edge.py --dataset cora --layer_sizes 256 --epochs 500 --batch_size 2000 --lr 0.001 --alpha 1.0 --beta 0.4 --eval_rounds 200 --cudaID 0
```

### Reference
```bibtex
@inproceedings{liu2024bourne,
  title={Bourne: Bootstrapped self-supervised learning framework for unified graph anomaly detection},
  author={Liu, Jie and He, Mengting and Shang, Xuequn and Shi, Jieming and Cui, Bin and Yin, Hongzhi},
  booktitle={2024 IEEE 40th International Conference on Data Engineering (ICDE)},
  pages={2820--2833},
  year={2024},
  organization={IEEE}
}
```
