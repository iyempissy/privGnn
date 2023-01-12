# Releasing Graph Neural Networks with Differential Privacy Guarantees
<!--### by Iyiola E. Olatunji, Thorben Funke, Megha Khosla-->
This repository contains additional details and reproducibility of the stated results. 

## Motivation: 
Real-world graphs, such often are associated with sensitive information about individuals and their activities. Hence, they cannot always be made public. Moreover, graph neural networks (GNNs) models trained on such sensitive data can leak significant amount of information e.g via membership inference attacks.

## Goal of the Paper: 
Release a GNN model that is trained on sensitive data yet robust to attacks. Importantly, it should have differential privacy (DP) guarantees.

## Proposed Method
<p align=center><img style="vertical-align:middle" src="https://github.com/iyempissy/privGnn/blob/main/images/PrivGNN.png" /></p>


# Results
## Privacy-accuracy Tradeoff
<p align=center><img style="vertical-align:middle" src="https://github.com/iyempissy/privGnn/blob/main/images/accuracy_vs_noise.png" /></p>

## Privacy Budget vs Noise
<p align=center><img style="vertical-align:middle" src="https://github.com/iyempissy/privGnn/blob/main/images/privacybudget_vs_noise.png" /></p>

## Membership Inference Attacks
## Performance on Attack 1
<p align=center><img style="vertical-align:middle" src="https://github.com/iyempissy/privGnn/blob/main/images/miattack_1.png" /> </p>

## Performance on Attack 2
<p align=center><img style="vertical-align:middle" src="https://github.com/iyempissy/privGnn/blob/main/images/miattack_2.png" /> </p>


## Reproducing Results
PrivGNN can be executed via 
```
python3 knn_graph.py
```

### Changing Parameters & Setting
Please adjust the `all_config.py` file to change parameters, such as the datasets, K, lambda, attack, baselines, or gamma.
