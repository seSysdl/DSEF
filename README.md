

# DSEM

This repository is for "Dynamic Siamese Expansion Framework for Improving Robustness in Online Continual Learning" in NeurIPS 2025.

## Abstract

Continual learning requires the model to continually capture novel information without forgetting prior knowledge. Nonetheless, existing studies predominantly address the catastrophic forgetting, often neglecting enhancements in model robustness. Consequently, these methodologies fall short in real-time applications, such as autonomous driving, where data samples frequently exhibit noise due to environmental and lighting variations, thereby impairing model efficacy and causing safety issues. In this paper, we address robustness in continual learning systems by introducing an innovative approach, the Dynamic Siamese Expansion Framework (DSEF) that employs a Siamese backbone architecture, comprising static and dynamic components, to facilitate the learning of both global and local representations over time. Specifically, the proposed framework dynamically generates a lightweight expert for each novel task, leveraging the Siamese backbone to enable rapid adaptation. A novel Robust Dynamic Representation Optimization (RDRO) approach is proposed to incrementally update the dynamic backbone by maintaining all previously acquired representations and prediction patterns of historical experts, thereby fostering new task learning without inducing detrimental knowledge transfer. Additionally, we propose a novel Robust Feature Fusion (RFF) approach to incrementally amalgamate robust representations from all historical experts into the expert construction process. A novel mutual information-based technique is employed to derive adaptive weights for feature fusion by assessing the knowledge relevance between historical experts and the new task, thus maximizing positive knowledge transfer effects. A comprehensive experimental evaluation, benchmarking our approach against established baselines, demonstrates that our method achieves state-of-the-art performance even under adversarial attacks.

### Content
```
DSEM 
├── backbone
│  ├── ViT.py
├── data
│  ├── README.md
├── datasets
│  ├── transforms
│  ├── utils
│  ├── seq_cifar10.py
│  ├── seq_cifar100.py
│  ├── seq_cub200.py
│  ├── seq_tinyimagenet.py
├── models
│  ├── utils
│  ├── siamesevit.py
├── utils
│  ├── main.py
│  ├── train.py
│  ├── ...
├── weights
│  ├── README.md
└── README.md
```

### Command

```
 python utils/main.py  \

	--dataset seq-cifar10  \

	--model robustdualvit  \

	--buffer_size 500  \

	--lr 0.03  \	

	--minibatch_size 32  \

	--alpha 0.3  \

	--r_alpha 0.3  \

	--batch_size 32  \

	--n_epochs 1  \
```

### License

This project is licensed under the MIT License.

### Citation

```
@inproceedings{
ye2025dynamic,
title={Dynamic Siamese Expansion Framework for Improving Robustness in Online Continual Learning},
author={Fei Ye and Yulong Zhao and Qihe Liu and Junlin Chen and Adrian G. Bors and Jingling sun and Rongyao Hu and shijie zhou},
booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
year={2025},
url={https://openreview.net/forum?id=M1OqlaNrw7}
}
```



