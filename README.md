> 📋 A template README.md for code accompanying a Machine Learning paper

# Prune and Tune Ensembles

This repository is the official implementation of [Prune and Tune Ensembles](#).

![Landscape Visual](./figures/landscape-visual.png)

<!-- ![Anti-Random Networks](./figures/antirandom-networks.png) -->

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

> 📋 Describe how to set up the environment, e.g. pip/conda/docker commands, download datasets, etc...

## Training

Our code is split up into three commands.

Train the parent network.

```train
python train.py --dir=<DIR> \
                --dataset=<DATASET> \
                --data_path=<DATA_PATH> \
                --model=<MODEL> \
                --epochs=<PARENT_EPOCHS> \
                --optimizer=<OPTIMIZER> \
                --lr=<LR>
```

Parameters:

- `DIR` &mdash; path to training directory where checkpoints will be stored (default: checkpoints/)
- `DATASET` &mdash; dataset name [CIFAR10/CIFAR100] (default: CIFAR10)
- `PATH` &mdash; path to the data directory (default: data/)
- `MODEL` &mdash; Model name:
  - LeNet
  - ResNet
  - DenseNet
  - WideResNet

```reproduce
python train.py --checkpoint_dir
```

## Evaluation

To evaluate my model on ImageNet, run:

```evaluate
python evaluate.py --dir=<DIR> \
                   --dataset=<DATASET> \
```

> 📋 Describe how to evaluate the trained models on benchmarks reported in the paper, give commands that produce the results (section below).

## Results

Our model achieves the following performance on :

### [Image Classification on ImageNet](https://paperswithcode.com/sota/image-classification-on-imagenet)

| Model name       | Top 1 Accuracy | Top 5 Accuracy |
| ---------------- | -------------- | -------------- |
| My awesome model | 85%            | 95%            |

> 📋 Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it.

## Contributing

> 📋 Pick a licence and describe how to contribute to your code repository.
