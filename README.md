> 📋 A template README.md for code accompanying a Machine Learning paper

# Prune and Tune Ensembles

This repository is the official implementation of [Prune and Tune Ensembles](#).

![Landscape Visual](./figures/landscape-visual.png)
![Anti-Random Networks](./figures/antirandom-networks.png)

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

> 📋 Describe how to set up the environment, e.g. pip/conda/docker commands, download datasets, etc...

## Training

To train the model(s) in the paper, run this command:

```train
python train.py --input-data <path_to_data> --alpha 10 --beta 20
```

```train
python train.py --dir=<DIR> \
                --dataset=<DATASET> \
                --data_path=<DATA_PATH> \
                --model=<MODEL> \
                --epochs=<EPOCHS> \
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
- `EPOCHS` &mdash; number of training epochs (default: 200)
- `OPTIMIZER` &mdash; optimizer name [ADAM/SGD] (default: ADAM)
- `LR` &mdash; initial learning rate (default: 0.001)

> 📋 Describe how to train the models, with example commands on how to train the models in your paper, including the full training procedure and appropriate hyperparameters.

## Evaluation

To evaluate my model on ImageNet, run:

```eval
python eval.py --model-file mymodel.pth --benchmark imagenet
```

> 📋 Describe how to evaluate the trained models on benchmarks reported in the paper, give commands that produce the results (section below).

## Pre-trained Models

You can download pretrained models here:

- [My awesome model](https://drive.google.com/mymodel.pth) trained on ImageNet using parameters x,y,z.

> 📋 Give a link to where/how the pretrained models can be downloaded and how they were trained (if applicable). Alternatively you can have an additional column in your results table with a link to the models.

## Results

Our model achieves the following performance on :

### [Image Classification on ImageNet](https://paperswithcode.com/sota/image-classification-on-imagenet)

| Model name       | Top 1 Accuracy | Top 5 Accuracy |
| ---------------- | -------------- | -------------- |
| My awesome model | 85%            | 95%            |

> 📋 Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it.

## Contributing

> 📋 Pick a licence and describe how to contribute to your code repository.
