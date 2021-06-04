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

Train the parent network, create the ensemble and tune the ensemble members.

```train
python train.py --checkpoint_dir=<DIR> \
                --dataset=<DATASET> \
                --data_path=<DATA_PATH> \
                --batch_size=<BATCH_SIZE> \
                --num_workers=<num_workers> \
                --model=<MODEL> \
                --parent_epochs=<PARENT_EPOCHS> \
                --save_freq=<SAVE_FREQ> \
                --optimizer=<OPTIMIZER> \
                --lr=<LR> \
                --momentum=<M> \
                --wd=<WD> \
                --num_children=<NUM> \
                --pruning_method=<METHOD> \
                --pruning_structure=<STRUCTURE> \
                --sparsity=<SPARSITY> \
                --child_epochs=<CHILD_EPOCHS> \
                --child_optimizer=<OPTIMIZER> \
                --child_lr=<LR> \
                --child_momentum=<M> \
                --child_wd=<WD> \
                --tuning_schedule=<SCHED> \
                --verbose=<V> \
                --seed=<SEED> \
```

Parameters:

- `checkpoint_dir` (default: checkpoints/)
- `dataset` [cifar10/cifar100] (default: cifar10)
- `data_path` (default: data/)
- `batch_size` (default: 128)
- `num_workers` (default: 4)
- `model` (default: None)
  - resnet18
  - densenet121
  - wideresnet28x10
- `parent_epochs` (default: 140)
- `save_freq` (default: 10)
- `optimizer` [sgd/adam] (default: sgd)
- `lr` (default: 0.1)
- `momentum` (default: 0.9)
- `wd` (default: 5e-4)
- `num_children` (default: 6)
- `pruning_method` [antirandom/random] (default: antirandom)
- `pruning_structure` [connections/neurons] (default: connections)
- `sparsity` (default: 0.5)
- `child_epochs` (default: 10)
- `child_optimizer` [sgd/adam] (default: sgd)
- `child_lr` (default: 0.1)
- `child_momentum` (default: 0.9)
- `child_wd` (default: 5e-4)
- `tuning_schedule` [onecycle/fixed] (default: onecycle)
- `verbose` [0/1] (default: 1)
- `seed` (default: 1)

## Evaluation

Uses the checkpoints saved from the training process. Each model is evaluated and logs their predictions to a file in predictions_dir. The ensemble is then evaluated on the test set.

```evaluate
python evaluate.py --predictions_dir=<PRED_DIR> \
                   --checkpoint_dir=<DIR> \
                   --dataset=<DATASET> \
                   --data_path=<DATA_PATH> \
                   --batch_size=<BATCH_SIZE> \
                   --num_workers=<num_workers> \
                   --model=<MODEL> \
                   --verbose=<V> \
                   --seed=<SEED> \
```

- `predictions_dir` (default: predictions/)
- `checkpoint_dir` (default: checkpoints/)
- `dataset` [cifar10/cifar100] (default: cifar10)
- `data_path` (default: data/)
- `batch_size` (default: 100)
- `num_workers` (default: 4)
- `model` (default: None)
  - resnet18
  - densenet121
  - wideresnet28x10
- `verbose` [0/1] (default: 1)
- `seed` (default: 1)

## Results

```reproduce
python train.py --
```

Our model achieves the following performance on :

### [Image Classification on ImageNet](https://paperswithcode.com/sota/image-classification-on-imagenet)

| Model name       | Top 1 Accuracy | Top 5 Accuracy |
| ---------------- | -------------- | -------------- |
| My awesome model | 85%            | 95%            |

> 📋 Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it.

## Contributing

> 📋 Pick a licence and describe how to contribute to your code repository.
