# Prune and Tune Ensembles

This repository is the official implementation of [Prune and Tune Ensembles](#).

![Landscape Visual](./figures/landscape-visual.png)

<!-- ![Anti-Random Networks](./figures/antirandom-networks.png) -->

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

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

- `checkpoint_dir` &mdash; (default: checkpoints/)
- `dataset` &mdash; [cifar10/cifar100] (default: cifar10)
- `data_path` &mdash; (default: data/)
- `batch_size` &mdash; (default: 128)
- `num_workers` &mdash; (default: 4)
- `model` &mdash; (default: None)
  - resnet18
  - densenet121
  - wideresnet28x10
- `parent_epochs` &mdash; (default: 140)
- `save_freq` &mdash; (default: 10)
- `optimizer` &mdash; [sgd/adam] (default: sgd)
- `lr` &mdash; (default: 0.1)
- `momentum` &mdash; (default: 0.9)
- `wd` &mdash; (default: 5e-4)
- `num_children` &mdash; (default: 6)
- `pruning_method` &mdash; [antirandom/random] (default: antirandom)
- `pruning_structure` &mdash; [connections/neurons] (default: connections)
- `sparsity` &mdash; (default: 0.5)
- `child_epochs` &mdash; (default: 10)
- `child_optimizer` &mdash; [sgd/adam] (default: sgd)
- `child_lr` &mdash; (default: 0.1)
- `child_momentum` &mdash; (default: 0.9)
- `child_wd` &mdash; (default: 5e-4)
- `tuning_schedule` &mdash; [onecycle/fixed] (default: onecycle)
- `verbose` &mdash; [0/1] (default: 1)
- `seed` &mdash; (default: 1)

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

- `predictions_dir` &mdash; (default: predictions/)
- `checkpoint_dir` &mdash; (default: checkpoints/)
- `dataset` &mdash; [cifar10/cifar100] (default: cifar10)
- `data_path` &mdash; (default: data/)
- `batch_size` &mdash; (default: 100)
- `num_workers` &mdash; (default: 4)
- `model` &mdash; (default: None)
  - resnet18
  - densenet121
  - wideresnet28x10
- `verbose` &mdash; [0/1] (default: 1)
- `seed` &mdash; (default: 1)

## Results

The defaults are set up to reproduce the "large training budget experiment" from the paper.

That is, parent is trained for 140 epochs. 6 children are created using antirandom pruning that are each tuned for 10 epochs using the one-cycle schedule.

```reproduce
python train.py --model wideresnet28x10 --dataset cifar10
```

```reproduce
python train.py --model wideresnet28x10 --dataset cifar100
```

| Model name             | CIFAR-10 Accuracy | CIFAR-100 Accuracy |
| ---------------------- | ----------------- | ------------------ |
| WRN28x10 PAT (AR + 1C) | 96.45%            | 82.44%             |
