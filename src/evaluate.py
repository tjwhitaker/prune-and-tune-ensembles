import argparse
import torch
import models
import pickle
import os
import sys
import numpy as np
from data import get_loaders
from utils import init_prune_layers
from utils import train_parent, create_ensemble, tune_child

parser = argparse.ArgumentParser(
    description='Prune and Tune Ensembles')


parser.add_argument('--predictions_dir', type=str, default='predictions', metavar='DIR',
                    help='predictions directory (default: predictions)')
parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', metavar='DIR',
                    help='training directory (default: checkpoints)')
parser.add_argument('--seed', type=int, default=1,
                    metavar='S', help='random seed (default: 1)')
parser.add_argument('--verbose', type=int, default=1,
                    metavar='S', help='logging verbosity (default: 1)')
# Data Args
parser.add_argument('--dataset', type=str, default='CIFAR10', metavar='DATASET',
                    help='dataset name (default: CIFAR10)')
parser.add_argument('--data_path', type=str, default=None, metavar='PATH',
                    help='path to datasets location (default: None)')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='input batch size (default: 128)')
parser.add_argument('--num-workers', type=int, default=4, metavar='N',
                    help='number of workers (default: 4)')

parser.add_argument('--model', type=str, default=None, metavar='MODEL', required=True,
                    help='model name (default: None)')

# Parent Training Args
parser.add_argument('--parent_epochs', type=int, default=140, metavar='N',
                    help='number of epochs to train (default: 140)')
parser.add_argument('--save_freq', type=int, default=50, metavar='N',
                    help='save frequency (default: 50)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='initial learning rate (default: 0.1)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--wd', type=float, default=5e-4, metavar='WD',
                    help='weight decay (default: 5e-4)')

# Ensemble Args
parser.add_argument('--num_children', type=int, default=6, metavar='N',
                    help='number of ensemble children (default: 6)')
parser.add_argument('--pruning_method', type=str, default='antirandom', metavar='M',
                    help='pruning method (default: antirandom)')
parser.add_argument('--pruning_structure', type=str, default='connections', metavar='S',
                    help='pruning structure (default: connections)')
parser.add_argument('--sparsity', type=float, default=0.5, metavar='S',
                    help='pruning sparsity (default: 0.5)')


# Tuning Args
parser.add_argument('--child_epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--tuning_schedule', type=str, default='onecycle', metavar='SCH',
                    help='Tuning schedule (default: onecycle)')
parser.add_argument('--tuning_lr', type=float, default=0.1, metavar='LR',
                    help='initial learning rate (default: 0.1)')


args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(args.predictions_dir, exist_ok=True)

(train_loader, test_loader), num_classes = get_loaders(
    dataset=args.dataset,
    data_path=args.data_path,
    train_batch_size=args.batch_size,
    test_batch_size=args.batch_size,
    num_workers=args.num_workers)

# Make predictions for each model in the ensemble
for filename in os.listdir(args.checkpoint_dir):
    if filename.startswith(args.model):
        architecture = getattr(models, args.model)
        model = architecture(num_classes=num_classes).to(device)

        # For parent predictions
        if not filename.endswith('-p.pt'):
            init_prune_layers(model)

        model.load_state_dict(torch.load(
            os.path.join(args.checkpoint_dir, filename)))

        with torch.no_grad():
            predictions = []

            model.eval()

            num_correct = 0
            targets = []

            for _, (inputs, target) in enumerate(test_loader):
                inputs = inputs.to(device)
                target = target.to(device)

                targets.append(target)

                logits = []

                output = torch.softmax(model(inputs), 1)
                logits.append(output.cpu().numpy())

                predictions.append(logits)

            # Pickle and save
            pickle.dump(predictions, open(os.path.join(
                args.predictions_dir, f"{filename}.p"), "wb"))

# Combine predictions
predictions = []

for filename in os.listdir(args.predictions_dir):
    if filename.startswith(args.model):
        predictions.append(pickle.load(
            open(os.path.join(args.predictions_dir, filename), "rb")))

predictions = np.array(predictions).squeeze()
logits_sum = np.sum(predictions, axis=0)
num_correct = 0

for i, (_, target) in enumerate(test_loader):
    target = target.numpy()
    prediction = np.argmax(logits_sum[i], axis=1)
    num_correct += np.sum(prediction == target)


accuracy = (num_correct / len(test_loader.dataset)) * 100

print("Accuracy: ", accuracy)
