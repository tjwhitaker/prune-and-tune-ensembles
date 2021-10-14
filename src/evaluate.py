import argparse
import torch
import models
import pickle
import os
import sys
import numpy as np
from data import get_loaders, get_corrupted_loaders
from utils import init_prune_layers

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
parser.add_argument('--dataset', type=str, default='cifar10', metavar='DATASET',
                    help='dataset name (default: cifar10)')
parser.add_argument('--data_path', type=str, default='data', metavar='PATH',
                    help='path to datasets location (default: data)')
parser.add_argument('--batch_size', type=int, default=100, metavar='N',
                    help='input batch size (default: 100)')
parser.add_argument('--num-workers', type=int, default=4, metavar='N',
                    help='number of workers (default: 4)')
parser.add_argument('--corrupted', type=bool, default=False, metavar='C', 
                    help='test on corrupted dataset (default: False)')
parser.add_argument('--model', type=str, default=None, metavar='MODEL', required=True,
                    help='model name (default: None)')

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(args.predictions_dir, exist_ok=True)

if args.corrupted:
    test_loader, targets, num_classes = get_corrupted_loaders(
        dataset=args.dataset,
        data_path=args.data_path,
        train_batch_size=args.batch_size,
        test_batch_size=args.batch_size,
        num_workers=args.num_workers)
else:
    _, test_loader, num_classes = get_loaders(
        dataset=args.dataset,
        data_path=args.data_path,
        train_batch_size=args.batch_size,
        test_batch_size=args.batch_size,
        num_workers=args.num_workers)

###################
# Make Predictions
###################

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

            for _, (inputs, _) in enumerate(test_loader):
                inputs = inputs.to(device)

                logits = []

                output = model(inputs)
                logits.append(output.cpu().numpy())

                predictions.append(logits)

            # Pickle and save
            pickle.dump(predictions, open(os.path.join(
                args.predictions_dir, f"{filename}.p"), "wb"))

######################
# Combine predictions
######################

predictions = []

for filename in os.listdir(args.predictions_dir):
    if filename.startswith(args.model):
        predictions.append(pickle.load(
            open(os.path.join(args.predictions_dir, filename), "rb")))

predictions = np.array(predictions, dtype=np.float).squeeze()
predictions = torch.from_numpy(predictions)

softmaxes = torch.softmax(predictions, dim=-1)

mean_softmaxes = torch.mean(softmaxes, dim=0)
mean_logits = torch.mean(predictions, dim=0)

mean_softmaxes = torch.reshape(
    mean_softmaxes, (len(test_loader.dataset), num_classes))
mean_logits = torch.reshape(
    mean_logits, (len(test_loader.dataset), num_classes))

if args.corrupted:
    labels = torch.tensor(targets).long()
else:
    labels = torch.tensor(test_loader.dataset.targets)

#################
# Accuracy + NLL
#################

_, predictions = torch.max(mean_logits, -1)
num_correct = (predictions == labels).sum().data.item()

print(f"ACC: {100 * num_correct / len(test_loader.dataset)}")

criterion = torch.nn.CrossEntropyLoss()
loss = criterion(mean_logits, labels)

print(f"NLL: {loss}")

#############################
# Expected Calibration Error
#############################

n_bins = 15
boundaries = torch.linspace(0, 1, n_bins + 1)
bin_lowers = boundaries[:-1]
bin_uppers = boundaries[1:]

confidences, predictions = torch.max(mean_softmaxes, -1)
accuracies = predictions.eq(labels)

ece = torch.zeros(1, device=device)
for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
    in_bin = confidences.gt(bin_lower.item()) * \
        confidences.le(bin_upper.item())
    prop_in_bin = in_bin.float().mean()

    if prop_in_bin.item() > 0:
        accuracy_in_bin = accuracies[in_bin].float().mean()
        avg_confidence_in_bin = confidences[in_bin].mean()
        ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

print(f"ECE: {ece.item()}")
