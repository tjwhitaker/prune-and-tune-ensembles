import argparse
import torch
import models
import os
import sys
from data import get_loaders
from utils import init_prune_layers
from utils import train_parent, create_ensemble, tune_child

parser = argparse.ArgumentParser(
    description='Prune and Tune Ensembles')

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
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='input batch size (default: 128)')
parser.add_argument('--num-workers', type=int, default=4, metavar='N',
                    help='number of workers (default: 4)')

parser.add_argument('--model', type=str, default=None, metavar='MODEL', required=True,
                    help='model name (default: None)')

# Parent Training Args
parser.add_argument('--parent_epochs', type=int, default=140, metavar='N',
                    help='number of epochs to train (default: 140)')
parser.add_argument('--save_freq', type=int, default=10, metavar='N',
                    help='save frequency (default: 10)')
parser.add_argument('--optimizer', type=str, default='sgd', metavar='OPT',
                    help='optimizer (default: sgd)')
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
parser.add_argument('--child_optimizer', type=str, default='sgd', metavar='OPT',
                    help='optimizer (default: sgd)')
parser.add_argument('--child_lr', type=float, default=0.1, metavar='LR',
                    help='initial learning rate (default: 0.1)')
parser.add_argument('--child_momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--child_wd', type=float, default=5e-4, metavar='WD',
                    help='weight decay (default: 5e-4)')
parser.add_argument('--tuning_schedule', type=str, default='onecycle', metavar='SCH',
                    help='Tuning schedule (default: onecycle)')


args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(args.checkpoint_dir, exist_ok=True)

(train_loader, test_loader), num_classes = get_loaders(
    dataset=args.dataset,
    data_path=args.data_path,
    train_batch_size=args.batch_size,
    test_batch_size=args.batch_size,
    num_workers=args.num_workers)

architecture = getattr(models, args.model)
model = architecture(num_classes=num_classes).to(device)

###################
# Train the Parent
###################

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(
    filter(lambda param: param.requires_grad, model.parameters()),
    lr=args.lr, momentum=args.momentum, weight_decay=args.wd, nesterov=True)

train_parent(model, optimizer, criterion, args.parent_epochs, args.lr,
             train_loader, test_loader, args.verbose, device)

torch.save(model.state_dict(), os.path.join(
    args.checkpoint_dir, f"{args.model}-p.pt"))


######################
# Create The Ensemble
######################

# Ensemble members contains a list of paths to the ensemble member state_dict files
ensemble_members = create_ensemble(
    args.model,
    num_classes,
    args.num_children,
    os.path.join(args.checkpoint_dir, f"{args.model}-p.pt"),
    args.pruning_method,
    args.pruning_structure,
    args.sparsity,
    args.checkpoint_dir,
    device)

####################
# Tune The Ensemble
####################

for member_path in ensemble_members:
    architecture = getattr(models, args.model)
    model = architecture(num_classes=num_classes).to(device)
    init_prune_layers(model)
    model.load_state_dict(torch.load(member_path))

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        filter(lambda param: param.requires_grad, model.parameters()),
        lr=args.child_lr, momentum=args.child_momentum, weight_decay=args.child_wd, nesterov=True)

    tune_child(
        model,
        member_path,
        criterion,
        optimizer,
        args.child_epochs,
        args.tuning_schedule,
        args.child_lr,
        train_loader,
        test_loader,
        args.verbose,
        device
    )
