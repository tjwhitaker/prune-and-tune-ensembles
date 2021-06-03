import argparse
import torch
import models
import os
import sys
from data import get_loaders

parser = argparse.ArgumentParser(
    description='Prune and Tune Ensembles')

parser.add_argument('--dir', type=str, default='checkpoints', metavar='DIR',
                    help='training directory (default: checkpoints)')
parser.add_argument('--seed', type=int, default=1,
                    metavar='S', help='random seed (default: 1)')

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


# Tuning Args
parser.add_argument('--child_epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--tuning_schedule', type=str, default='onecycle', metavar='SCH',
                    help='Tuning schedule (default: onecycle)')


args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(args.dir, exist_ok=True)

(train_loader, test_loader), num_classes = get_loaders(
    dataset=args.dataset,
    data_path=args.data_path,
    train_batch_size=args.batch_size,
    test_batch_size=args.batch_size,
    num_workers=args.num_workers)

architecture = getattr(models, args.model)
model = architecture(num_classes=num_classes).to(device)


def learning_rate_schedule(base_lr, epoch, total_epochs):
    alpha = epoch / total_epochs
    if alpha <= 0.5:
        factor = 1.0
    elif alpha <= 0.9:
        factor = 1.0 - (alpha - 0.5) / 0.4 * 0.99
    else:
        factor = 0.01
    return factor * base_lr


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

###################
# Train the Parent
###################


criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(
    filter(lambda param: param.requires_grad, model.parameters()),
    lr=args.lr, momentum=args.momentum, weight_decay=args.wd, nesterov=True)

for epoch in range(args.epochs):
    train_loss = 0
    test_loss = 0
    lr = learning_rate_schedule(args.lr, epoch, args.epochs)
    adjust_learning_rate(optimizer, lr)

    # Training Loop
    model.train()
    num_train_batches = 0
    for _, (inputs, target) in enumerate(train_loader):
        inputs = inputs.to(device)
        target = target.to(device)

        prediction = model(inputs)

        loss = criterion(prediction, target)
        train_loss += loss.item()

        model.zero_grad()
        loss.backward()
        optimizer.step()
        num_train_batches += 1

    # Eval Loop
    model.eval()
    with torch.no_grad():
        num_correct = 0
        num_test_batches = 0
        for _, (inputs, target) in enumerate(test_loader):
            inputs = inputs.to(device)
            target = target.to(device)

            output = model(inputs)
            loss = criterion(output, target)
            test_loss += loss.item()

            _, predictions = torch.max(output, -1)

            num_correct += (predictions == target).sum().data.item()

            num_test_batches += 1

        accuracy = (num_correct / len(test_loader.dataset)) * 100

    if epoch % 1 == 0:
        print(f"Epoch: {epoch}")
        print(
            f"Train Loss: {train_loss / num_train_batches}, Test Loss: {test_loss / num_test_batches}, Accuracy:{accuracy}")

    if (epoch % args.save_freq == 0) or (epoch + 1 == args.epochs):
        torch.save(model.state_dict(), os.path.join(
            args.dir, f"{args.model}.pt"))

##############################
# Prune and Tune The Ensemble
##############################

# Only antirandom + one cycle set up for now

for i in range(3):
    architecture = getattr(models, args.model)
    model = architecture(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(
        os.path.join(args.dir, f"{args.model}.pt")))

    params = []
    for name, module in list(model.named_modules())[:-1]:
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            params.append((module, "weight"))

    for p in params:
        prune.random_unstructured(p[0], p[1], amount=0.5)

    # prune.global_unstructured(
    #     params, pruning_method=prune.RandomUnstructured, amount=0.2)

    tune(model, train_loader, test_loader, device)

    torch.save(model.state_dict(),
               f"checkpoints/wrn_pruned_{run}_50p_c100.pt")

    anti = WideResNetBase(28, 10, 0, 10).to(device)
    anti.load_state_dict(torch.load(f"checkpoints/wrn_100_3.pt"))

    init_prune_layers(anti)

    # Flip all bits in model weight_mask
    params = []
    for module in model.state_dict().keys():
        if 'weight_mask' in module:
            mask = (1 - model.state_dict()[module]).to(device)
            name = module.rsplit('.', 1)[0]
            params.append((name, mask))

    for p in params:
        prune.custom_from_mask(attrgetter(p[0])(anti), "weight", p[1])

    tune(anti, train_loader, test_loader, device)

    torch.save(anti.state_dict(),
               f"checkpoints/wrn_pruned_{run}_50p_c100_anti.pt")
