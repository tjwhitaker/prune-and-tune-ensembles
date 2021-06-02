import argparse
import torch
import models
import os
import sys
from data import get_loaders

parser = argparse.ArgumentParser(
    description='Prune and Tune Ensemble Training')
parser.add_argument('--dir', type=str, default='/tmp/curve/', metavar='DIR',
                    help='training directory (default: /tmp/curve/)')

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

parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train (default: 200)')
parser.add_argument('--save_freq', type=int, default=50, metavar='N',
                    help='save frequency (default: 50)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='initial learning rate (default: 0.1)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--wd', type=float, default=5e-4, metavar='WD',
                    help='weight decay (default: 5e-4)')

parser.add_argument('--seed', type=int, default=1,
                    metavar='S', help='random seed (default: 1)')

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(args.dir, exist_ok=True)

(train_loader, test_loader), num_classes = get_loaders(
    dataset=args.dataset,
    batch_size=args.batch_size,
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
            args.dir, f"{args.model}-{str(epoch)}.pt"))
