import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import os
import torch.nn.utils.prune as prune
import models
from operator import attrgetter


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


def init_prune_layers(model):
    params = []
    for name, module in list(model.named_modules())[:-1]:
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            params.append((module, "weight"))

    prune.global_unstructured(
        params, pruning_method=prune.RandomUnstructured, amount=0.0)


def train_parent(model, optimizer, criterion, epochs, init_lr, train_loader, test_loader, verbose, device):
    for i in range(epochs):
        train_loss = 0
        test_loss = 0
        lr = learning_rate_schedule(init_lr, i, epochs)
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

        if verbose == 1:
            print(f"Epoch: {i}")
            print(
                f"Train Loss: {train_loss / num_train_batches}, Test Loss: {test_loss / num_test_batches}, Accuracy:{accuracy}")


def create_ensemble(model_name, num_classes, num_children, parent_path, method, structure, sparsity, checkpoint_dir, device):
    member_paths = []

    if method == "antirandom":
        for i in range(int(num_children / 2)):
            architecture = getattr(models, model_name)
            model = architecture(num_classes=num_classes).to(device)
            model.load_state_dict(torch.load(parent_path))

            params = []
            for name, module in list(model.named_modules())[:-1]:
                if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
                    params.append((module, "weight"))

            if structure == "connections":
                for p in params:
                    prune.random_unstructured(p[0], p[1], amount=0.5)
            elif structure == "neurons":
                for p in params:
                    prune.random_structured(p[0], p[1], amount=0.5, dim=0)

            path = os.path.join(checkpoint_dir, f"{model_name}-c{i*2}.pt")
            member_paths.append(path)
            torch.save(model.state_dict(), path)

            antiarchitecture = getattr(models, model_name)
            antimodel = antiarchitecture(num_classes=num_classes).to(device)
            antimodel.load_state_dict(torch.load(parent_path))

            init_prune_layers(antimodel)

            # Flip all bits in model weight_mask
            params = []
            for module in model.state_dict().keys():
                if 'weight_mask' in module:
                    mask = (1 - model.state_dict()[module]).to(device)
                    name = module.rsplit('.', 1)[0]
                    params.append((name, mask))

            for p in params:
                prune.custom_from_mask(attrgetter(
                    p[0])(antimodel), "weight", p[1])

            path = os.path.join(checkpoint_dir, f"{model_name}-c{(i*2)+1}.pt")
            member_paths.append(path)
            torch.save(antimodel.state_dict(), path)
    else:
        for i in range(num_children):
            architecture = getattr(models, model_name)
            model = architecture(num_classes=num_classes).to(device)
            model.load_state_dict(torch.load(parent_path))

            params = []
            for name, module in list(model.named_modules())[:-1]:
                if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
                    params.append((module, "weight"))

            if structure == "connections":
                for p in params:
                    prune.random_unstructured(p[0], p[1], amount=sparsity)
            elif structure == "neurons":
                for p in params:
                    prune.random_structured(p[0], p[1], amount=sparsity, dim=0)

            path = os.path.join(checkpoint_dir, f"{model_name}-c{i}.pt")
            member_paths.append(path)
            torch.save(model.state_dict(), path)

    return member_paths


def tune_child(model, member_path, criterion, optimizer, child_epochs, schedule, lr, train_loader, test_loader, verbose, device):

    if schedule == 'onecycle':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=lr,
            steps_per_epoch=len(train_loader),
            div_factor=100,
            epochs=child_epochs,
            pct_start=0.1)

    for i in range(child_epochs):
        train_loss = 0
        test_loss = 0

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
            if schedule == 'onecycle':
                scheduler.step()

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

            accuracy = (num_correct / 10000) * 100

        if verbose == 1:
            print(f"Epoch: {i}")
            print(
                f"Train Loss: {train_loss / num_train_batches}, Test Loss: {test_loss / num_test_batches}, Accuracy:{accuracy}")

    torch.save(model.state_dict(), member_path)
