import torch
import torchvision


def get_loaders(dataset='CIFAR10', data_path='data', train_batch_size=128, test_batch_size=1, num_workers=4):
    if dataset == 'CIFAR10':
        num_classes = 10

        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2471, 0.2435, 0.2616)

        train_transforms = torchvision.transforms.Compose(
            [torchvision.transforms.RandomCrop(32, padding=4),
             torchvision.transforms.RandomHorizontalFlip(),
             torchvision.transforms.ToTensor(),
             torchvision.transforms.Normalize(mean, std)])

        test_transforms = torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor(),
             torchvision.transforms.Normalize(mean, std)])

        train_dataset = torchvision.datasets.CIFAR10(
            data_path, train=True, transform=train_transforms, download=False)
        test_dataset = torchvision.datasets.CIFAR10(
            data_path, train=False, transform=test_transforms, download=False)
    else:
        num_classes = 100

        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)

        train_transforms = torchvision.transforms.Compose(
            [torchvision.transforms.RandomCrop(32, padding=4),
             torchvision.transforms.RandomHorizontalFlip(),
             torchvision.transforms.ToTensor(),
             torchvision.transforms.Normalize(mean, std)])

        test_transforms = torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor(),
             torchvision.transforms.Normalize(mean, std)])

        train_dataset = torchvision.datasets.CIFAR100(
            data_path, train=True, transform=train_transforms, download=False)
        test_dataset = torchvision.datasets.CIFAR100(
            data_path, train=False, transform=test_transforms, download=False)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=train_batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=test_batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True)

    return (train_loader, test_loader), num_classes
