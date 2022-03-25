import torch
import torchvision
<<<<<<< HEAD
import numpy as np
=======
>>>>>>> 8d7ab12caf393fc44a94e427c450b2ee6150a206


def get_loaders(dataset='cifar10', data_path='data', train_batch_size=128, test_batch_size=1, num_workers=4):
    if dataset == 'cifar10':
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
            data_path, train=True, transform=train_transforms, download=True)
        test_dataset = torchvision.datasets.CIFAR10(
            data_path, train=False, transform=test_transforms, download=True)
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
            data_path, train=True, transform=train_transforms, download=True)
        test_dataset = torchvision.datasets.CIFAR100(
            data_path, train=False, transform=test_transforms, download=True)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=train_batch_size,
        num_workers=num_workers,
<<<<<<< HEAD
        shuffle=True)
=======
        shuffle=True,
        pin_memory=True)
>>>>>>> 8d7ab12caf393fc44a94e427c450b2ee6150a206
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=test_batch_size,
        num_workers=num_workers,
<<<<<<< HEAD
        shuffle=False)

    return train_loader, test_loader, num_classes


def get_corrupted_loaders(dataset='cifar10', data_path='data', train_batch_size=128, test_batch_size=1, num_workers=4):
    corruptions = [
        "brightness",
        "contrast",
        "defocus_blur",
        "elastic_transform",
        "fog",
        "frost",
        "gaussian_blur",
        "gaussian_noise",
        "glass_blur",
        "impulse_noise",
        "jpeg_compression",
        "motion_blur",
        "pixelate",
        "saturate",
        "shot_noise",
        "snow",
        "spatter",
        "speckle_noise",
        "zoom_blur"
    ]

    if dataset == "cifar10":
        num_classes = 10
        labels_path = "data/CIFAR-10-C/labels.npy"

        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2471, 0.2435, 0.2616)

        images = []

        for c in corruptions:
            data = torch.from_numpy(
                np.load(f"data/CIFAR-10-C/{c}.npy")).permute(0, 3, 1, 2) / 255.0
            images.append(data)

        images = torch.cat(images, dim=0)

        for image in images:
            image[0] = (image[0] - mean[0]) / std[0]
            image[1] = (image[1] - mean[1]) / std[1]
            image[2] = (image[2] - mean[2]) / std[2]

    else:
        num_classes = 100
        labels_path = "data/CIFAR-100-C/labels.npy"

        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)

        images = []

        for c in corruptions:
            data = torch.from_numpy(
                np.load(f"data/CIFAR-100-C/{c}.npy")).permute(0, 3, 1, 2) / 255.0
            images.append(data)

        images = torch.cat(images, dim=0)

        for image in images:
            image[0] = (image[0] - mean[0]) / std[0]
            image[1] = (image[1] - mean[1]) / std[1]
            image[2] = (image[2] - mean[2]) / std[2]

    labels = np.load(labels_path)
    labels = np.repeat([labels], len(corruptions), axis=0).flatten()

    dataset = torch.utils.data.TensorDataset(images, torch.from_numpy(labels))

    test_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=test_batch_size,
        num_workers=num_workers,
        shuffle=False)

    return test_loader, labels, num_classes
=======
        shuffle=False,
        pin_memory=True)

    return (train_loader, test_loader), num_classes
>>>>>>> 8d7ab12caf393fc44a94e427c450b2ee6150a206
