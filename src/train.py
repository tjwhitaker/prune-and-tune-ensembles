import argparse
from data import get_loaders


model = WideResNetBase(28, 10, 0, 10).to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(
    filter(lambda param: param.requires_grad, model.parameters()),
    lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)

for i in range(140):
    train_loss = 0
    test_loss = 0
    lr = learning_rate_schedule(0.1, i, 140)
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

            # prediction = torch.argmax(output)
            _, predictions = torch.max(output, -1)

            num_correct += (predictions == target).sum().data.item()

            # if prediction == target:
            #     num_correct += 1

            num_test_batches += 1

        accuracy = (num_correct / len(test_dataset)) * 100

    print(f"{train_loss / num_train_batches}, {test_loss / num_test_batches}, {accuracy}")


# torch.save(model.state_dict(), f"checkpoints/wrn_10.pt")


# # CHILDREN
# ##########

for run in range(3):
    model = WideResNetBase(28, 10, 0, 100).to(device)
    model.load_state_dict(torch.load(f"checkpoints/wrn_100_3.pt"))

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

    model.to('cpu')

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
