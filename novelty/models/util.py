import torch
import torch.nn.functional as F


def forward_pass(model, data, target):
    output = model(data)
    loss = F.nll_loss(output, target)
    pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    return loss, pred


def itemwise_accuracy(pred, target):
    is_correct = pred.eq(target.view_as(pred))
    correct_digits = torch.bincount(target[is_correct.flatten()],minlength=10)
    return correct_digits


def minibatch_step(model, device, optimizer, data, target,
                   train_loss, kept, correct):
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()
    loss, pred = forward_pass(model, data, target)
    loss.backward()
    optimizer.step()
    train_loss = train_loss + loss.item()
    correct = correct + itemwise_accuracy(pred, target)
    kept = kept + torch.bincount(target,minlength=10)
    return train_loss, correct, kept


def train(model, device, train_loader, optimizer):
    model.train()
    train_loss = 0
    correct = torch.zeros(10)
    kept = torch.zeros(10)
    for batch_idx, (data, target) in enumerate(train_loader):
        train_loss, correct, kept = minibatch_step(model, device, optimizer, data, target, train_loss, kept, correct)
    return train_loss / kept.sum(), correct / kept, kept


def train_streaming(model, device, train_loader, optimizer, epoch,
                    batches_per_epoch = 100):
    model.train()
    train_loss = 0
    correct = torch.zeros(10)
    kept = torch.zeros(10)
    for batch_idx, (data, target) in enumerate(train_loader):
        if batch_idx < batches_per_epoch * (epoch-1):
            continue
        if batch_idx >= batches_per_epoch * epoch:
            break
        train_loss, correct, kept = minibatch_step(model, device, optimizer, data, target, train_loss, kept, correct)
    return train_loss / kept.sum(), correct / kept, kept


def train_streaming_unbalanced(model, device, train_loader, optimizer, epoch,
                               experimental_dist, instrumental_dist, max_scaler,
                               batches_per_epoch = 100):
    model.train()
    train_loss = 0
    correct = torch.zeros(10)
    kept = torch.zeros(10)
    for batch_idx, (data, target) in enumerate(train_loader):
        if batch_idx < batches_per_epoch * (epoch-1):
            continue
        if batch_idx >= batches_per_epoch * epoch:
            break
        mask = instrumental_dist(target, max_scaler) < experimental_dist(target)
        data = data[mask]
        target = target[mask]
        train_loss, correct, kept = minibatch_step(model, device, optimizer, data, target, train_loss, kept, correct)
    return train_loss / kept.sum(), correct / kept, kept


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = torch.zeros(10)
    kept = torch.zeros(10)
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            loss, pred = forward_pass(model, data, target)
            test_loss += loss.sum().item()
            correct = correct + itemwise_accuracy(pred, target)
            kept = kept + torch.bincount(target,minlength=10)
    return test_loss / kept.sum(), correct / kept
