from tqdm import tqdm

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
                    batches_per_epoch = 100, step_sizes = []):
    model.train()
    train_loss = 0
    correct = torch.zeros(10)
    kept = torch.zeros(10)
    for batch_idx, (data, target) in enumerate(train_loader):
        if batch_idx < batches_per_epoch * (epoch-1):
            continue
        if batch_idx >= batches_per_epoch * epoch:
            break
        if step_sizes:
            optimizer.param_groups[0]['lr'] = step_sizes[target]
        train_loss, correct, kept = minibatch_step(model, device, optimizer, data, target, train_loss, kept, correct)
    return train_loss / kept.sum(), correct / kept, kept


def train_streaming_unbalanced(model, device, train_loader, optimizer, epoch,
                               experimental_dist, instrumental_dist, max_scaler,
                               batches_per_epoch = 100, lr_scaler = False, prob_func = lambda prob: prob):
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
        if mask.sum() == 0:
            continue
        data = data[mask]
        target = target[mask]
        if lr_scaler:
            probs = experimental_dist(torch.arange(0, 10))
            optimizer.param_groups[0]['lr'] = prob_func(probs[target]).item()
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


def train_and_test_streaming(model, optimizer, train_loader, test_loader,
                             epochs = 10, batches_per_epoch = 5000, step_sizes = [], device = 'cpu'):
    kept_all = torch.zeros(10)
    train_accs = []
    test_accs = []

    for epoch in tqdm(range(1, epochs + 1)):
        train_loss, train_acc, kept = train_streaming(model, device, train_loader, optimizer, epoch,
                                                      batches_per_epoch, step_sizes)
        test_loss, test_acc = test(model, device, test_loader)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        kept_all = kept_all + kept

    return train_accs, test_accs, kept_all
