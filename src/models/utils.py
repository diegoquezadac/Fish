import torch
import pandas as pd
import time

def _train_epoch_classifier(model, dataloader, optimizer, criterion, epoch):
    model.train()
    total_acc, total_count = 0, 0
    log_interval = 1000
    start_time = time.time()

    for idx, (text, label, offsets) in enumerate(dataloader):
        optimizer.zero_grad()
        predicted_label = model(text, offsets)
        loss = criterion(predicted_label, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        total_acc += (predicted_label.argmax(1) == label).sum().item()
        total_count += label.size(0)
        if idx % log_interval == 0 and idx > 0:
            elapsed = time.time() - start_time
            print(
                "| epoch {:3d} | {:5d}/{:5d} batches "
                "| accuracy {:8.3f}".format(
                    epoch, idx, len(dataloader), total_acc / total_count
                )
            )
            total_acc, total_count = 0, 0
            start_time = time.time()
    return model

def evaluate_classifier(model, dataloader, criterion):
    model.eval()
    total_acc, total_count = 0, 0

    with torch.no_grad():
        for idx, (text, label, offsets) in enumerate(dataloader):
            predicted_label = model(text, offsets)
            loss = criterion(predicted_label, label)
            total_acc += (predicted_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
    return total_acc / total_count

def train_classifier(model, train_dataloader, val_dataloader):
    EPOCHS = 10
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=5.0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)
    total_accu = None

    for epoch in range(1, EPOCHS + 1):
        epoch_start_time = time.time()
        model = _train_epoch_classifier(model, train_dataloader, optimizer, criterion, epoch)
        accu_val = evaluate_classifier(model, val_dataloader, criterion)
        if total_accu is not None and total_accu > accu_val:
            scheduler.step()
        else:
            total_accu = accu_val
        print('-' * 59)
        print('| end of epoch {:3d} | time: {:5.2f}s | '
            'valid accuracy {:8.3f} '.format(epoch,
                                            time.time() - epoch_start_time,
                                            accu_val))
        print('-' * 59)