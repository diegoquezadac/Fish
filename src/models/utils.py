import torch
import pandas as pd
import time

def evaluate_classifier(model, dataloader, criterion):
    model.eval()
    total_acc, total_loss, total_count = 0, 0, 0
    with torch.no_grad():
        for _, (text, label, offsets) in enumerate(dataloader):
            predicted_label = model(text, offsets)
            loss = criterion(predicted_label, label)
            total_acc += (predicted_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
            total_loss += loss.item()
    return total_acc / total_count

def evaluate_autoencoder(model, dataloader, criterion):
    model.eval()
    total_loss, total_count = 0, 0

    with torch.no_grad():
        for _, (text, label, offsets) in enumerate(dataloader):
            reconstructued = model(text, offsets)
            loss = criterion(reconstructued, model.embed(text, offsets))
            total_loss += loss.item()
            total_count += label.size(0)
    return total_loss / total_count

def train_classifier(model, train_dataloader, val_dataloader, epochs=10):

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=5.0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)
    total_acc = None
    log_interval = 1000

    # For each epoch
    for epoch in range(1, epochs + 1):

        model.train()
        epoch_acc, total_count = 0, 0
        start_time = time.time()

        # For each batch in the training set
        for idx, (text, label, offsets) in enumerate(train_dataloader):
            optimizer.zero_grad()
            predicted_label = model(text, offsets)
            loss = criterion(predicted_label, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()
            epoch_acc += (predicted_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
            if idx % log_interval == 0 and idx > 0:
                print(
                    "| epoch {:3d} | {:5d}/{:5d} batches "
                    "| accuracy {:8.3f}".format(
                        epoch, idx, len(train_dataloader), epoch_acc / total_count
                    )
                )
                epoch_acc, total_count = 0, 0
                start_time = time.time()
        
        # Evaluate the model on the validation set
        val_acc = evaluate_classifier(model, val_dataloader, criterion)

        if total_acc is not None and total_acc > val_acc:
            scheduler.step()
        else:
            total_acc = val_acc

        print('-' * 59)
        print('| end of epoch {:3d} | time: {:5.2f}s | '
            'valid accuracy {:8.3f} '.format(epoch,
                                            time.time() - start_time,
                                            val_acc))
        print('-' * 59)

def train_autoencoder(model, train_dataloader, val_dataloader, epochs=10):

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=5.0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)
    total_accu = None

    # For each epoch
    for epoch in range(1, epochs + 1):

        start_time = time.time()
        model.train()
        epoch_loss, total_count = 0, 0
        log_interval = 1000

        # For each batch in the training set
        for idx, (text, label, offsets) in enumerate(train_dataloader):
            optimizer.zero_grad()
            reconstructued = model(text, offsets)
            loss = criterion(reconstructued, model.embed(text, offsets))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()
            epoch_loss += loss.item()
            total_count += label.size(0)
            if idx % log_interval == 0 and idx > 0:
                elapsed = time.time() - start_time
                print(
                    "| epoch {:3d} | {:5d}/{:5d} batches "
                    "| loss {:8.3f}".format(
                        epoch, idx, len(train_dataloader), epoch_loss / total_count
                    )
                )
                epoch_loss, total_count = 0, 0
                start_time = time.time()
        
        # Evaluate the model on the validation set
        val_acc = evaluate_autoencoder(model, val_dataloader, criterion)
        if total_accu is not None and total_accu > val_acc:
         scheduler.step()
        else:
            total_accu = val_acc

        print('-' * 59)
        print('| end of epoch {:3d} | time: {:5.2f}s | '
            'valid accuracy {:8.3f} '.format(epoch,
                                            time.time() - start_time,
                                            val_acc))
        print('-' * 59)