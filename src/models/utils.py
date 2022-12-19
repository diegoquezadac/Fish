import torch
import pandas as pd
import time
import logging

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

def train_classifier(model, train_dataloader, val_dataloader, epochs=10, step_size=20, gamma=0.1, log_file=None):

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    log_interval = 1000

    if(log_file):
        logging.basicConfig(filename=log_file, encoding='utf-8', level=logging.DEBUG)

    # For each epoch
    for epoch in range(1, epochs + 1):

        model.train()
        epoch_acc, total_count = 0, 0
        start_time = time.time()
        acc_per_epoch, count_per_apoch = [], []

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
                    "| accuracy {:8.6f}".format(
                        epoch, idx, len(train_dataloader), epoch_acc / total_count
                    )
                )
                acc_per_epoch.append(epoch_acc)
                count_per_apoch.append(total_count)
                epoch_acc, total_count = 0, 0
                start_time = time.time()
        
        # Evaluate the model on the validation set
        train_acc = sum(acc_per_epoch) / sum(count_per_apoch)
        val_acc = evaluate_classifier(model, val_dataloader, criterion)

        print('-' * 100)
        print('| end of epoch {:3d} | time: {:5.2f}s | '
            'train accuracy {:8.6f} | validation accuracy {:8.6f} '.format(epoch,
                                            time.time() - start_time,
                                            train_acc,
                                            val_acc))
        if(log_file):
            logging.info('| epoch {:3d} | time: {:5.2f}s | '
            'train accuracy {:8.6f} | validation accuracy {:8.6f} '.format(epoch,
                                            time.time() - start_time,
                                            train_acc,
                                            val_acc))
        print('-' * 100)

        scheduler.step()

def train_autoencoder(model, train_dataloader, val_dataloader, epochs=10, step_size=20, log_file=None):

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=5.0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma=0.1)

    if(log_file):
        logging.basicConfig(filename=log_file, encoding='utf-8', level=logging.DEBUG)

    # For each epoch
    for epoch in range(1, epochs + 1):

        start_time = time.time()
        model.train()
        epoch_loss, total_count = 0, 0
        loss_per_epoch, count_per_apoch = [], []
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
                    "| loss {:8.6f}".format(
                        epoch, idx, len(train_dataloader), epoch_loss / total_count
                    )
                ) 
                loss_per_epoch.append(epoch_loss)
                count_per_apoch.append(total_count)
                epoch_loss, total_count = 0, 0
                start_time = time.time()
        
        # Evaluate the model on the validation set
        train_loss = sum(loss_per_epoch) / sum(count_per_apoch)
        val_loss = evaluate_autoencoder(model, val_dataloader, criterion)

        print('-' * 59)
        print('| end of epoch {:3d} | time: {:5.2f}s | '
            'train loss {:8.6f} | validation loss {:8.6f} '.format(epoch,
                                            time.time() - start_time,
                                            train_loss,
                                            val_loss))
        if(log_file):
            logging.info('| epoch {:3d} | time: {:5.2f}s | '
            'train loss {:8.6f} | validation loss {:8.6f} '.format(epoch,
                                            time.time() - start_time,
                                            train_loss,
                                            val_loss))
        print('-' * 59)

        scheduler.step()