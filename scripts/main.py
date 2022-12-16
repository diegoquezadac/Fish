import os
import torch
import pandas as pd
from ray.air import session
from torchtext.data.utils import get_tokenizer
from src.data.dataset import CustomDataset
from torch.utils.data import DataLoader
from functools import partial
from ray import tune
from src.data.utils import collate_batch
from src.models.fish import Fish
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.air.checkpoint import Checkpoint
from torchtext.data.utils import get_tokenizer

def load_data(data_dir):

    # Load dataframes
    df_train = pd.read_csv(f"{data_dir}/train.csv")
    df_val = pd.read_csv(f"{data_dir}/val.csv")
    df_test = pd.read_csv(f"{data_dir}/test.csv")

    # Load tokenizer
    tokenizer = get_tokenizer('spacy', language='en_core_web_sm')

    # Create datasets
    train_dataset = CustomDataset(
    df_train["text"].values.tolist(), df_train["toxic"].values.tolist(), tokenizer
    )
    val_dataset = CustomDataset(
        df_val["text"].values.tolist(), df_val["toxic"].values.tolist(), tokenizer
    )
    test_dataset = CustomDataset(
        df_test["text"].values.tolist(), df_test["toxic"].values.tolist(), tokenizer
    )

    return train_dataset, val_dataset, test_dataset

def evaluate_fish(model, dataloader, criterion):
    model.eval()
    eval_acc, eval_loss, total_count = 0, 0

    with torch.no_grad():
        for idx, (text, label, offsets) in enumerate(dataloader):
            predicted_label = model(text, offsets)
            loss = criterion(predicted_label, label)
            eval_acc += (predicted_label.argmax(1) == label).sum().item()
            eval_loss += loss.item()
            total_count += label.size(0)
    return eval_acc / total_count, eval_loss / total_count

def train_fish(config, data_dir=None):

    # Set training parameters
    num_class = 2
    train_dataset, val_dataset, test_dataset = load_data(data_dir)
    tokenizer = get_tokenizer("spacy", language="en_core_web_sm")
    vocab = torch.load("/Users/diego/dev/fish/data/vocab.pt")
    vocab_size = len(vocab)

    model = Fish(vocab_size, config["embed_dim"], num_class, n3=config["n3"])
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=5.0, momentum=0.9)

    # Checkpoint state
    loaded_checkpoint = session.get_checkpoint()
    if loaded_checkpoint:
        with loaded_checkpoint.as_directory() as loaded_checkpoint_dir:
           model_state, optimizer_state = torch.load(os.path.join(loaded_checkpoint_dir, "checkpoint.pt"))
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    # Get DataLoader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        collate_fn=lambda x: collate_batch(batch=x, vocab=vocab, tokenizer=tokenizer),
        sampler=train_dataset.get_sampler(),
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        collate_fn=lambda x: collate_batch(batch=x, vocab=vocab, tokenizer=tokenizer),
        sampler=val_dataset.get_sampler(),
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        collate_fn=lambda x: collate_batch(batch=x, vocab=vocab, tokenizer=tokenizer),
    )

    # Train model
    EPOCHS = 1
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)
    total_acc = None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_acc, total_count = 0, 0    
        for idx, (text, label, offsets) in enumerate(train_dataloader):
            optimizer.zero_grad()
            predicted_label = model(text, offsets)
            loss = criterion(predicted_label, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()
            train_acc += (predicted_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
            if idx % 20 == 0:  # print every 2000 mini-batches
                print("[%d, %5d] accuracy: %.3f" % (epoch + 1, idx + 1, train_acc / total_count))

        # Evaluate model
        val_acc, val_loss = evaluate_fish(model, val_dataloader, criterion)
        if total_acc is not None and total_acc > val_acc:
            scheduler.step()
        else:
            total_acc = val_acc
        
        # Checkpoint
        os.makedirs("fish", exist_ok=True)  
        torch.save(
            (model.state_dict(), optimizer.state_dict()), "fish/checkpoint.pt")
        checkpoint = Checkpoint.from_directory("fish")

        # Report results
        session.report({"loss": val_loss, "accuracy": val_acc}, checkpoint=checkpoint)

def main(num_samples=3, max_num_epochs=10):

    config = {
        "embed_dim": tune.choice([64, 128, 256]),
        "n3": tune.choice([128,256]),
        "batch_size": tune.choice([32, 64]),
    }

    scheduler = ASHAScheduler(
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)

    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(partial(train_fish, data_dir="/Users/diego/dev/fish/data/processed")),
            resources={"cpu": 2}
        ),
        tune_config=tune.TuneConfig(
            metric="loss",
            mode="min",
            scheduler=scheduler,
            num_samples=num_samples,
        ),
        param_space=config,
    )   
    results = tuner.fit()

    best_result = results.get_best_result("loss", "min")

    print("Best trial config: {}".format(best_result.config))
    print("Best trial final validation loss: {}".format(
        best_result.metrics["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_result.metrics["accuracy"]))

    #evaluate_fish(best_result)

if __name__ == "__main__":

    main(num_samples=1, max_num_epochs=1)