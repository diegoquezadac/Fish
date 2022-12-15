from torchtext.data.utils import get_tokenizer
from src.data.dataset import CustomDataset
import pandas as pd
    
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