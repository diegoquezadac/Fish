import pandas as pd
import spacy
from sklearn.model_selection import train_test_split

def preprocess(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:

    nlp = spacy.load("en_core_web_sm", disable=["ner"])

    LABEL_COLUMNS = df.columns.tolist()[2:]

    train_toxic = df[df[LABEL_COLUMNS].sum(axis=1) > 0]
    train_clean = df[df[LABEL_COLUMNS].sum(axis=1) == 0]

    train_toxic = train_toxic.assign(toxic=1)
    train_clean = train_clean.assign(toxic=0)

    #df = pd.concat([train_toxic, train_clean.sample(train_toxic.shape[0])])
    df = pd.concat([train_toxic, train_clean])[["text", "toxic"]]
    
    n = df.shape[0]
    column = []
    for i, doc in enumerate(nlp.pipe(df["text"])):
        tokens = [token.text.lower() for token in doc if token.is_alpha]
        column.append(" ".join(tokens))
        if verbose and i % 5000 == 0:
            print(f"Loading {100 * i/n}%...")

    df["text"] = column
    df.reset_index(inplace=True, drop=True)

    return df

def main():
    # Extract
    df_train = pd.read_csv("../../data/raw/kaggle/train.csv")

    df_test = pd.read_csv("../../data/raw/kaggle/test.csv")
    df_test_labels = pd.read_csv("../../data/raw/kaggle/test_labels.csv")
    df_test = df_test.merge(df_test_labels, on="id", how="left")

    # Concat
    df_raw = pd.concat([df_train, df_test]).rename(columns={"comment_text": "text"})

    # Transform
    df = preprocess(df_raw, verbose=True)

    # Split
    df_train, df_val = train_test_split(df, test_size=0.2, random_state=21)
    df_val, df_test = train_test_split(df_val, test_size=0.5, random_state=21)

    # Load
    df_train.to_csv("../../data/processed/train.csv", index=False)
    df_val.to_csv("../../data/processed/val.csv", index=False)
    df_test.to_csv("../../data/processed/test.csv", index=False)


if(__name__ == "__main__"):
    main()