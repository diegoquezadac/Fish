import torch
from torchtext.vocab import build_vocab_from_iterator

def collate_batch(batch, vocab, tokenizer):
    text_pipeline = lambda x: vocab(tokenizer(str(x).lower()))
    label_pipeline = lambda x: x
    label_list, text_list, offsets = [], [], [0]
    for (_text,_label) in batch:
         label_list.append(label_pipeline(_label))
         processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
         text_list.append(processed_text)
         offsets.append(processed_text.size(0))
    label_list = torch.tensor(label_list, dtype=torch.int64)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text_list = torch.cat(text_list)
    return text_list, label_list, offsets

def yield_tokens(iterator, tokenizer):
    for text,_ in iterator:
        yield tokenizer(str(text))

def get_vocab(dataset, tokenizer):
    train_iterator = iter(dataset)
    vocab = build_vocab_from_iterator(yield_tokens(train_iterator, tokenizer), specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])
    return vocab