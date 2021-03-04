import json
from typing import List, Tuple, NamedTuple, Optional

import torch
from tokenizers import SentencePieceBPETokenizer
from torch.utils.data import Dataset


GPTInputsType = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
GPTFeaturesType = Tuple[List[int], List[float], List[int]]

MAX_SUMMARY_SPACE = 200
MIN_SUMMARY_SPACE = 20


class Pair(NamedTuple):
    context: str
    summary: str


class CustomDataset(Dataset):
    def __init__(self, pairs: List[Pair], tokenizer: SentencePieceBPETokenizer):
        self.pairs = pairs
        self.tokenizer = tokenizer

        self.bos_token = tokenizer.token_to_id("<s>")
        self.eos_token = tokenizer.token_to_id("</s>")
        self.task_token = tokenizer.encode("요약:").ids

    def __getitem__(self, index: int) -> GPTFeaturesType:
        pair = self.pairs[index]

        train_tokens = self.tokenizer.encode(f"문맥:{pair.context}").ids
        target_tokens = self.tokenizer.encode(f"{pair.summary}").ids

        len_train_tokens = 1 + len(train_tokens) + len(self.task_token)
        len_target_tokens = len(target_tokens) + 1

        if len_train_tokens + len_target_tokens >= 1000:
            len_exceed_tokens = len_train_tokens + len_target_tokens - 1000
            train_tokens = train_tokens[:len_train_tokens - len_exceed_tokens]

        input_ids = [self.bos_token] + train_tokens + self.task_token + target_tokens + [self.eos_token]
        labels = input_ids
        attention_mask = [1.0] * len(input_ids)

        return input_ids, attention_mask, labels

    def __len__(self) -> int:
        return len(self.pairs)


def dynamic_padding_collate_fn(features: List[GPTFeaturesType]) -> GPTInputsType:
    max_seq_len = max([len(feature[0]) for feature in features])
    input_ids, attention_mask, labels = [], [], []

    for feature in features:
        padded_input_ids = feature[0] + [0] * (max_seq_len - len(feature[0]))
        padded_attention_mask = feature[1] + [0.0] * (max_seq_len - len(feature[1]))
        padded_labels = feature[2] + [-100] * (max_seq_len - len(feature[2]))

        input_ids.append(padded_input_ids)
        attention_mask.append(padded_attention_mask)
        labels.append(padded_labels)

    return torch.tensor(input_ids), torch.tensor(attention_mask), torch.tensor(labels)


def load_dataset(path: str) -> List[Pair]:
    with open(path, encoding="utf-8") as f:
        json_file = json.load(f)

    pairs = []
    for document in json_file['document']:
        context = ' '.join(document['body'])
        summary = ' '.join(document['abstract_sentences'])
        pair = Pair(context, summary)
        pairs.append(pair)

    return pairs