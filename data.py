import numpy as np
import pandas as pd


class Dataset:
    def __init__(self, path, vocab, tokenizer):
        self.path = path
        self.vocab = vocab
        self.tokenizer = tokenizer

        self.data = []

        """ START: Customization for dataset is necessary """
        file = open(self.path, 'r', encoding='utf-8')
        # df = pd.read_csv(self.path)
        # df = pd.read_excel(self.path)

        lines = file.readlines()
        for line in lines:
            tokenized_line = tokenizer(line)

            data = [vocab[vocab.bos_token], ] + vocab[tokenized_line] + [vocab[vocab.eos_token]]
            # data = [vocab[vocab.bos_token], ] + train + '[vocab[vocab.sep_token], ] + target + [vocab[vocab.eos_token]

            self.data.append(data)
        """ END: Customization for dataset is necessary """

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
