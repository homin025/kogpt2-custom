import os
import random
import argparse
import traceback
import numpy as np

import torch
from tokenizers import SentencePieceBPETokenizer
from transformers import GPT2Config, GPT2LMHeadModel

import samples


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--length', type=int, default=250)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--top_k', type=int, default=40)
    parser.add_argument('--top_p', type=float, default=0.9)
    parser.add_argument('--load', type=str, default='./checkpoint/kogpt2_subject_epoch.ckpt')
    args = parser.parse_args()

    seed = random.randint(0, 2147483647)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{device} is used for testing")

    subject = 'subject'

    if args.load != './checkpoint/kogpt2_subject_epoch.ckpt':
        checkpoint = torch.load(args.load, map_location=device)
        subject = args.load.split('_')[1]
        args.load = None

    tokenizer = SentencePieceBPETokenizer.from_file(
        vocab_filename="./tokenizer/tokenizers_vocab.json",
        merges_filename="./tokenizer/tokenizers_merges.txt",
        add_prefix_space=False
    )

    model = GPT2LMHeadModel.from_pretrained(pretrained_model_name_or_path="taeminlee/kogpt2")

    if not args.load:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("loading model succeeds")

    model.to(device)
    model.eval()

    print('Enter "quit" to quit')
    while True:
        content = input('input: ')
        count = 1

        if content == 'quit':
            break
        else:
            sentences = samples_tk.sample_sequence_sentence(model, tokenizer, device, content, args.temperature, args.top_k, args.top_p, count)
            print('output:', content + sentences[0])
    