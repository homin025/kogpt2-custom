import random
import argparse
import numpy as np

import torch
import gluonnlp
from gluonnlp.data import SentencepieceTokenizer
from transformers import GPT2Config, GPT2LMHeadModel
from kogpt2.utils import download

import samples


pytorch_kogpt2 = {
    'url':
        'https://kobert.blob.core.windows.net/models/kogpt2/pytorch/pytorch_kogpt2_676e9bcfa7.params',
    'fname': 'pytorch_kogpt2_676e9bcfa7.params',
    'chksum': '676e9bcfa7'
}

tokenizer = {
    'url':
        'https://kobert.blob.core.windows.net/models/kogpt2/tokenizer/kogpt2_news_wiki_ko_cased_818bfa919d.spiece',
    'fname': 'kogpt2_news_wiki_ko_cased_818bfa919d.spiece',
    'chksum': '818bfa919d'
}

kogpt2_config = {
    "initializer_range": 0.02,
    "layer_norm_epsilon": 1e-05,
    "n_ctx": 1024,
    "n_embd": 768,
    "n_head": 12,
    "n_layer": 12,
    "n_positions": 1024,
    "vocab_size": 50000,
    "activation_function": "gelu"
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', type=str, default='./checkpoint/kogpt2_subject_epoch.ckpt')
    parser.add_argument('--length', type=int, default=250)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--top_k', type=int, default=40)
    parser.add_argument('--top_p', type=float, default=0.9)
    args = parser.parse_args()

    seed = random.randint(0, 2147483647)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cachedir = '~/kogpt2/'

    subject = 'subject'
    if args.load != './checkpoint/kogpt2_subject_epoch.ckpt':
        checkpoint = torch.load(args.load, map_location=device)
        subject = args.load.split('_')[1]
        args.load = None

    model = GPT2LMHeadModel(config=GPT2Config.from_dict(kogpt2_config))

    # download model
    if not args.load:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model_info = pytorch_kogpt2
        model_file = download(model_info['url'],
                              model_info['fname'],
                              model_info['chksum'],
                              cachedir=cachedir)
        model.load_state_dict(torch.load(model_file))

    model.to(device)
    model.eval()

    # download vocab
    vocab_info = tokenizer
    vocab_file = download(vocab_info['url'],
                          vocab_info['fname'],
                          vocab_info['chksum'],
                          cachedir=cachedir)

    vocab = gluonnlp.vocab.BERTVocab.from_sentencepiece(vocab_file,
                                                        mask_token='<msk>',
                                                        sep_token='<sep>',
                                                        cls_token='<cls>',
                                                        unknown_token='<unk>',
                                                        padding_token='<pad>',
                                                        bos_token='<s>',
                                                        eos_token='</s>')

    # download tokenizer
    token_info = tokenizer
    tokenizer = download(token_info['url'],
                         token_info['fname'],
                         token_info['chksum'],
                         cachedir=cachedir)

    tokenizer = SentencepieceTokenizer(tokenizer)

    while True:
        text = input('text: ')

        sentence = samples.sample_sequence(model, vocab, tokenizer, device, text, args.length, args.temperature, args.top_p, args.top_k)

        sentence = sentence.replace("<unused0>", "\n")
        print(sentence)