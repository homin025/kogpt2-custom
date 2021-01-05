import os
import sys
import random
import requests
import argparse
import numpy as np

import torch
import gluonnlp
from torch.utils.data import DataLoader
from gluonnlp.data import SentencepieceTokenizer
from transformers import GPT2Config, GPT2LMHeadModel
from kogpt2.utils import download
from tensorboardX import SummaryWriter

import data

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
    parser.add_argument('--epoch', type=int, default=-1)
    parser.add_argument('--batch_size', type=int, default=-1)
    parser.add_argument('--dataset', type=str, default='./dataset/')
    parser.add_argument('--save_model', type=str, default='./checkpoint/')
    parser.add_argument('--load_model', type=str, default='./checkpoint/kogpt2_subject_epoch.ckpt')
    args = parser.parse_args()

    if args.epoch == -1:
        args.epoch = 10
    if args.batch_size == -1:
        args.batch_size = 1

    summary = SummaryWriter()

    seed = random.randint(0, 2147483647)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cachedir = '~/kogpt2/'

    subject = 'subject'
    if args.load_model != './checkpoint/kogpt2_subject_epoch.ckpt':
        checkpoint = torch.load(args.load_model, map_location=device)
        subject = args.load_model.split('_')[1]
        args.load_model = None

    model = GPT2LMHeadModel(config=GPT2Config.from_dict(kogpt2_config))

    # download model
    if not args.load_model:
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

    try:
        dataset = data.Dataset(args.dataset, vocab, tokenizer)
        print("loading dataset succeeds")
    except Exception as e:
        print("loading dataset fails")
        exit(0)
    finally:
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)

    loss = 0
    epoch = 0
    learning_rate = 1e-5
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    # criterion = torch.nn.CrossEntropyLoss()

    if not args.load_model:
        epoch = checkpoint['epoch']
        learning_rate = checkpoint['learning_rate']
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    print("KoGPT2 Training Starts")
    model.train()

    iter = 0
    average_loss = (0.0, 0.0)

    for epoch in range(epoch, args.epoch):
        for data in dataloader:
            optimizer.zero_grad()

            data = torch.stack(data)
            data = data.transpose(1, 0)
            data = data.to(device)

            outputs = model(data, labels=data)
            loss, logits = outputs[:2]
            loss = loss.to(device)
            loss.backward()
            optimizer.step()
            scheduler.step(loss)

            average_loss = (average_loss[0] * 0.99 + loss, average_loss[1] * 0.99 + 1)

            if iter % 10 == 0:
                print("[Epoch {0}: {1}] Loss = {2:.5f} Average loss = {3:.5f}".format(epoch, iter, loss,
                                                                                      average_loss[0] / average_loss[
                                                                                          1]))
            iter += 1

        if epoch % 5 == 0:
            try:
                if not os.path.exists(args.save_path):
                    os.mkdir(args.save_path)

                torch.save({
                    'epoch': epoch,
                    'learning_rate': learning_rate,
                    'model_state_dict': model.state_dict(),
                    'optimzer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict()
                }, args.save_path + 'kogpt2_' + subject + '_' + str(epoch) + '.ckpt')
            except Exception as e:
                print("saving model fails")
                exit(0)