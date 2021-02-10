import os
import random
import argparse
import traceback
import numpy as np

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from tokenizers import SentencePieceBPETokenizer
from transformers import GPT2Config, GPT2LMHeadModel
from tqdm import tqdm

from data import CustomDataset, dynamic_padding_collate_fn, load_dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=-1)
    parser.add_argument('--batch_size', type=int, default=-1)
    parser.add_argument('--save', type=str, default='./checkpoint/')
    parser.add_argument('--load', type=str, default='./checkpoint/kogpt2_subject_epoch.ckpt')
    parser.add_argument('--train_dataset', type=str, default='./dataset/none_train.json', required=True)
    parser.add_argument('--valid_dataset', type=str, default='./dataset/none_valid.json')
    args = parser.parse_args()

    if args.epoch == -1:
        args.epoch = 10
    if args.batch_size == -1:
        args.batch_size = 1

    seed = random.randint(0, 2147483647)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{device} is used for training")

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

    train_dataset = None
    try:
        pairs = load_dataset(args.train_dataset)
        train_dataset = CustomDataset(pairs, tokenizer)
        print("loading train dataset succeeds")
    except Exception as e:
        print("loading train dataset fails")
        traceback.print_exc()
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=dynamic_padding_collate_fn)

    if args.valid_dataset == './dataset/none_valid.json':
        valid_flag = False
    else:
        valid_flag = True

    if valid_flag:
        valid_dataset = None
        try:
            pairs = load_dataset(args.valid_dataset)
            valid_dataset = CustomDataset(pairs, tokenizer)
            print("loading valid dataset succeeds")
        except Exception as e:
            print("loading valid dataset fails")
            traceback.print_exc()
        valid_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=dynamic_padding_collate_fn)

    model = GPT2LMHeadModel.from_pretrained(pretrained_model_name_or_path="taeminlee/kogpt2")

    if not args.load:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("loading model succeeds")

    model.to(device)
    model.eval()

    loss = 0
    epoch = 1
    learning_rate = 1e-5
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    # criterion = torch.nn.CrossEntropyLoss()

    if not args.load:
        epoch = checkpoint['epoch']
        learning_rate = checkpoint['learning_rate']
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    print("KoGPT2 Training Starts")

    for epoch in range(epoch, args.epoch + 1):
        best_epoch = 0
        best_loss = 10000

        average_train_loss = (0.0, 0.0)
        average_valid_loss = (0.0, 0.0)

        model.train()
        for step, batch in tqdm(enumerate(train_dataloader), desc=f"[TRAIN] Epoch: {epoch}", total=len(train_dataloader)):
            optimizer.zero_grad()

            input_ids, attention_mask, labels = tuple(value.to(device) for value in batch)
            outputs = model.forward(input_ids, attention_mask=attention_mask, labels=labels, return_dict=True)
            loss = outputs.loss.item()

            outputs.loss.backward()
            optimizer.step()

            average_train_loss = (average_train_loss[0] * 0.99 + loss, average_train_loss[1] * 0.99 + 1)

            if step % 10 == 0:
                print("[Epoch {0}: {1}] Loss = {2:.5f} Average Train loss = {3:.5f}".format(epoch, step, loss, average_train_loss[0] / average_train_loss[1]))

        # scheduler.step(average_loss[0] / average_loss[1])

        if valid_flag:
            model.eval()
            for batch in tqdm(valid_dataloader, desc="[EVALUATE]"):
                with torch.no_grad():
                    input_ids, attention_mask, labels = tuple(value.to(device) for value in batch)
                    outputs = model.forward(input_ids, attention_mask=attention_mask, labels=labels, return_dict=True)
                    loss = outputs.loss.item()

                    average_valid_loss = (average_valid_loss[0] * 0.99 + loss, average_valid_loss[1] * 0.99 + 1)

            print("[Epoch {0}] Average Valid loss = {1:.5f}".format(epoch, average_valid_loss[0] / average_valid_loss[1]))

            if best_loss > average_valid_loss[0] / average_valid_loss[1]:
                best_loss = average_valid_loss[0] / average_valid_loss[1]
                best_epoch = epoch

            print("[Epoch {0}] Best Epcoh {1} Best loss = {2:.5f}".format(epoch, best_epoch, best_loss))

        if epoch % 2 == 0:
            try:
                if not os.path.exists(args.save):
                    os.mkdir(args.save)

                torch.save({
                    'epoch': epoch,
                    'learning_rate': learning_rate,
                    'model_state_dict': model.state_dict(),
                    # 'optimizer_state_dict': optimizer.state_dict()
                    # 'scheduler_state_dict': scheduler.state_dict()
                }, args.save + 'kogpt2_' + subject + '_' + str(epoch) + '.ckpt')
                print("saving model succeeds")
            except Exception as e:
                traceback.print_exc()
                print("saving model fails")
                exit(0)

    torch.save({
        'epoch': epoch,
        'learning_rate': learning_rate,
        'model_state_dict': model.state_dict(),
        # 'optimizer_state_dict': optimizer.state_dict()
        # 'scheduler_state_dict': scheduler.state_dict()
    }, args.save + 'kogpt2_' + subject + '_' + str(args.epoch + 1) + '.ckpt')
    