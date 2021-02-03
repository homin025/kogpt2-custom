import argparse

import torch
from transformers import GPT2LMHeadModel

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', type=str, required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.load, map_location=device)
    model_state_dict = checkpoint['model_state_dict']

    model = GPT2LMHeadModel.from_pretrained(pretrained_model_name_or_path="taeminlee/kogpt2")
    model.load_state_dict(model_state_dict)

    torch.save(model.state_dict(), args.load)
