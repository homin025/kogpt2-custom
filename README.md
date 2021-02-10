# KoGPT2 Custom

## Train
``` python
python train.py --epoch 200 --batch_size 1 --save ./checkpoint/ --load ./checkpoint/kogpt2_subject_epoch.ckpt --data ./dataset/data
```

- `--epoch` : number of epochs
- `--batch_size` : number of batch size
- `--save` : directory to save trained model for every 5 epochs (ex: ./checkpoint/)
- `--load` : directory to load pretrained model (ex: ./checkpoint/kogpt2_summarization_150.ckpt)
- `--train_dataset` [required] : directory to load dataset (ex: ./dataset/data)
- `--valid_dataset` : directory to load dataset (ex: ./dataset/data)
- Notice: You should CUSTOMIZE train.py and data.py suitable to your own dataset.


## Generate
``` python
python generate.py --load ./checkpoint/kogpt2_subject_epoch.ckpt --length 250 --temperature 0.7 --top_k 40 --top_p 0.9
```

- `--load` : directory to load pretrained model (ex: ./checkpoint/kogpt2_summarization_150.ckpt)
- `--length` : maximum length of generated text (default: 250)
- `--temperature` : the thermodynamic temperature in distribution (default: 0.7)
- `--top_k` : value of top_k (default: 40)
- `--top_p` : value of top_p (default: 0.9)
