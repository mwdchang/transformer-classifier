# Transformer classifier
A transformer based classifier that conveniently uses your file system as training datasets, all you need is to place data into appropriatedly named folders.


## Setting up and running the classifier
Create a folder with subfolders, each representing a class, for example a cuisine classifier:

```
receipes
  |- vietnamese
    |- pho.txt
    |- banhmi.txt
    |- ...
  |- spanish
    |- paella.txt
    |- gazpacho.txt
    |- ...
  |- icelandic
    |- ...
```

Once you have a good sized sample, you can run the training and inference processes

Either train with a pre-trained model or from scratch

```
# pre-trained
#  python train.py <data_folder>
python train.py receipes

# from scratch
#  python train_raw.py <data_folder>
python train_raw.py receipes

```

Once training is complete, it will dump the model config/parameters into a checkpoint folder, in this example `receipes.model` that you can then use for inference.

```
# inference with pre-trained
#   python inference.py <path_to_checkpoint> <path_to_file>
python inference.py receipes.model ~/bibimbap.txt

# inference with 
#   python inference_raw.py <path_to_checkpoint> <path_to_file>
python inference_raw.py receipes.model ~/bibimbap.txt
```



## Notes
The dependency setup is a bit strange to accommodate hardware constraints, it looks like the highest version of torch I can use is version 2.2.2 (See https://discuss.pytorch.org/t/pip3-install-torch-locked-on-2-2-2/218086).

As a result there are a few other constraints to get things to work.
- transformers==4.39.3
- numpy==1.26.4

As well a few constraints on huggingface models such as safe-tensors.

If you have a silicon-mac, or other hardware, you can probably just install the latest versions of everything.

