# Transformer classifier
A transformer based classifier that conveniently uses your file system as training datasets, all you need is to place data into appropriatedly named folders.




## Notes
The dependency setup is a bit strange to accommodate hardware constraints, it looks like the highest version of torch I can use is version 2.2.2 (See https://discuss.pytorch.org/t/pip3-install-torch-locked-on-2-2-2/218086).

As a result there are a few other constraints to get things to work.
- transformers==4.39.3
- numpy==1.26.4

As well a few constraints on huggingface models such as safe-tensors.

If you have a silicon-mac, or other hardware, you can probably just install the latest versions of everything.

