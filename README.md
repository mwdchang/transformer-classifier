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
export PYTHONPATH=src 

# pre-trained
python -m transform_classifier.pretrained.train <data_folder>

# from scratch
python -m transform_classifier.raw.train <data_folder>

```

Once training is complete, it will dump the model config/parameters into a checkpoint folder, in this example `receipes.model` that you can then use for inference.

```
export PYTHONPATH=src 

# inference with pre-trained
python -m transform_classifier.pretrained.inference.py <data_folder.model> ~/bibimbap.txt

# inference with 
python -m transform_classifier.raw.inference.py <data_folder.model> ~/bibimbap.txt
```

## Package install
To install as a python package, run

```
pip install .
```

Then access the `transform_classifier` package. For example:

```python
from transform_classifier.raw.inference import run_inference

text = """
## Ingredients
- 2 slices white bread  
- 1 tsp Dijon mustard (optional)  
- 2 slices ham  
- ½ cup grated Gruyère or Emmental  
- Butter

## Instructions
1. Butter one side of each bread slice.  
2. On the non-buttered side, spread mustard, layer ham and cheese.  
3. Close the sandwich, buttered sides out.  
4. Toast in a skillet over medium heat until golden on both sides and cheese is melted (about 3–4 min per side).  
"""

result = run_inference("receipes.model", text)
print(f"Inferenced cuisine={result}")
```



## Notes
The dependency setup is a bit strange to accommodate hardware constraints on an intel-mac.
- See https://discuss.pytorch.org/t/pip3-install-torch-locked-on-2-2-2/218086)

As a result `requirements.txt` uses older versions to get packages to work nicely with each other, likewise there were a few limiting factors in using huggingface models like safe-tensors.

If you have different hardware, you may want (or need) to install the latest packages.
