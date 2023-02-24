# codis

<p align="left">
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
</p>
Continual disentanglement

## Install

Install the requirements and the package (ideally in a virtual environment):

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Getting started

Here's how to use the dataset:

```python
from torch.utils.data import DataLoader
from codis.data import InfiniteDSprites

dataset = InfiniteDSprites()
dataloader = DataLoader(dataset, batch_size=4)

batch = next(iter(dataloader))
draw_batch_grid(batch, show=True)
```

[Result of the above code](examples/batch.png)


For other use cases and a more detailed introduction, see the notebooks in the `examples` folder.

## (Optional) Download DSprites

On macOS:

```bash
curl -LO https://github.com/deepmind/dsprites-dataset/raw/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz --output_dir codis/data
```
