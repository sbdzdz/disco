# 🪩 Disco: Disentangled Continual Learning

<p align="left">
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
</p>

[Infinite dSprites for Disentangled Continual Learning: Separating Memory Edits from Generalization](https://arxiv.org/abs/2312.16731)

Published at [CoLLAs 2024](https://lifelong-ml.cc/Conferences/2024/acceptedpapersandvideos/conf-2024-30).

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
from disco.data import InfiniteDSprites

dataset = InfiniteDSprites()
dataloader = DataLoader(dataset, batch_size=4)

batch = next(iter(dataloader))
draw_batch(batch, show=True)
```

For other use cases and a more detailed introduction, see the notebooks in the [examples](examples/) folder.

## Plots

To reproduce the paper plots, see [plots.sh](plots/plots.sh) script.

Rendering the figures requires TeX Live. To install it on macOS, use Homebrew:

```bash
brew install --cask mactex
```

Make sure the executables are in your PATH:

```bash
find / -name kpsewhich 2>/dev/null
```

Add the directory from the output to your PATH, e.g.:

```bash
export PATH=/usr/local/texlive/2023/bin/universal-darwin:$PAT
```

## Citation
If you use this work in your research, please consider citing:
```
@article{dziadzio2023disentangled,
  title={Disentangled Continual Learning: Separating Memory Edits from Model Updates},
  author={Dziadzio, Sebastian and Y{\i}ld{\i}z, {\c{C}}a{\u{g}}atay and van de Ven, Gido M and Trzci{\'n}ski, Tomasz and Tuytelaars, Tinne and Bethge, Matthias},
  journal={arXiv preprint arXiv:2312.16731},
  year={2023}
}
```
Thanks!
