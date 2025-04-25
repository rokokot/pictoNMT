# PictoNMT: Neural Machine Translation for Pictogram-to-Text Generation

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-310/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![Vision Transformer](https://img.shields.io/badge/Model-ViT-green)](https://arxiv.org/abs/2010.11929)
[![Eole 0.2.0](https://img.shields.io/badge/Eole-0.2.0-red)](https://github.com/OpenNMT/Eole)
[![ARASAAC](https://img.shields.io/badge/Pictograms-ARASAAC-purple)](https://arasaac.org/)
[![License MIT](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

PictoNMT is a neural machine translation system designed for converting pictogram sequences (currently only supports ARASAAC) into natural language french text. This research project adapts existing NMT architectures built on transformer-based encoders, with custom embedding and decoding strategies.

Augmentative and Alternative Communication (AAC) systems use pictograms to help users express themselves with natural languge proxies. Converting pictogram sequences into natural language text enables easier communication and information exchange for users with intellectual disabilities or language use deficits. Rather than treating this as a standard multimodal task, we model it as a translation problem where pictograms serve as the source language and natural french text as the target.

Our system is released with the following contributions:

  1. Hybrid Pictogram Encoding

Built on top of the Eole (former OpenNMT) framework, our system implements a custom encoder that processes both visual and semantic information from pictograms. We use a lightweight vision transformer to generate embeddings of picto sequences, and a traditional text ecoder to process category information, word types, and actual word labels.

  2. Custom Decoding Modules

The system supports, and in large part depends, on the strategies used to convert final model representations into natural text sentences. The base system includes: schema induction module, functional word prediction module, beam search module.

### Eole integration

Eole is a machine translation framework which contains bluebrints with boileplate code that makes development of custom MT system easier and more customizable.

```python
# Example of how we register our custom encoder with Eole
from eole.registry import register_encoder
from pictollms.models.encoders.eole_encoder import PictoEoleEncoder

@register_encoder("picto_encoder")
class RegisteredPictoEncoder(PictoEoleEncoder):
    pass
```

### Dataset

We train and evaluate our model on the Propicto Orféo corpus of aligned French text with pictogram sequences, enhanced with automatically generated annotations of metadata entries from the ARASAAC API. We collected and processed the data by parsing the corpus in json format availale [here](link), extracting picto IDs and french texts (gold sentence, and picto transcription), and formatting data for Eole (see scripts/data_processing/eole_data.py).

We annotate the datasets with comprehensive pictogram metadata, by extracting categories, types, and French keywords and aligning them with the propicto corpus.

#### Installation

```bash
git clone https://github.com/yourusername/pictoNMT.git    # Clone repository
cd pictoNMT

conda create -n pictoNMT python=3.10    # Create environment
conda activate pictoNMT

pip install torch torchvision numpy matplotlib pillow tqdm requests pytest

git clone https://github.com/OpenNMT/Eole.git     # Install Eole (required for full system
cd Eole
pip install -e .
cd ..

pip install -e .
```
