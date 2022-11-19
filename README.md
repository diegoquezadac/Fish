# About
Research about Fish neural network. A task-conscious autoencoder. 

# VENV Set up

1. pip install -U pip setuptools wheel
2. pip install -U 'spacy[apple]'
3. python -m spacy download en_core_web_sm
4. pip3 install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu

# Spacy
1. python -m spacy init fill-config base_config.cfg config.cfg
2. python -m spacy debug data config.cfg
3. python -m spacy train config.cfg --output ./output --paths.train ./train.spacy --paths.dev ./dev.spacy

# Pytorch
1. pip3 install --pre torch torchvision torchaudio torchtext --extra-index-url https://download.pytorch.org/whl/nightly/cpu