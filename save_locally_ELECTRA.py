from transformers import *
import torch
from pathlib import Path

embedding_path = "embeddings/ELECTRA/"

Path(embedding_path).mkdir(parents=True, exist_ok=True)

# Transformers has a unified API
# for 8 transformer architectures and 30 pretrained weights.
#          Model          | Tokenizer          | Pretrained weights shortcut
MODELS = [(ElectraModel,      ElectraTokenizerFast,        'google/electra-large-discriminator')]

# To use TensorFlow 2.0 versions of the models, simply prefix the class names with 'TF', e.g. `TFRobertaModel` is the TF 2.0 counterpart of the PyTorch model `RobertaModel`

# Let's encode some text in a sequence of hidden-states using each model:
for model_class, tokenizer_class, pretrained_weights in MODELS:
    # Load pretrained model/tokenizer
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    model = model_class.from_pretrained(pretrained_weights,
                                        output_hidden_states=True,
                                        output_attentions=False)

    for name, param in model.named_parameters():
        print(name)
        print(param.size())

    model.save_pretrained(embedding_path)  # save
    tokenizer.save_pretrained(embedding_path)  # save
