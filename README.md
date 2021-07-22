## Named Entity Recognition in Social Media

See project description [here](https://github.com/JRC1995/SocialMediaNER/blob/main/Named_Entity_Recognition_in_Social_Media.pdf)

#### Abstract:
We address the challenges posed by noise and emerging/rare entities in Named Entity Recognition task for social media domain. Following the recent advances, we employ Contextualized Word Embeddings from Language Models pretrained on large corpora along with some normalization techniques to reduce
noise. Our best model achieves state-of-the-art results (F1 52.47%) on WNUT 2017 dataset. Additionally, we adapt a modular approach to systematically evaluate different contextual embeddings and downstream labeling mechanism using Sequence Labeling and a Question Answering framework.

<b>Note:</b> This is a project report for the CS512 Advanced Machine Learning course, done back in 2020 Q2. It's no longer SOTA.

## Credits
It was a group project.
Team members:
* Jishnu Ray Chowdhury
* Usman Shahid
* Tuhin Kundu
* Zhimming Zou

Code credits:
* Files under `models/layers/BigTransformers/` are from: https://github.com/huggingface/transformers
* Files under `process_toools/ark_tweet` are adapted from: https://github.com/ianozsvald/ark-tweet-nlp-python
* `conlleval.py` is from here: https://github.com/sighsmile/conlleval
* `conlleval.perl` is from here: https://www.clips.uantwerpen.be/conll2000/chunking/conlleval.txt
* We use Twitter word embeddings from godin et al. (see more:  https://fredericgodin.com/research/twitter-word-embeddings/)
```
@phdthesis{godin2019,
     title    = {Improving and Interpreting Neural Networks for Word-Level Prediction Tasks in Natural Language Processing},
     school   = {Ghent University, Belgium},
     author   = {Godin, Fr\'{e}deric},
     year     = {2019},
 }
```

## Requirements
* tqdm
* panphon
* epitran # make sure lex_lookup is installed (https://github.com/dmort27/epitran#lex_lookup)
* transformers
* flairNLP
* fasttext
* gensim
* pytorch-crf

## Downloads

 * DOWNLOAD: tweet word2vec from [here](https://drive.google.com/file/d/1lw5Hr6Xw0G0bMT1ZllrtMqEgCTrM7dzc/view)
 * Keep the above download in `embeddings/word2vec/`
 * DOWNLOAD: fasttext (crawl-300d-2M-subword.zip) from [here](https://fasttext.cc/docs/en/english-vectors.html)
 * Keep the above download in `embeddings/fasttext/`
 * Run save_locally_BERT.py and save_locally_ELECTRA.py to save pretrained contextualized transformer models
 
## Preprocessing
 * run process_WNUT_phase1.py and process_WNUT_phase2.py (in that order) for preprocessing

## Training 
`python train.py --model=[INSERET MODEL NAME HERE]`

## Testing
`python train.py --model=[INSERET MODEL NAME HERE] --test=True`

## Evaluation
* Generate file for evaluation: `python generate_eval_file.py --model=[INSERET MODEL NAME HERE]` or `python generate_eval_file_MRC.py --model=[INSERET MODEL NAME HERE]` for MRC based models 
* Run conlleval.py or conlleval.perl on the generated evaluation files. 
