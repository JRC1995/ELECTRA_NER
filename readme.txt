make sure jdk is installed
install all these:
tqdm
panphon
epitran # make sure lex_lookup is installed (https://github.com/dmort27/epitran#lex_lookup)
transformers
flairNLP
fasttext
gensim
pytorch-crf
@phdthesis{godin2019,
     title    = {Improving and Interpreting Neural Networks for Word-Level Prediction Tasks in Natural Language Processing},
     school   = {Ghent University, Belgium},
     author   = {Godin, Fr\'{e}deric},
     year     = {2019},
 }
 https://fredericgodin.com/research/twitter-word-embeddings/
 
 DOWNLOAD: tweet word2vec from here: https://drive.google.com/file/d/1lw5Hr6Xw0G0bMT1ZllrtMqEgCTrM7dzc/view
 
 keep it in embeddings/word2vec/.....bin
 
 DOWNLOAD: fasttext (crawl-300d-2M-subword.zip) from here: https://fasttext.cc/docs/en/english-vectors.html
 
 keep it in embeddings/fasttext/...bin

 files under models/layers/BigTransformers/ are from: https://github.com/huggingface/transformers
 ark_tweet from:
 ark_tokenize....or something modified from: 
 
 
 run process_WNUT_phase1.py and process_WNUT_phase2.py (in that order) for preprocessing
 
 run save_locally_BERT.py and save_locally_ELECTRA.py to save pretrained contextualized transformer models
