from flair.embeddings import FlairEmbeddings, StackedEmbeddings
from flair.data import Sentence
import numpy as np



def space_tokenizer(s):
    for tok in s.split(' '):
        yield tok


class CSEGenerator():
    def __init__(self, use_forward, use_backward):
        if not(use_forward or use_backward):
            raise ValueError("Must use either forward or backward embeddings for CSE")

        mod_list = []
        if use_forward:
            mod_list.append(FlairEmbeddings('news-forward'))
        if use_backward:
            mod_list.append(FlairEmbeddings('news-backward'))
        self.model = StackedEmbeddings(mod_list)
        self.cache = dict()
    

    def get_emb(self, tokens):
        sent_str = ' '.join(tokens)
        if sent_str in self.cache:
            return self.cache[sent_str]
        sent = Sentence(sent_str, use_tokenizer=space_tokenizer)
        self.model.embed(sent)
        embs = []
        for tok_idx, tok in enumerate(sent):
            tok_emb = tok.embedding.cpu().numpy()
            embs.append(tok_emb)
        res = np.array(embs)
        self.cache[sent_str] = res
        return res

