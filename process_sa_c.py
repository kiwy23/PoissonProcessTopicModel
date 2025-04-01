# NOTE-Topic-Score: prepare word_count mat.
# NOTE: prepare contextualized embed.

import datetime
import os
import re
import numpy as np
import pandas as pd
from tqdm import tqdm

import nltk
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer
lancaster = LancasterStemmer()
from nltk.stem import SnowballStemmer
snowball = SnowballStemmer("english")

from flair.embeddings import TransformerWordEmbeddings
from flair.data import Sentence, Token

from scipy.io import mmread

if __name__ == "__main__":
    '''
    need to request the permission to use llama from meta
    recommend submitting the request from huggingface
    once accepted, follow the instructions there to download the pretrained llama model
    '''
    # initialize language model for contextual embedding
    model_name = "bert"
    if model_name == "bert":
        embedding_model = TransformerWordEmbeddings("bert-base-uncased")
    # please comment out the following two lines if you don't have the llama model ready
    else:
        raise NotImplementedError("language model not supported yet")

    def get_vec(word: str) -> np.ndarray:
        word_token = Token(word)
        embedding_model.embed(Sentence([word_token]))
        return word_token.embedding.numpy()

    # get the dimension of the embedding
    dim = get_vec("good").shape[0]

    # read from dataset
    paper = pd.read_csv('./dataset/SA/paper.csv')
    AuPapMat = pd.read_csv('./dataset/SA/AuPapMat.csv')
    PapJur = AuPapMat[['idxPap','journal']].drop_duplicates()
    paper['idxPap'] = paper.index + 1
    paper = pd.merge(paper, PapJur, how='left', on='idxPap')
    word_count_raw = np.array(mmread('./dataset/SA/statabs_word_count.mtx').todense())
    with open('./dataset/SA/statabs_words.txt', 'r') as file:
        lines = file.readlines()
        dic_raw = [line.strip() for line in lines]
    dic_raw = np.array(dic_raw)
    # dic_stem = np.array([snowball.stem(word) for word in dic_raw])
    # note: dic has already been stemmed.
    
    # # remove documents not in particular journals and year range
    # journal_filter = paper['journal'].isin(['AoS', 'Bka', 'JASA', 'JRSSB'])
    # year_filter = paper['year'].astype(int)>=2003
    # filter_all = journal_filter & year_filter
    # doc_indices = np.array(range(len(paper)))[filter_all]

    # # remove short documents, keeping track of the index
    # min_doc_length = 56  # NOTE: rm ~40% of papers
    # doc_indices = doc_indices[np.where(word_count_raw[:,doc_indices].sum(0) >= min_doc_length)[0]]
    # word_count = word_count_raw[:, doc_indices]
    
    min_doc_length = 38 # rm ~40% papers # in 1029113821 we use 50.
    doc_indices  = np.where(word_count_raw.sum(0) >= min_doc_length)[0]
    word_count = word_count_raw[:, doc_indices]

    # get stop words
    nltk.data.path.append("./nltk_data/")
    try:
        STOPWORDS = np.array([word.replace("'", "") for word in stopwords.words("english")])
    except LookupError:
        nltk.download("stopwords", download_dir="./nltk_data/")
        STOPWORDS = np.array([word.replace("'", "") for word in stopwords.words("english")])
        
    greek_letters = [
    'Alpha', 'Beta', 'Gamma', 'Delta', 'Epsilon', 'Zeta', 'Eta', 'Theta', 'Iota', 'Kappa', 
    'Lambda', 'Mu', 'Nu', 'Xi', 'Omicron', 'Pi', 'Rho', 'Sigma', 'Tau', 'Upsilon', 'Phi', 
    'Chi', 'Psi', 'Omega'
    ]
    greek_letters = [letter.lower() for letter in greek_letters]
    
    latex_symbols = [
        'rootn', 'cap', 'vertical', 'tilde', 'log', 'infinity', 'cup'
    ]
    addtional_word_to_remove = []
    addtional_word_to_remove.extend(greek_letters)
    addtional_word_to_remove.extend(latex_symbols)
    addtional_word_to_remove = [snowball.stem(word) for word in addtional_word_to_remove]
    
    word_to_remove = []
    word_to_remove.extend(list(STOPWORDS))
    word_to_remove.extend(addtional_word_to_remove)
    word_to_remove = np.array(word_to_remove)

    min_word_count = 10
    # remove stop words and short words
    word_indicator = np.array(
        [word not in word_to_remove and len(word) >= 3 for word in dic_raw]
        )
    # remove low frequency words
    word_indicator *= word_count.sum(1) >= min_word_count
    dic = dic_raw[word_indicator]
    _word_count = word_count[word_indicator]
    num_word, num_doc = _word_count.shape

    # normalize each document
    word_freq = _word_count / _word_count.sum(0)

    # save metadata
    dt = datetime.datetime.now().strftime('%m%d%H%M%S')
    os.makedirs(f"./npz/{dt}", exist_ok=True)
    np.savez(
        f"npz/{dt}/SA_{model_name}_c_{min_word_count}_metadata.npz",
        dic=dic,
        # word_freq=word_freq, # word_freq is large. 
        word_count=_word_count,
        min_word_count=min_word_count,
        doc_indices=doc_indices,
        num_doc=num_doc,
        num_word=num_word,
        dim=dim,
        )

    # _word_count.shape, word_count_raw.shape, sum(word_indicator), len(doc_indices)
    # import sys
    # sys.getsizeof(word_freq)
    
    # read abstracts
    paragraphs = np.array(["\n "+str(abstract)+"\n " for abstract in list(paper['abstract'])])
    
    word_repr = dict()
    for doc_id in tqdm(doc_indices):
        doc_id = int(doc_id)
        tmp = Sentence(paragraphs[doc_id])
        embedding_model.embed(tmp)
        word_repr[f"doc_{doc_id}_vec"] = np.array(
        [token.embedding.numpy() for token in tmp if snowball.stem(token.text.lower()) in dic]
        )
        word_repr[f"doc_{doc_id}_txt"] = np.array(
        [token.text for token in tmp if snowball.stem(token.text.lower()) in dic]
        )
        if len(word_repr[f"doc_{doc_id}_txt"]) == 0:
            print('warning: short length docs')
        del tmp

    np.savez(
        f'npz/{dt}/SA_{model_name}_c_{min_word_count}_repr.npz',
        **word_repr,
        )
    
    word_repr_combined = np.vstack([word_repr[f'doc_{doc_id}_vec'] for doc_id in doc_indices])

    np.save(f'npz/{dt}/SA_{model_name}_c_{min_word_count}_combined_repr.npy', word_repr_combined)

    
    
    