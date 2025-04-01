import datetime
import os
import re
import numpy as np
import pandas as pd
from tqdm import tqdm

import nltk
from nltk.corpus import stopwords

from flair.embeddings import TransformerWordEmbeddings
from flair.data import Sentence, Token



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

  # read data from the CSV file
  # dic_raw = pd.read_csv("dataset/AP/dic.csv").to_numpy()[:, 1]
  read_raw = pd.read_csv("dataset/AP/word_count.csv")   # note: scipy npz sparse mat.
  dic_raw = read_raw.columns.to_numpy()[1:]
  word_count_raw = read_raw.to_numpy()[:, 1:].T

  # remove short documents, keeping track of the index
  min_doc_length = 50
  doc_indices = np.where(word_count_raw.sum(0) >= min_doc_length)[0]
  word_count = word_count_raw[:, doc_indices]

  # get stop words
  nltk.data.path.append("./nltk_data/")
  try:
    STOPWORDS = np.array([word.replace("'", "") for word in stopwords.words("english")])
  except LookupError:
    nltk.download("stopwords", download_dir="./nltk_data/")
    STOPWORDS = np.array([word.replace("'", "") for word in stopwords.words("english")])

  min_word_count = 10
  # remove stop words and short words
  word_indicator = np.array(
    [word not in STOPWORDS and len(word) >= 3 for word in dic_raw]
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
    f"npz/{dt}/AP_{model_name}_c_{min_word_count}_metadata.npz",
    dic=dic,
    word_freq=word_freq,
    word_count=_word_count,
    min_word_count=min_word_count,
    doc_indices=doc_indices,
    num_doc=num_doc,
    num_word=num_word,
    dim=dim,
  )

  # read the corpus
  with open("dataset/AP/raw_text.txt", "r") as file:
    content = file.read()

  # use regular expression to find all text between <TEXT> and </TEXT>
  paragraphs = np.array(re.findall(r"<TEXT>(.*?)</TEXT>", content, re.DOTALL))

  word_repr = dict()
  for doc_id in tqdm(doc_indices):
    doc_id = int(doc_id)
    tmp = Sentence(paragraphs[doc_id])
    embedding_model.embed(tmp)
    word_repr[f"doc_{doc_id}_vec"] = np.array(
    [token.embedding.numpy() for token in tmp if token.text.lower() in dic]
    )
    word_repr[f"doc_{doc_id}_txt"] = np.array(
    [token.text for token in tmp if token.text.lower() in dic]
    )
    if len(word_repr[f"doc_{doc_id}_txt"]) == 0:
        print('warning: short length docs')
    del tmp

  np.savez(
    f'npz/{dt}/AP_{model_name}_c_{min_word_count}_repr.npz',
    **word_repr,
    )
  
  word_repr_combined = np.vstack([word_repr[f'doc_{doc_id}_vec'] for doc_id in doc_indices])

  np.save(f'npz/{dt}/AP_{model_name}_c_{min_word_count}_combined_repr.npy', word_repr_combined)

  # # split doc_indices into batches
  # num_doc_per_file = 1200
  # num_batch = (doc_indices.max() + num_doc_per_file) // num_doc_per_file
  # doc_indicator = np.zeros(num_batch * num_doc_per_file)
  # doc_indicator[doc_indices] = 1
  # doc_indicator = doc_indicator.reshape(num_batch, num_doc_per_file)
  # all_batches = [
  #   np.argwhere(doc_indicator[k]).reshape(-1) + k * num_doc_per_file
  #   for k in range(num_batch)
  # ]
  
  # for batch_k, batch_index in tqdm(enumerate(all_batches)):
  #   # get word embeddings for batch k
  #   word_repr = dict()
  #   for doc_id in batch_index:
  #     doc_id = int(doc_id)
  #     tmp = Sentence(paragraphs[doc_id])
  #     embedding_model.embed(tmp)
  #     word_repr[f"doc_{doc_id}_vec"] = np.array(
  #       [token.embedding.numpy() for token in tmp if token.text.lower() in dic]
  #     )
  #     word_repr[f"doc_{doc_id}_txt"] = np.array(
  #       [token.text for token in tmp if token.text.lower() in dic]
  #     )
  #     del tmp

  #   # save word embeddings for batch k
  #   np.savez(
  #     f'npz/AP_{model_name}_c_{min_word_count}_batch_{batch_k}_{dt}.npz',
  #     **word_repr,
  #     )
