import numpy as np
import torch
from flair.data import Sentence
from flair.embeddings import TokenEmbeddings
from transformers import AutoModel, AutoTokenizer


class LLaMAEmbeddings(TokenEmbeddings):
  def __init__(self, model_name="meta-llama/Meta-Llama-3.1-8B"):
    super().__init__()
    # Load the LLaMA model and tokenizer
    self.model = AutoModel.from_pretrained(model_name)
    self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    self.name = model_name

  def embed(self, sentences):
    # Make sure sentences is a list
    if not isinstance(sentences, list):
      sentences = [sentences]

    for sentence in sentences:
      # Tokenize the entire sentence with LLaMA tokenizer
      tokenized_input = self.tokenizer(sentence.to_tokenized_string(), return_tensors='pt', add_special_tokens=False)
      
      # Get token embeddings from LLaMA model for the entire sentence
      with torch.no_grad():
          outputs = self.model(**tokenized_input)
      
      # Get the hidden states (embeddings) of the last layer
      token_embeddings = outputs.last_hidden_state.squeeze(0)
      
      # Align Flair tokens with LLaMA tokens and assign embeddings
      token_idx = 0
      
      for flair_token in sentence.tokens:
        subwords = self.tokenizer.tokenize(flair_token.text)
        subword_embeddings = token_embeddings[token_idx:token_idx + len(subwords)]
        
        # Aggregate subword embeddings (mean pooling here)
        aggregated_embedding = torch.mean(subword_embeddings, dim=0)
        
        # Set aggregated embedding to the Flair token
        flair_token.set_embedding(self.name, aggregated_embedding)
        
        # Move the token index forward
        token_idx += len(subwords)

  @property
  def embedding_length(self):
    # LLaMA models typically have a hidden size of 4096 for the large models
    return self.model.config.hidden_size


if __name__ == '__main__':
  # Example: Testing the custom LLaMA embedding class
  # llama_embedding = LLaMAEmbeddings(model_name="meta-llama/LLaMA-2-7b-hf")
  llama_embedding = LLaMAEmbeddings(model_name="meta-llama/Meta-Llama-3.1-8B")

  # Create a Flair Sentence
  sentence = Sentence("LLaMA is a powerful and efficient language model.")

  # Embed the sentence using the LLaMA embeddings
  llama_embedding.embed(sentence)

  # Access the embeddings for each token
  for token in sentence:
    print(f"Token: {token.text}")
    print(f"Embedding shape: {token.embedding.shape}")




  # Save embeddings
  embeddings = np.array([token.embedding.cpu().numpy() for token in sentence])
  np.save("llama_embeddings.npy", embeddings)

  # Load embeddings
  loaded_embeddings = np.load("llama_embeddings.npy")