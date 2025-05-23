import re
import torch
from typing import List
from unidecode import unidecode

PAD_IDX, BOS_IDX, EOS_IDX, UNK_IDX = 2, 0, 1, 3

class Tokenizer: # Generic tokenizer class
  def __init__(self, model_name: str, **kwargs) -> None:
      super().__init__()
      self.model_name = model_name
      self.bos_token = "[BOS]"
      self.eos_token = "[EOS]"
      self.pad_token = "[PAD]"
      self.unk_token = "[UNK]"
      self.bos_id = 0
      self.eos_id = 1
      self.pad_id = 2
      self.unk_id = 3

  def tokenize(self, text: str) -> List[int]:
      return self.encode(text)

  def detokenize(self, token_ids: List[int]) -> str:
      return self.decode(token_ids)

  def save(self, output_dir: str) -> None:
      self.model.save(output_dir, self.model_name)

  def prepare_text(self, text: str) -> List[int]:
      # Add BOS and EOS tokens to the encoded text and return tensor
      return torch.cat([
        torch.tensor([BOS_IDX]), 
        torch.tensor(self.encode(text)), 
        torch.tensor([EOS_IDX])
      ])

class CharTokenizer(Tokenizer):
    def __init__(self, model_name: str, vocab_file: str) -> None:
        super().__init__(model_name)
        self.load(vocab_file)

    def encode(self, text: str) -> List[int]:
        text = self.pre_tokenize(text)
        return [self.vocab[char] for char in text if char in self.vocab]

    def decode(self, token_ids: List[int]) -> str:
        # Remove padding, BOS, and EOS tokens
        # token_ids = [token_id for token_id in token_ids if token_id not in [PAD_IDX, BOS_IDX, EOS_IDX]]
        # Add tokens until EOS is found
        _token_ids = []
        for token_id in token_ids:
            if token_id in [EOS_IDX]:
                break
            # Filter blank token CTC
            if token_id == self.vocab_size:
                continue
            if token_id not in [PAD_IDX, BOS_IDX]:
              _token_ids.append(token_id)
        return "".join([self.ids_to_tokens[i] for i in _token_ids])

    def pre_tokenize(self, text: str) -> str:
        text = unidecode(text)
        if 'EUR' in text:
          text = text.replace('EUR', '€')
        
        if 'PS' in text:
          text = text.replace('PS', '£')

        return unidecode(text)

    
    def load(self, vocab_file: str) -> None:
      # Load file and create a dictionary
      with open(vocab_file, "r") as f:
          vocab = f.readlines()
      
      # Remove \n from each line
      vocab = [re.sub(r"\n", "", line) for line in vocab]
      
      self.vocab = dict({
          "[BOS]": 0,
          "[EOS]": 1,
          "[PAD]": 2,
          "[UNK]": 3
      })

      for i, char in enumerate(vocab):
        self.vocab[char] = i + 4

      self.ids_to_tokens = {v: k for k, v in self.vocab.items()}
      self._vocab_size = len(self.vocab)
      print(f'VOCAB SIZE TOKENIZER: {self.vocab_size}')
      print(f'COMPLETE VOCAB: {self.vocab}')

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)