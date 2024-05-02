# Create classes for 3 types of tokenizers: char-level, BPE, and SentencePiece. We'll use Hugging Face's tokenizers library to create these tokenizers.
#
import os
import re
import torch
from typing import List, Tuple
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors
from transformers.tokenization_utils import AddedToken, PreTrainedTokenizer
from unidecode import unidecode

import tokenizers

PAD_IDX, SOS_IDX, EOS_IDX, UNK_IDX = 2, 0, 1, 3

class Tokenizer: # Generic tokenizer class
  def __init__(self, model_name: str, **kwargs) -> None:
      super().__init__()
      self.model_name = model_name
      self.bos_token = "[BOS]"
      self.eos_token = "[EOS]"
      self.pad_token = "[PAD]"
      self.unk_token = "[UNK]"
      # self.mask_token = "[MASK]"
      # self.sep_token = "[SEP]"
      self.bos_id = 0
      self.eos_id = 1
      self.pad_id = 2
      self.unk_id = 3
      # self.mask_id = 4
      # self.sep_id = 5
      # self.vocab_size = None

  def tokenize(self, text: str) -> List[int]:
      return self.encode(text)

  def detokenize(self, token_ids: List[int]) -> str:
      return self.decode(token_ids)

  def save(self, output_dir: str) -> None:
      self.model.save(output_dir, self.model_name)

  # def prepare_text(self, text: str) -> List[int]:
  #     return self.prepare_text(text)

  def prepare_text(self, text: str) -> List[int]:
      # Add BOS and EOS tokens to the encoded text and return tensor
      return torch.cat([
        torch.tensor([SOS_IDX]), 
        torch.tensor(self.encode(text)), 
        torch.tensor([EOS_IDX])
      ])

  # def load(self, model_name: str) -> None:
  #     self.model = models.SentencePiece.from_file(f"{model_name}.model")


class CharTokenizer(Tokenizer):
    def __init__(self, model_name: str, vocab_file: str) -> None:
        super().__init__(model_name)
        self.load(vocab_file)

    def encode(self, text: str) -> List[int]:
        text = self.pre_tokenize(text)
        return [self.vocab[char] for char in text]

    def decode(self, token_ids: List[int]) -> str:
        # Remove padding, BOS, and EOS tokens
        token_ids = [token_id for token_id in token_ids if token_id not in [PAD_IDX, SOS_IDX, EOS_IDX]]
        return "".join([self.ids_to_tokens[i] for i in token_ids])

    def pre_tokenize(self, text: str) -> str:
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

      # ADD ['et'] + ['ç'] TO VOCAB
      self.vocab['et'] = len(self.vocab)
      self.vocab['ç'] = len(self.vocab)


      self.ids_to_tokens = {v: k for k, v in self.vocab.items()}
      self._vocab_size = len(self.vocab)
      print(f'VOCAB SIZE TOKENIZER: {self.vocab_size}')

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)


class BPETokenizer(Tokenizer):
    def __init__(self, model_name: str, path_checkpoint: str) -> None:
        super().__init__(model_name)
        self.model_name = model_name
        self.load(path_checkpoint)
    
    def load(self, path_checkpoint: str) -> None:
        self.tokenizer = tokenizers.Tokenizer.from_file(f"{path_checkpoint}")
        self.tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
        self.tokenizer.decoder = decoders.ByteLevel()
        self._vocab_size = self.tokenizer.get_vocab_size(with_added_tokens=True)

    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text).ids

    def decode(self, token_ids: List[int]) -> str:
        return self.tokenizer.decode(token_ids)

    @property
    def vocab_size(self) -> int:
        return self._vocab_size
    
    
# class SentencePieceTokenizer(Tokenizer):
#     def __init__(self, vocab_size: int) -> None:
#         super().__init__("sp_tokenizer", vocab_size)
        

    # def tokenize(self, text: str) -> List[int]:
    #     return self.tokenizer.encode(text).ids

    # def detokenize(self, token_ids: List[int]) -> str:
    #     return self.tokenizer.decode(token_ids)

    # def save(self, output_dir: str) -> None:
    #     self.tokenizer.model.save(output_dir, self.model_name)

    # def load(self, model_name: str) -> None:
    #     self.tokenizer.model = models.SentencePiece.from_file(f"{model_name}.model")