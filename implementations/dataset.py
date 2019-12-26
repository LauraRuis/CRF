from collections import defaultdict
from collections import Counter
from typing import List
from typing import Tuple
import torch
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Vocabulary(object):
    """
    Object that maps words in string-form to indices that can be processed by numerical models.
    """
    def __init__(self, sos_token="<ROOT>", eos_token="<EOS>", pad_token="<PAD>"):
        super(self, Vocabulary).__init__()
        """
        NB: <PAD> token is by construction idx 0.
        """
        self.pad_token = pad_token
        self.sos_token = sos_token
        self.eos_token = eos_token
        self._idx_to_word = [pad_token, sos_token, eos_token]
        self._word_to_idx = defaultdict(lambda: self._idx_to_word.index(self.pad_token))
        self._word_to_idx[sos_token] = 1
        self._word_to_idx[eos_token] = 2
        self._word_frequencies = Counter()

    def word_to_idx(self, word: str) -> int:
        return self._word_to_idx[word]

    def idx_to_word(self, idx: int) -> str:
        return self._idx_to_word[idx]

    @property
    def sos_idx(self) -> int:
        return self.word_to_idx(self.sos_token)

    @property
    def eos_idx(self) -> int:
        return self.word_to_idx(self.eos_token)

    @property
    def size(self) -> int:
        return len(self._idx_to_word)

    @property
    def pad_idx(self) -> int:
        return self.word_to_idx(self.pad_token)

    def add_sentence(self, sentence: List[str]) -> None:
        for word in sentence:
            if word not in self._word_to_idx:
                self._word_to_idx[word] = self.size
                self._idx_to_word.append(word)
            self._word_frequencies[word] += 1

    def most_common(self, n=10) -> List[Tuple[str, int]]:
        return self._word_frequencies.most_common(n=n)


class TaggingDataset(object):
    """
    Holds a POS tagging dataset (i.e. sequence classification).
    """
    def __init__(self, input_vocabulary: Vocabulary, target_vocabulary: Vocabulary):
        self._input_vocabulary = input_vocabulary
        self._target_vocabulary = target_vocabulary

        self._examples = []
        self._example_lengths = []

    def read_dataset(self, input_data: List[Tuple[List, List]]) -> None:
        """Convert each example to a tensor and saves it's length."""
        for input_list, target_list in input_data:
            assert len(input_list) == len(target_list), "Invalid data example."
            input_array = self.sentence_to_array(input_list, vocabulary="input")
            target_array = self.sentence_to_array(target_list, vocabulary="target")
            input_tensor = torch.tensor(input_array, dtype=torch.long, device=device)
            target_tensor = torch.tensor(target_array, dtype=torch.long, device=device)
            self._example_lengths.append(input_tensor.size(0))
            self._examples.append({"input_tensor": input_tensor.unsqueeze(0),
                                   "target_tensor": target_tensor.unsqueeze(0)})

    def get_vocabulary(self, vocabulary: str) -> Vocabulary:
        if vocabulary == "input":
            vocab = self._input_vocabulary
        elif vocabulary == "target":
            vocab = self._target_vocabulary
        else:
            raise ValueError(
                "Specified unknown vocabulary in sentence_to_array: {}".format(
                    vocabulary))
        return vocab

    def sentence_to_array(self, sentence: List[str], vocabulary: str) -> List[int]:
        """
        Convert each string word in a sentence to the corresponding integer from the vocabulary.
        :param sentence: the sentence in words (strings).
        :param vocabulary: whether to use the input or target vocabulary.
        :return: the sentence in integers.
        """
        vocab = self.get_vocabulary(vocabulary)
        sentence_array = [vocab.sos_idx]
        for word in sentence:
            sentence_array.append(vocab.word_to_idx(word))
        sentence_array.append(vocab.eos_idx)
        return sentence_array

    def array_to_sentence(self, sentence_array: List[int], vocabulary: str) -> List[str]:
        """
        Translate each integer in a sentence array to the corresponding word.
        :param sentence_array: array with integers representing words from the vocabulary.
        :param vocabulary: whether to use the input or target vocabulary.
        :return: the sentence in words.
        """
        vocab = self.get_vocabulary(vocabulary)
        return [vocab.idx_to_word(word_idx) for word_idx in sentence_array.squeeze()]

    def get_batch(self, batch_size=2) -> Tuple[torch.Tensor, List[int],
                                               torch.Tensor]:
        for example_i in range(0, len(self._examples) - batch_size, batch_size):
            examples = self._examples[example_i:example_i + batch_size]
            example_lengths = self._example_lengths[example_i:example_i + batch_size]
            max_length = np.max(example_lengths)
            input_batch = []
            target_batch = []
            for example in examples:
                to_pad = max_length - example["input_tensor"].size(1)
                padded_input = torch.cat([
                    example["input_tensor"],
                    torch.zeros(int(to_pad), dtype=torch.long, device=device).unsqueeze(0)], dim=1)
                padded_target = torch.cat([
                    example["target_tensor"],
                    torch.zeros(int(to_pad), dtype=torch.long, device=device).unsqueeze(0)], dim=1)
                input_batch.append(padded_input)
                target_batch.append(padded_target)

            yield (torch.cat(input_batch, dim=0), example_lengths,
                   torch.cat(target_batch, dim=0))
