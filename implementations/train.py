import torch

from implementations.dataset import TaggingDataset
from implementations.models import ChainCRF


def train(data: TaggingDataset, model: ChainCRF, batch_size: int):
    for input_sequence, example_lengths, target_sequence in data.get_batch(batch_size=batch_size):
        print(input_sequence)
        print(target_sequence)
        loss, score, output_tags = model(input_sequence=input_sequence, target_sequence=target_sequence,
                                         input_mask=torch.ones_like(input_sequence))
        print(output_tags)
