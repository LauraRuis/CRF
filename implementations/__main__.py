import argparse

from implementations.train import train
from implementations.models import Tagger
from implementations.dataset import TaggingDataset
from implementations.dataset import Vocabulary

parser = argparse.ArgumentParser(description="CRF implementations.")


parser.add_argument("--mode", type=str, default="train")
parser.add_argument("--embedding_dimension", type=int, default=5)
parser.add_argument("--hidden_dimension", type=int, default=4)
parser.add_argument("--training_batch_size", type=int, default=2)


def main(flags):

    if flags["mode"] == "train":

        # Make up some training data
        training_data = [(
            "What if Google morphed into googleOS ?".split(),
            "PRON IN PROPN VERB ADP-IN PROPN PUNCT".split()
        ), (
            "Google is a nice search engine".split(),
            "PROPN AUX-VBZ DET0DT ADJ NOUN NOUN".split()
        ),
            (
                "What if Google morphed into googleOS ?".split(),
                "PRON IN PROPN VERB ADP-IN PROPN PUNCT".split()
            ), (
                "Google is a nice search engine .".split(),
                "PROPN AUX-VBZ DET0DT ADJ NOUN NOUN PUNCT".split()
            )]

        input_vocabulary = Vocabulary()
        target_vocabulary = Vocabulary()
        for sentence, tags in training_data:
            input_vocabulary.add_sentence(sentence)
            target_vocabulary.add_sentence(tags)
        print("Input vocabulary size training set: {}".format(input_vocabulary.size))
        print("  Most common input words: {}".format(input_vocabulary.most_common(5)))
        print("Target vocabulary size training set: {}".format(target_vocabulary.size))
        print("  Most common target words: {}".format(target_vocabulary.most_common(5)))

        model = Tagger(input_vocabulary=input_vocabulary, target_vocabulary=target_vocabulary,
                       embedding_dimension=flags["embedding_dimension"],
                       hidden_dimension=flags["hidden_dimension"])

        pos_tagging_dataset = TaggingDataset(input_vocabulary=input_vocabulary, target_vocabulary=target_vocabulary)
        pos_tagging_dataset.read_dataset(training_data)

        train(data=pos_tagging_dataset, model=model, batch_size=flags["training_batch_size"])
    else:
        raise ValueError("Unrecognized value for flags mode: '{}'.".format(flags["mode"]))


if __name__ == "__main__":
    input_flags = vars(parser.parse_args())
    main(flags=input_flags)
