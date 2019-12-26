import torch
import torch.nn as nn
from typing import Tuple

from implementations.dataset import Vocabulary
from implementations.helpers import logsumexp


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Encoder(nn.Module):
    """
    A simple encoder model to encode sentences. Bi-LSTM over word embeddings.
    """

    def __init__(self, vocabulary_size: int, embedding_dim: int, hidden_dimension: int, padding_idx: int):
        super(Encoder, self).__init__()
        # embeddings
        self.embedding = nn.Embedding(vocabulary_size, embedding_dim=embedding_dim,
                                      padding_idx=padding_idx)

        # bi-LSTM
        input_size = embedding_dim
        self.bi_lstm = nn.LSTM(input_size, hidden_dimension, num_layers=1, bias=True,
                               bidirectional=True)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        """
        :param sentence: input sequence of size [batch_size, sequence_length]
        returns: tensor of size [batch_size, sequence_length, hidden_size * num_directions]
        representing the hidden states for the forward and backward LSTM of each time step of the sequences in the batch.
        """
        embedded = self.embedding(sentence)  # [batch_size, sequence_length, embedding_dimension]
        # [batch_size, sequence_length, hidden_size* num_directions],
        # ([batch_size, hidden_size * num_directions], [batch_size, hidden_size * num_directions])
        output, (hidden, cell) = self.bi_lstm(embedded)
        return output  # [batch_size, sequence_length, hidden_size * num_directions]


class ChainCRF(nn.Module):
    """
    A linear-chain conditional random field.
    """

    def __init__(self, num_tags: int, tag_vocabulary: Vocabulary):
        super(ChainCRF, self).__init__()

        self.tag_vocabulary = tag_vocabulary
        self.num_tags = num_tags
        self.root_idx = tag_vocabulary.sos_idx
        self.end_idx = tag_vocabulary.eos_idx

        # Matrix of transition parameters.  Entry (i, j) is the score of
        # transitioning *to* i *from* j.
        self.log_transitions = nn.Parameter(torch.randn(self.num_tags, self.num_tags), requires_grad=True)

        # Initialize the log transitions with xavier uniform (TODO: refer)
        self.xavier_uniform()

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.log_transitions.data[:, self.root_idx] = -10000.

        self.log_transitions.data[self.end_idx, :] = -10000.
        print(self.log_transitions.size())

    def xavier_uniform(self, gain=1.) -> None:
        torch.nn.init.xavier_uniform_(self.log_transitions)

    def forward_belief_propagation(self, input_features: torch.Tensor, input_mask: torch.Tensor) -> torch.Tensor:
        """
        ... TODO
        :param input_features: tensor of size [batch_size, max_time, num_tags] defining the features for each input
        sequence.
        :param input_mask: the binary mask determining which of the input entries are padding [batch_size, max_time]
        """
        batch_size, max_time, feature_dimension = input_features.size()

        # Initialize the recursion variables with transitions from root token + first emission probabilities.
        init_alphas = self.log_transitions[self.root_idx, :] + input_features[:, 0, :]

        # Set recursion variable.
        forward_var = init_alphas

        # Make time major.
        input_features = torch.transpose(input_features, 0, 1)  # [time, batch_size, num_tags]
        input_mask = torch.transpose(input_mask.float(), 0, 1)  # [time, batch_size]

        # Loop over sequence and calculate the transition probability for the next tag at each step (from t - 1 to t)
        # current tag at t - 1, next tag at t
        # emission probabilities: (example, next tag)
        # transition probabilities: (current tag, next tag)
        # forward var: (instance, current tag)
        # next tag var: (instance, current tag, next tag)
        for time in range(1, max_time):
            # Get emission scores for this time step.
            features = input_features[time]

            # Broadcast emission probabilities.
            emit_scores = features.view(batch_size, self.num_tags).unsqueeze(1)

            # Calculate transition probabilities (broadcast over example axis, same for all examples in batch).
            transition_scores = self.log_transitions.unsqueeze(0)

            # Calculate next tag probabilities.
            next_tag_var = forward_var.unsqueeze(2) + emit_scores + transition_scores

            # Calculate next forward var by taking logsumexp over next tag axis, mask all instances that ended
            # and keep old forward var for instances those.
            forward_var = (logsumexp(next_tag_var, 1) * input_mask[time].view(batch_size, 1) +
                           forward_var * (1 - input_mask[time]).view(batch_size, 1))

        final_transitions = self.log_transitions[:, self.end_idx]

        alphas = forward_var + final_transitions.unsqueeze(0)
        partition_function = logsumexp(alphas)

        return partition_function

    def score_sentence(self, input_features: torch.Tensor,
                       target_tags: torch.Tensor,
                       input_mask: torch.Tensor) -> torch.Tensor:
        batch_size, max_time, feature_dimension = input_features.size()

        # Make time major.
        input_features = input_features.transpose(0, 1)  # (time, batch_size, dim)
        input_mask = input_mask.float().transpose(0, 1)  # (time, batch_size)
        target_tags = target_tags.transpose(0, 1)  # (time, batch_size)

        # Get tensor of root tokens and tensor of next tags (first tags).
        root_tags = torch.LongTensor([self.root_idx] * batch_size, device=device)
        next_tags = target_tags[0].squeeze()

        # Initial transition is from root token to first tags.
        initial_transition = self.log_transitions[root_tags, next_tags]

        # Initialize scores.
        scores = initial_transition

        # Loop over time and at each time calculate the score from t to t + 1.
        for time in range(max_time - 1):
            # Get emission scores, transition scores and calculate score for current time step.
            features_t = input_features[time]
            next_tags = target_tags[time + 1].squeeze()
            current_tags = target_tags[time].squeeze()
            emission = torch.gather(features_t, 1, current_tags.unsqueeze(1)).squeeze()
            transition = self.log_transitions[current_tags, next_tags]

            # Add scores.
            scores = scores + transition * input_mask[time + 1] + emission * input_mask[time]

        # Add scores for transitioning to stop tag.
        last_tag_index = input_mask.sum(0).long() - 1
        last_tags = torch.gather(target_tags, 0, last_tag_index.view(1, batch_size)).view(-1)

        # end_tags
        end_tags = torch.LongTensor([self.end_idx] * batch_size)
        end_tags = end_tags.cuda() if torch.cuda.is_available() else end_tags
        last_transition = self.log_transitions[last_tags, end_tags]

        # Add the last input if its not masked.
        last_inputs = input_features[-1]
        last_input_score = last_inputs.gather(1, last_tags.view(-1, 1))
        last_input_score = last_input_score.squeeze()

        scores = scores + last_transition + last_input_score * input_mask[-1]

        # Sum over sequence length.
        return scores

    def viterbi_decode(self, input_features: torch.Tensor, input_lengths: torch.Tensor):
        batch_size, max_time, dim = input_features.size()

        # initialize the viterbi variables in log space
        init_vars = torch.full((1, self.num_tags), -10000., device=device)
        init_vars[0][self.root_idx] = 0

        # initialize tensor and list to keep track of backpointers
        backpointers = torch.zeros(batch_size, max_time, self.num_tags, device=device).long() - 1
        best_last_tags = []
        best_path_scores = []

        # forward_var at step t holds the viterbi variables for step t - 1, diff per example in batch
        forward_var = init_vars.unsqueeze(0).repeat(batch_size, 1, 1)

        # counter counting down from number of examples in batch to 0
        counter = batch_size

        # loop over sequence
        for t in range(max_time):

            # if time equals some lengths in the batch, these sequences are ending
            ending = (input_lengths == t).nonzero()
            n_ending = len(ending)

            # if there are sequences ending
            if n_ending > 0:

                # grab their viterbi variables
                forward_ending = forward_var[(counter - n_ending):counter]

                # the terminal var giving the best last tag is the viterbi variables + trans. prob. to end token
                terminal_var = forward_ending + self.log_transitions[:, self.end_idx].unsqueeze(0)
                path_scores, best_tag_idx = torch.max(terminal_var, 1)

                # first sequence to end is last sequence in batch (sorted on sequence length)
                for tag, score in zip(reversed(list(best_tag_idx)), reversed(list(path_scores))):
                    best_last_tags.append(tag)
                    best_path_scores.append(score)

                # update counter keeping track of how many sequences already ended
                counter -= n_ending

            # get emission probabilities at current time step
            feat = input_features[:, t, :].view(batch_size, self.n_tags)

            # calculate scores of next tag
            forward_var = forward_var.view(batch_size, self.n_tags, 1)
            trans_scores = self.log_transitions.unsqueeze(0)
            next_tag_vars = forward_var + trans_scores

            # get best next tags and viterbi vars
            viterbivars_t, idx = torch.max(next_tag_vars, 1)
            best_tag_ids = idx.view(batch_size, -1)

            # add emission scores and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (viterbivars_t + feat).view(batch_size, -1)

            # save best tags as backpointers
            backpointers[:, t, :] = best_tag_ids.long()

        # get final ending sequence(s) and calculate the best last tag(s)
        ending = (input_lengths == max_time).nonzero()
        ending = ending.cuda() if torch.cuda.is_available() else ending
        n_ending = len(ending)

        if n_ending > 0:

            forward_ending = forward_var[(counter - n_ending):counter]

            # transition to STOP_TAG
            terminal_var = forward_ending + self.log_transitions[:, self.end_idx].unsqueeze(0)
            path_scores, best_tag_idx = torch.max(terminal_var, 1)

            for tag, score in zip(reversed(list(best_tag_idx)), reversed(list(path_scores))):
                best_last_tags.append(tag)
                best_path_scores.append(score)

        # reverse the best last tags (and scores) to put them back in the original batch order
        best_last_tags = torch.LongTensor(list(reversed(best_last_tags)))
        best_last_tags = best_last_tags.cuda() if torch.cuda.is_available() else best_last_tags
        best_path_scores = torch.LongTensor(list(reversed(best_path_scores)))
        best_path_scores = best_path_scores.cuda() if torch.cuda.is_available() else best_path_scores

        # follow the back pointers to decode the best path
        best_paths = torch.zeros(batch_size, max_time + 1).long()
        best_paths = best_paths.cuda() if torch.cuda.is_available() else best_paths
        best_paths = best_paths.index_put_((torch.LongTensor([i for i in range(backpointers.size(0))]), input_lengths),
                                           best_last_tags)

        # counter keeping track of number of active sequences
        num_active = 0

        # loop from max time to 0
        for t in range(max_time - 1, -1, -1):

            # if time step equals lengths of some sequences, they are starting
            starting = (input_lengths - 1 == t).nonzero()
            n_starting = len(starting)

            # if there are sequences starting, grab their best last tags
            if n_starting > 0:
                if t == max_time - 1:
                    best_tag_id = best_paths[num_active:num_active + n_starting, t + 1]
                else:
                    last_tags = best_paths[num_active:num_active + n_starting, t + 1]
                    best_tag_id = torch.cat((best_tag_id, last_tags.unsqueeze(1)), dim=0)

                # update number of active sequences
                num_active += n_starting

            # get currently relevant backpointers based on sequences that are active
            active = backpointers[:num_active, t]

            # follow the backpointers to the best previous tag
            best_tag_id = best_tag_id.view(num_active, 1)
            best_tag_id = torch.gather(active, 1, best_tag_id)
            best_paths[:num_active, t] = best_tag_id.squeeze()

        # sanity check that first tag is the root token
        assert best_paths[:, 0].sum().item() == best_paths.size(0) * self.root_idx

        return best_path_scores, best_paths[:, 1:]

    def negative_log_likelihood(self, input_features: torch.Tensor, target_tags: torch.Tensor,
                                input_mask: torch.Tensor) -> torch.Tensor:
        """
        ... TODO
        :param input_features: tensor of size [batch_size, sequence_length, feature_dimension] defining the features for each input sequence.
        :param target_tags: the target tags of size [batch_size, sequence_length]
        :param input_mask: the binary mask determining which of the input entries are padding [batch_size, sequence_length]
        """
        forward_score = self.forward_belief_propagation(input_features=input_features, input_mask=input_mask)
        gold_score = self.score_sentence(input_features=input_features, target_tags=target_tags, input_mask=input_mask)
        return forward_score - gold_score

    def forward(self, input_features: torch.Tensor,
                target_tags: torch.Tensor,
                input_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor,
                                                   torch.Tensor]:
        """
        :param input_features: tensor of size [batch_size, sequence_length, feature_dimension] defining the features for each input sequence.
        :param target_tags: the target tags of size [batch_size, sequence_length]
        :param input_mask: the binary mask determining which of the input entries are padding [batch_size, sequence_length]
        """
        loss = self.negative_log_likelihood(input_features=input_features, target_tags=target_tags,
                                            input_mask=input_mask)
        # score, tag_sequence = self.viterbi_decode()
        score, tag_sequence = 0, [0, 0]
        return loss, score, tag_sequence


class Tagger(nn.Module):
    """
    A POS-tagger.
    """

    def __init__(self, input_vocabulary: Vocabulary, target_vocabulary: Vocabulary,
                 embedding_dimension: int, hidden_dimension: int):
        super(Tagger, self).__init__()
        self.encoder = Encoder(vocabulary_size=input_vocabulary.size, embedding_dim=embedding_dimension,
                               hidden_dimension=hidden_dimension, padding_idx=input_vocabulary.pad_idx)
        self.encoder_to_tags = nn.Linear(hidden_dimension * 2, target_vocabulary.size)
        self.tagger = ChainCRF(num_tags=target_vocabulary.size, tag_vocabulary=target_vocabulary)

    def get_accuracy(self):
        raise NotImplementedError()

    def get_loss(self):
        raise NotImplementedError()

    def forward(self, input_sequence: torch.Tensor, target_sequence: torch.Tensor, input_mask: torch.Tensor):
        lstm_features = self.encoder(input_sequence)
        crf_features = self.encoder_to_tags(lstm_features)
        loss, score, tag_sequence = self.tagger(input_features=crf_features,
                                                target_tags=target_sequence,
                                                input_mask=input_mask)
        return
