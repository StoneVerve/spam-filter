from utils.message_type import MessageType
from utils.message import Message
from math import log
from typing import List, Dict
from filters.nbayes_filter import NBayesFilter


class BernoulliFilter(NBayesFilter):

    def train_model(self, train_messages: List[Message]):
        ham_messages = [m for m in train_messages if not m.is_spam()]
        spam_messages = [m for m in train_messages if m.is_spam()]
        num_messages = len(train_messages)
        num_spam_messages = len(spam_messages)
        num_ham_messages = len(ham_messages)
        self.prob_spam = log(num_spam_messages) - log(num_messages)
        self.prob_ham = log(num_ham_messages) - log(num_messages)
        self.prob_words_spam = self.calculate_words_probabilities(spam_messages)
        self.prob_words_ham = self.calculate_words_probabilities(ham_messages)
        self.default_prob_word_spam = 0.0 - log(num_spam_messages + 2)
        self.default_prob_word_ham = 0.0 - log(num_ham_messages + 2)

    def get_word_probability(self, messages: List[Message], word: str) -> float:
        occurrences_in_messages = 0
        for message in messages:
            if message.is_word_in_message(word):
                occurrences_in_messages += 1
        return log(occurrences_in_messages + 1) - log(len(messages) + 2)
