from utils.message import Message
from math import log
from typing import List
from filters.nbayes_filter import NBayesFilter


class MultinomialFilter(NBayesFilter):

    def __init__(self, alpha: int):
        NBayesFilter.__init__(self)
        self.alpha = alpha

    def train_model(self, train_messages: List[Message]):
        ham_messages = []
        spam_messages = []
        total_words_spam = 0
        total_words_ham = 0
        for m in train_messages:
            if m.is_spam():
                total_words_spam += m.number_of_words() + self.alpha
                spam_messages.append(m)
            else:
                total_words_ham += m.number_of_words() + self.alpha
                ham_messages.append(m)
        num_spam_messages = len(spam_messages)
        num_ham_messages = len(ham_messages)
        print("num_spam_messages:", num_spam_messages)
        print("num_ham_messages:", num_ham_messages)
        self.prob_spam = log(num_spam_messages) - log(len(train_messages))
        self.prob_ham = log(num_ham_messages) - log(len(train_messages))
        self.prob_words_spam = self.calculate_words_probabilities(spam_messages)
        self.prob_words_ham = self.calculate_words_probabilities(ham_messages)
        self.default_prob_word_spam = 0.0 - log(total_words_spam)
        self.default_prob_word_ham = 0.0 - log(total_words_ham)

    def get_word_probability(self, messages: List[Message], word: str) -> float:
        occurrences_in_messages = 0
        total_words = 0
        for message in messages:
            total_words += message.number_of_words() + self.alpha
            if message.is_word_in_message(word):
                occurrences_in_messages += message.word_frequency(word)
        return log(occurrences_in_messages + self.alpha) - log(total_words)
