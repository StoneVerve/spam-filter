from utils.message_type import MessageType
from utils.message import Message
from math import log
from typing import List, Dict


def get_word_probability(messages: List[Message], word: str) -> float:
    occurrences_in_messages = 0
    for message in messages:
        if message.is_word_in_message(word):
            occurrences_in_messages += 1
    return log(occurrences_in_messages + 1) - log(len(messages) + 2)


def calculate_words_probabilities(messages: List[Message]) -> Dict[str, float]:
    words_probabilities = dict()
    for message in messages:
        words_messages = message.get_list_of_words()
        for word in words_messages:
            if word not in words_probabilities:
                words_probabilities[word] = get_word_probability(messages, word)
    return words_probabilities


class BernoulliFilter:

    def __init__(self):
        self.prob_words_spam = dict()
        self.prob_words_ham = dict()
        self.prob_ham = 0.0
        self.prob_spam = 0.0
        self.default_prob_word_spam = 0.0
        self.default_prob_word_ham = 0.0

    def train_model(self, train_messages: List[Message]):
        ham_messages = []
        spam_messages = []
        for m in train_messages:
            if m.is_spam():
                spam_messages.append(m)
            else:
                ham_messages.append(m)
        num_spam_messages = len(spam_messages)
        num_ham_messages = len(ham_messages)
        print("num_spam_messages:", num_spam_messages)
        print("num_ham_messages:", num_ham_messages)
        self.prob_spam = log(num_spam_messages) - log(len(train_messages))
        self.prob_ham = log(num_ham_messages) - log(len(train_messages))
        self.prob_words_spam = calculate_words_probabilities(spam_messages)
        self.prob_words_ham = calculate_words_probabilities(ham_messages)
        self.default_prob_word_spam = 0.0 - log(num_spam_messages + 2)
        self.default_prob_word_ham = 0.0 - log(num_ham_messages + 2)

    def prob_message_is_ham(self, message: Message) -> float:
        words = message.get_list_of_words()
        words_prob = 0.0
        for word in words:
            if word in self.prob_words_ham:
                words_prob += self.prob_words_ham[word]
            else:
                words_prob += self.default_prob_word_ham
        return self.prob_ham + words_prob

    def prob_message_is_spam(self, message: Message) -> float:
        words = message.get_list_of_words()
        words_prob = 0.0
        for word in words:
            if word in self.prob_words_spam:
                words_prob += self.prob_words_spam[word]
            else:
                words_prob += self.default_prob_word_spam
        return self.prob_spam + words_prob

    def predict_message_type(self, message: Message) -> MessageType:
        prob_message_is_ham = self.prob_message_is_ham(message)
        prob_message_is_spam = self.prob_message_is_spam(message)
        if prob_message_is_ham > prob_message_is_spam:
            return MessageType.HAM
        else:
            return MessageType.SPAM
