from utils.message_type import MessageType
from utils.message import Message
from typing import List, Dict
from abc import ABC, abstractmethod


class NBayesFilter(ABC):

    def __init__(self):
        self.prob_words_spam = dict()
        self.prob_words_ham = dict()
        self.prob_ham = 0.0
        self.prob_spam = 0.0
        self.default_prob_word_spam = 0.0
        self.default_prob_word_ham = 0.0

    @abstractmethod
    def train_model(self, train_messages: List[Message]):
        pass

    @abstractmethod
    def get_word_probability(self, messages: List[Message], word: str) -> float:
        pass

    def calculate_words_probabilities(self, messages: List[Message]) -> Dict[str, float]:
        words_probabilities = dict()
        for message in messages:
            words_messages = message.get_list_of_words()
            for word in words_messages:
                if word not in words_probabilities:
                    words_probabilities[word] = self.get_word_probability(messages, word)
        return words_probabilities

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
