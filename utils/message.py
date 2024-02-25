from typing import List

from utils.message_type import MessageType


class Message:

    def __init__(self, words: List[str], freq: dict, spam: MessageType):
        self.words = words
        self.freq = freq
        self.spam = spam

    def __str__(self):
        return str(self.words)

    def is_spam(self) -> bool:
        return self.spam == MessageType.SPAM

    def is_word_in_message(self, word: str) -> bool:
        return word in self.freq

    def word_frequency(self, word: str) -> int:
        if word not in self.freq:
            raise ValueError(f"Word '{word}' is not in the message")
        else:
            return self.freq[word]

    def get_message(self) -> str:
        return ''.join(self.words)

    def number_of_words(self) -> int:
        return len(self.words)

    def get_list_of_words(self) -> List[str]:
        return list(self.words)
