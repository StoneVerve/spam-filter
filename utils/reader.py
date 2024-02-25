from typing import List, Dict

from utils.message import Message
from utils.message_type import MessageType


def get_words_in_text(text: str) -> List[str]:
    text_trim = text.strip()
    if len(text) < 1:
        raise ValueError("The text does not contain any words or symbols")
    words_lower_c = text_trim.casefold().split()
    return [x.strip() for x in words_lower_c if len(x) > 0]


def get_messages_in_file(filename: str):
    messages = read_file(filename)
    messages = [remove_none_letters(m.decode('ascii')) for m in messages]
    message_list = []
    for message in messages:
        try:
            if len(message) > 0:
                words = get_words_in_text(message)
                m_words_freq = get_words_freq(words[1:])
                if words[0] == 'spam':
                    is_spam = MessageType.SPAM
                else:
                    is_spam = MessageType.HAM
                message_list.append(Message(words, m_words_freq, is_spam))
        except ValueError as err:
            print(err)
            continue
    return message_list


def get_words_freq(words: List[str]) -> Dict[str, int]:
    words_freq = dict()
    for word in words:
        if word not in words_freq:
            words_freq[word] = 1
        else:
            words_freq[word] += 1
    return words_freq


def remove_none_letters(text: str) -> str:
    ascii_text = text
    letters_text = ''
    for letter in ascii_text:
        if letter.isalpha():
            letters_text += letter
        else:
            letters_text += ' '
    return letters_text.strip()


# Returns all the messages in the file in an ascii format
def read_file(filename: str) -> list[bytes]:
    with open(filename, 'r') as f:
        messages = f.readlines()
        return [x.encode('ascii', 'replace') for x in messages]
