from utils.reader import get_messages_in_file
from utils.message import Message
from utils.message import MessageType
from filters.bernoulli_filter import BernoulliFilter

if __name__ == '__main__':
    try:
        messages = get_messages_in_file("data/messages.txt")
        spam_messages = []
        ham_messages = []
        for message in messages:
            if message.spam == MessageType.SPAM:
                spam_messages.append(message)
            else:
                ham_messages.append(message)
        print('amount of spam messages', len(spam_messages))
        print('amount of ham messages', len(ham_messages))
        message_filter = BernoulliFilter()
        num_messages = len(messages)
        num_train_messages = (80 * num_messages) // 100
        training_set = messages[:num_train_messages]
        test_set = messages[num_train_messages:]
        print("Number of training messages:", num_train_messages)
        print("Number of messages:", num_messages)
        message_filter.train_model(training_set)
        num_errors = 0
        for m in test_set:
            y = message_filter.predict_message_type(m)
            if m.spam != y:
                num_errors += 1
                #print('Correct message:', m.spam)
        print('Percentage wrong', num_errors * 100 / len(test_set))
    except FileNotFoundError:
        print("We couldn't find the file")
