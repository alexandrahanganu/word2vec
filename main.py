import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from constants import stopwords_en


def get_file_data(file):
    text = []
    with open(file, 'r') as fin:
        file_content = fin.read()
        for sentence in file_content.split('.'):
            if len(sentence):
                sentence = sentence[1:] if sentence[0] == ' ' else sentence
                text.append([])
                for word in sentence.split(" "):
                    word = ''.join(re.split(r'\W+', word.lower()))
                    if word not in stopwords_en and str.isalpha(word):
                        text[len(text) - 1].append(word)
    return text


def generate_data(processed_text):
    _word_index, _index_word, _vocabulary, word_count = {}, {}, [], 0
    for sentence in processed_text:
        for word in sentence:
            _vocabulary.append(word)
            if word not in _word_index:
                _word_index.update({word: word_count})
                _index_word.update({word_count: word})
                word_count += 1
    return _word_index, _index_word, _vocabulary


def one_hot_vectors(word, context_words, size, _word_index):
    target = np.zeros(size)
    target[_word_index.get(word)] = 1
    context = np.zeros(size)
    for _word in context_words:
        context[_word_index.get(_word)] = 1
    return target, context


def generate_training_data(_vocabulary, _word_index, _threshold=2):
    training_data = []
    sample_data = []
    for index, word in enumerate(_vocabulary):
        index_target, target, context = index, word, []
        if index == 0:
            context = [_vocabulary[x] for x in range(index + 1, index + 1 + _threshold)]
        elif index == len(_vocabulary) - 1:
            context = [_vocabulary[x] for x in range(len(_vocabulary) - 2, len(_vocabulary) - 2 - _threshold, -1)]
        else:
            context = [_vocabulary[x] for x in range(index_target - 1, index_target - 1 - _threshold, -1) if x >= 0]
            context.extend(
                [_vocabulary[x] for x in range(index_target + 1, index_target + 1 + _threshold) if
                 x < len(_vocabulary)])
        target_vector, context_vector = one_hot_vectors(target, context, len(_vocabulary), _word_index)
        training_data.append([target_vector, context_vector])
        sample_data.append([target, context])
    return training_data, sample_data


def activation(x): return np.exp(x - np.max(x)) / np.exp(x - np.max(x)).sum(axis=0)


def forward_propagation(weights, target_vector):
    input_hidden = np.dot(weights[0].T, target_vector)
    hidden_output = np.dot(weights[1].T, input_hidden)
    return activation(hidden_output), input_hidden, hidden_output


def backward_propagation(weights, error, layer, target_vector, learning_rate=0.01):
    temp_weights = [np.outer(target_vector, np.dot(weights[1], error.T)), np.outer(layer, error)]
    weights[0], weights[1] = weights[0] - (learning_rate * temp_weights[0]), weights[1] - (
            learning_rate * temp_weights[1])
    return weights


def compute_error(y, context_words):
    error = [None] * len(y)
    indexes = [x for x in np.where(context_words == 1)[0]]
    for index, value in enumerate(y):
        if index in indexes:
            error[index] = value - 1 + (len(indexes) - 1) * value
        else:
            error[index] = len(indexes) * value
    return np.array(error)


def compute_loss(hidden, context):
    sum_1 = sum([hidden[x] for x in np.where(context == 1)[0]]) * -1
    sum_2 = len(np.where(context == 1)[0]) * np.log(np.sum(np.exp(hidden)))
    return sum_1 + sum_2


def train(_dimension, _epochs, size, data, learning_rate, verbose=False, interval=10):
    weight_input, weight_hidden = np.random.uniform(-1, 1, (size, _dimension)), np.random.uniform(-1, 1,
                                                                                                  (_dimension, size))
    epoch_loss, computed_weights = [], []
    _weights_1, _weights_2 = [], []
    for epoch in range(_epochs):
        _loss = 0
        for target, context in data:
            y, input_hidden, hidden_output = forward_propagation([weight_input, weight_hidden], target)
            error = compute_error(y, context)
            weight_input, weight_hidden = backward_propagation([weight_input, weight_hidden], error, input_hidden,
                                                               target, learning_rate)
            _loss += compute_loss(hidden_output, context)
        epoch_loss.append(_loss)
        _weights_1.append(weight_input)
        _weights_2.append(weight_hidden)
        if verbose:
            if epoch == 0 or epoch % interval == 0:
                print('Epoch: {value1}\tLoss: {value2}'.format(value1=epoch, value2=_loss))
    return epoch_loss, np.array(_weights_1), np.array(_weights_2)


def similarity(word, weight, _word_index, size, _index_word):
    index = _word_index[word]
    vector = weight[index]
    similar = {}
    for i in range(size):
        vector_2 = weight[i]
        theta_sum = np.dot(vector, vector_2)
        theta_den = np.linalg.norm(vector) * np.linalg.norm(vector_2)
        theta = theta_sum / theta_den
        word = _index_word[i]
        similar[word] = theta
    return similar


def print_similar_words(_word_index, _index_word, top_n_words, weight, _words_subset):
    columns = []
    for i in range(0, len(_words_subset)):
        columns.append('similar:' + str(i + 1))
    _df = pd.DataFrame(columns=columns, index=_words_subset)
    _df.head()
    row = 0
    for word in _words_subset:
        similarity_matrix = similarity(word, weight, _word_index, len(_index_word), _index_word)
        col = 0
        words_sorted = dict(sorted(similarity_matrix.items(), key=lambda x: x[1], reverse=True)[1:top_n_words + 1])
        for similar_word, similarity_value in words_sorted.items():
            _df.iloc[row][col] = (similar_word, round(similarity_value, 2))
            col += 1
        row += 1
    return _df


def word_similarity_scatter_plot(index_to_word, weight, _axes):
    labels = []
    tokens = []
    for key, value in index_to_word.items():
        tokens.append(weight[key])
        labels.append(value)
    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(tokens)
    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])
    for i in range(len(x)):
        _axes.scatter(x[i], y[i])
        _axes.annotate(labels[i],
                       xy=(x[i], y[i]),
                       xytext=(5, 2),
                       textcoords='offset points',
                       ha='right',
                       va='bottom')
    _axes.set_title('Similarities', loc='center')


if __name__ == '__main__':
    epochs = 50
    top_words = 5
    dimension = 20
    threshold = 2
    word_index, index_word, vocabulary = generate_data(get_file_data('input.txt'))
    tr_data, sa_data = generate_training_data(vocabulary, word_index, _threshold=threshold)
    loss, weights_1, weights_2 = train(dimension, epochs, len(vocabulary), tr_data, 0.01, verbose=True)
    words_subset = np.random.choice(list(word_index.keys()), top_words)
    words_subset2 = np.array(['melinda', 'bill', 'gates', 'harvard', 'software'])
    df = print_similar_words(word_index, index_word, top_words, weights_1[epochs - 1], words_subset)
    df2 = print_similar_words(word_index, index_word, top_words, weights_1[epochs - 1], words_subset2)
    fig, axes = plt.subplots(figsize=(10, 10), )
    word_similarity_scatter_plot(index_word, weights_1[epochs - 1], axes)
    plt.show()
    with open('result.html', 'w') as fout:
        fout.write(df.to_html())
        fout.write(df2.to_html())
