import re

import nltk

sample = """
The intuition behind an n-gram model is to estimate the probability of a word sequence by considering only a limited context of the preceding n-1 words. Instead of looking at the entire history of a sentence, it simplifies the problem by assuming that the probability of the next word depends only on the immediately preceding words. 
Here's a more detailed explanation:
The Problem:
In natural language processing, especially for tasks like speech recognition or machine translation, we often need to predict the next word in a sequence. A full probability model would consider the entire history of the sequence, which can be computationally expensive and difficult to estimate accurately. 
The Solution (N-grams):
N-gram models approximate this by considering only the n-1 previous words. This makes the problem more manageable and allows us to train models on large text corpora. 
Example (Bigrams):
A bigram model (n=2) considers only the immediately preceding word. For example, to predict the probability of "rain" given "heavy", it would look at how often "heavy rain" occurs in the training data, compared to "heavy" followed by other words. It essentially simplifies the calculation to P(word | previous word), according to LunarTech. 
Generalization (N-grams):
A trigram model (n=3) would consider the two preceding words, a 4-gram would consider three, and so on. The higher the value of n, the more context is considered, but also the more data is needed to train the model reliably, notes LunarTech. 
Markov Assumption:
This simplification, where the future depends only on a limited past, is known as the Markov assumption. 
Practical Use:
N-gram models are used in a variety of applications, including:
Speech recognition: Predicting the most likely sequence of words based on acoustic input. 
Machine translation: Helping to determine the most probable translation of a sentence. 
Spelling correction and auto-completion: Suggesting the most likely word based on the preceding text. 
Plagiarism detection: Identifying instances of copied or paraphrased text by comparing n-gram sequences
"""


def preprocess(text):
    def lowercase(txt):
        """Get Lower-Case Text"""
        return str(txt).lower()

    def remove_symbols(txt):
        """remove characters which are not letter or numbers or space"""
        return re.sub(r"[^a-zA-Z0-9 ]", "", txt)

    def get_tokens(txt):
        """Get the list of words separated by one or more spaces"""
        return str(txt).split()

    # apply lower case
    text = remove_symbols(lowercase(text))

    return get_tokens(text)


def paired_ngram_probabilities(ngram_paired_count):
    ngram_prob = dict()
    for ngram, words in ngram_paired_count.items():
        total = len(words)
        cnt_dict = dict()

        for word in words:
            cnt_dict[word] = cnt_dict.get(word, 0) + 1

        prob_dict = {word: cnt / total for word, cnt in cnt_dict.items()}
        sorted_prob = dict(sorted(prob_dict.items(), key=lambda i: i[1], reverse=True))

        ngram_prob[ngram] = sorted_prob

    return ngram_prob


class Ngram01:
    def __init__(self, data, n):
        self.data = data
        self.n = n
        self.n_gram_probabilities = None
        self.tokenized_sentences = None

        self.ngram()

    def tokenize_sentences(self, sentences):
        """
        Tokenize sentences into tokens (words)

        Args:
            sentences: List of strings

        Returns:
            List of lists of tokens
        """

        # Initialize the list of lists of tokenized sentences
        tokenized_sentences = []

        # Go through each sentence
        for sentence in sentences:
            # Convert to lowercase letters
            sentence = sentence.lower()

            # Convert into a list of words
            tokenized = preprocess(sentence)

            # append the list of words to the list of lists
            tokenized_sentences.append(tokenized)

        self.tokenized_sentences = tokenized_sentences

    def count_ngram(self):
        """
        Get a dictionary of count of n-grams

        Args:
            tokenized_sentences: List of lists of tokens
            n: number of n-grams

        Returns:
            List of lists of tokens
        """
        n_gram_count = dict()

        for sentence in self.tokenized_sentences:
            # "<S>" - is Start Token
            # "</S>"] - is End Token
            tokens = ["<S>"] * self.n + sentence + ["</S>"]

            # start loop from 0 to len(words) - n elements in a list
            for i in range(len(tokens) - self.n + 1):
                n_gram_tuple = tuple(tokens[i: i + self.n])

                # add to dict or increment count if present
                n_gram_count[n_gram_tuple] = n_gram_count.get(n_gram_tuple, 0) + 1

        return n_gram_count

    def get_ngram_paired_count(self):
        ngram_pairs = dict()

        for sentence in self.tokenized_sentences:
            # tokens = ["<S>"] * n + sentence + ["<S>"]
            tokens = sentence
            for i in range(len(tokens) - self.n):
                pair = tuple(tokens[i:i + self.n])
                ngram_pairs[pair] = ngram_pairs.get(pair, []) + [tokens[i + self.n]]

        return ngram_pairs

    def generate_sequence(self, start_token, max_length=20):
        result = []
        start_token = preprocess(start_token)

        if len(start_token) < self.n:
            start_token = ["<S>"] * (self.n - len(start_token)) + start_token

        pair = start_token[-self.n - 1:]
        print(f"first pair => {pair}")

        for i in range(max_length):
            probabilities = self.n_gram_probabilities.get(tuple(pair))

            if probabilities:
                key = list(probabilities)[0]
                # value = probabilities.get(key)
                result.append(key)
                pair = pair[1:] + [key]

        print(result)

    def ngram(self):
        # split sentences
        sentences = self.data.split(".")

        # tokenize sentences
        self.tokenize_sentences(sentences)
        cnt_ngram = self.get_ngram_paired_count()
        self.n_gram_probabilities = paired_ngram_probabilities(cnt_ngram)


class Ngram02:
    def __init__(self, data, n):
        self.tokens = None
        self.n_gram_model = None
        self.data = data
        self.n = n

        self.create_tokens()
        self.create_ngram_model()

    def create_tokens(self):
        self.tokens = self.data.split()

    def create_ngram_model(self):
        # create empty dictionary
        n_gram_model = {}

        # loop through given text data with window of size n upto len(text) - n
        for i in range(len(self.tokens) - self.n):
            # creating tuple of n_grams of size 'n' and assign as a key to n_gram_model
            n_gram_key = tuple(self.tokens[i:i + self.n])

            # next word
            next_word = self.tokens[i + self.n]

            # checking for key in dictionary if not present then create new empty dictionary for that key
            if n_gram_model.get(n_gram_key) is None:
                n_gram_model[n_gram_key] = {}

            n_gram_model[n_gram_key][next_word] = n_gram_model[n_gram_key].get(next_word, 0) + 1

        # calculating relative probabilities for each possible words

        for ngram, next_words in n_gram_model.items():
            total = sum(next_words.values())
            for word in next_words:
                n_gram_model[ngram][word] /= total

        # print(n_gram_model)
        self.n_gram_model = n_gram_model

    def generate_ngram_text(self, start_words, length):
        import random

        generated_text = start_words.split()
        for _ in range(length):
            # print(f"generated_text => {generated_text}")
            current_ngram = tuple(generated_text[-self.n:])
            #         print(current_ngram)
            if current_ngram in self.n_gram_model:
                next_word_probs = self.n_gram_model[current_ngram]
                next_word = random.choices(list(next_word_probs.keys()), list(next_word_probs.values()))[0]
                generated_text.append(next_word)
            else:
                break
        result = " ".join(generated_text)
        return result


if __name__ == "__main__":
    with open("../data/shakespeare.txt", 'r', encoding='utf-8', errors='ignore') as f:
        sample_data = f.read()

    start = "I can do"
    # ngram01_model = Ngram01(sample_data, 3)
    # ngram01_model.generate_sequence(start)

    ngram02_model = Ngram02(sample_data, 3)
    response_txt = ngram02_model.generate_ngram_text(start, 50)
    print(response_txt)
