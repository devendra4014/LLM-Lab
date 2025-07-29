import re
import random


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


class NGRAM:
    def __init__(self, data):
        self.probabilities = None
        self.n = None
        # self.tokens = preprocess(data)
        self.tokens = data.split()

    def create_ngram(self, n):
        self.n = n
        probability_dict = dict()

        # loop (vocab - n) + 1 times to create n gram pairs
        loops = len(self.tokens) - n + 1
        for i in range(loops):
            # get the complete pair for words
            # like for bi-gram ==> (I, am)
            # for tri-gram ==> (I, saw, cat)
            n_gram_pair = [self.tokens[i + j] for j in range(n)]

            # get the n - 1 pair
            pair = tuple(n_gram_pair[:-1])

            # get the last word from list
            next_word = n_gram_pair[-1]

            next_word_dict = probability_dict.get(pair, dict())
            next_word_dict[next_word] = next_word_dict.get(next_word, 0) + 1
            probability_dict[pair] = next_word_dict

        for pair, cnt_dict in probability_dict.items():
            total_cnt = sum(cnt_dict.values())

            for word, cnt in cnt_dict.items():
                probability_dict[pair][word] = cnt / total_cnt

        self.probabilities = probability_dict

    def generate(self, start_seq, length=50):
        generated_list = start_seq.split()
        start_seq = start_seq.split()

        if len(start_seq) < (self.n - 1):
            print(f"start seq is less than {self.n}")
        else:
            i = 0
            start_pair = tuple(start_seq[-(self.n - 1):])
            while i < (length + 1):
                next_word_prob = self.probabilities.get(start_pair)
                # print(f"{start_pair}  ==> {next_word_prob}")
                if next_word_prob is None:
                    break

                next_word = random.choices(population=list(next_word_prob.keys()), weights=list(next_word_prob.values()), k=1)[0]
                generated_list.append(next_word)
                next_pair = start_pair[-1] + " " + next_word
                start_pair = tuple(next_pair.split())

            return " ".join(generated_list)


if __name__ == '__main__':
    test_data = ("Data science is fun because data science lets us discover patterns. Data science is not just about "
                 "data; data science is about insights.")

    model = NGRAM(test_data)
    model.create_ngram(3)
    print(model.generate("Data science"))
