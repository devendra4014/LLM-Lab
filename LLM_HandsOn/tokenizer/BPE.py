import itertools
import os
import regex as re

GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[
\r\n]|\s+(?!\S)|\s+"""


def split_with_leading_spaces(s):
    # Match optional leading spaces + non-space word
    matches = re.findall(r'\s*\S+', s)
    return matches


def get_freq_cnt(tokens, freq):
    freq = dict() if freq is None else freq

    for ch1, ch2 in zip(tokens[:], tokens[1:]):
        pair = (ch1, ch2)
        freq[pair] = freq.get(pair, 0) + 1

    return freq


class BytePairEncoding:
    def __init__(self):
        # self.tokens = list(str(txt).encode("utf-8"))
        self.compiled_pattern = re.compile(GPT4_SPLIT_PATTERN)
        self.merges = dict()
        self.vocab = {idx: bytes([idx]) for idx in range(256)}
        # self.num_iterations = num_iterations

    @staticmethod
    def merge(tokens, merge_pair, pair_id):
        new_tokens = []
        i = 0

        while i < len(tokens):
            if i < len(tokens) - 1 and tokens[i] == merge_pair[0] and tokens[i + 1] == merge_pair[1]:
                new_tokens.append(pair_id)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        return new_tokens

    def train(self, text, vocab_size):
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        # split the text up into text chunks
        text_chunks = re.findall(self.compiled_pattern, text)

        # input text preprocessing
        ids = [list(ch.encode("utf-8")) for ch in text_chunks]

        for i in range(num_merges):
            pair_freq = dict()
            for chunk in ids:
                # get the frequency of adjacent pairs
                pair_freq = get_freq_cnt(chunk, pair_freq)

            if not pair_freq:
                break

            most_frequent_pair = max(pair_freq, key=pair_freq.get)

            # merge the most freq pair
            pair_id = 256 + i
            ids = [self.merge(chunk, most_frequent_pair, pair_id) for chunk in ids]
            self.merges[most_frequent_pair] = pair_id

        # create a final vocab dictionary
        for (p0, p1), idx in self.merges.items():
            self.vocab[idx] = self.vocab[p0] + self.vocab[p1]

    def encode_chunk(self, text_bytes):
        ids = list(text_bytes)

        while len(ids) >= 2:
            stats = get_freq_cnt(ids, {})

            # min is used to get a pair which has minimum index in the merges dictionary
            pair = min(stats, key=lambda k: self.merges.get(k, float('inf')))

            if pair not in self.merges:
                break

            idx = self.merges[pair]
            ids = self.merge(ids, pair, idx)
        return ids

    def encode(self, txt):
        text_chunks = re.findall(self.compiled_pattern, txt)
        ids = []

        for chunk in text_chunks:
            byte_chunk = chunk.encode('utf-8')
            enc_ids = self.encode_chunk(byte_chunk)
            ids.extend(enc_ids)

        return ids

    def decode(self, ids):
        tokens = b"".join(self.vocab[i] for i in ids)
        result = tokens.decode('utf-8', errors='replace')
        return result


# text = ("Stemming 123 ! is a technique in natural language processing (NLP) that reduces words to their base or root "
#         "form.")


if __name__ == "__main__":
    sample = "Stemming 123 ! is a technique in natural language processing (NLP) that reduces words to their base or root "

    file_path = os.path.join("data", "the-verdict.txt")
    data = None
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            data = f.read()

    # print(data)
    bpe = BytePairEncoding()
    bpe.train(data, 500)

    print(bpe.decode(bpe.encode(sample)))
    print(bpe.encode(sample))
