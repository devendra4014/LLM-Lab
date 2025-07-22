import os
from nltk.stem import PorterStemmer
import re

# Create an instance of PorterStemmer
ps = PorterStemmer()


def bpe(text):
    """
    Perform Byte Pair Encoding (BPE) on the input text.
    """
    text = "Stemming 123 ! is a technique in natural language processing (NLP) that reduces words to their base or root form."
    split = re.findall(r'[^ ]+| ', text)
    print("Original split:")
    print(split)
    print()
    words = []

    # Apply stemming
    for w in split:
        if w.strip() == "":
            words.append(w)
        else:
            stem_value = ps.stem(w, to_lowercase=False)
            words.append(stem_value)
            remaining_word = w.split(stem_value)[-1]
            # If the stemmed value is not empty, append it to the list  
            if remaining_word.strip() != "":
                words.append("##" + remaining_word)

    print("Words after stemming:", words)


if __name__ == "__main__":
    folder = 'data'
    file_name = 'starwars.txt'
    file_path = os.path.join(folder, file_name)
    sample_data = ""
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
            sample_data = text[:1000]

    bpe(sample_data)

# def BPE(text, num_merges):
#     """
#     Perform Byte Pair Encoding (BPE) on the input text.

#     Args:
#         text (str): The input text to encode.
#         num_merges (int): The number of merges to perform.

#     Returns:
#         str: The BPE encoded text.
#     """
#     # Initialize the vocabulary with individual characters
#     vocab = {}
#     for char in text:
#         vocab[char] = vocab.get(char, 0) + 1

#     for _ in range(num_merges):
#         # Find the most frequent pair
#         pairs = {}
#         for i in range(len(text) - 1):
#             pair = text[i:i + 2]
#             pairs[pair] = pairs.get(pair, 0) + 1

#         if not pairs:
#             break

#         most_frequent_pair = max(pairs, key=pairs.get)

#         # Merge the most frequent pair
#         new_text = []
#         i = 0
#         while i < len(text):
#             if i < len(text) - 1 and text[i:i + 2] == most_frequent_pair:
#                 new_text.append(most_frequent_pair)
#                 i += 2
#             else:
#                 new_text.append(text[i])
#                 i += 1

#         text = ''.join(new_text)

#     return text
