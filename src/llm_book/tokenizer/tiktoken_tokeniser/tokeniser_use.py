import tiktoken

"""
Tiktoken provides three encoding models optimized for different use cases:

`o200k_base`: Encoding for the newest GPT-4o-Mini model.
`cl100k_base`: Encoding model for newer OpenAI models such as GPT-4 and GPT-3.5-Turbo.
`p50k_base`: Encoding for Codex models, these models are used for code applications.
`r50k_base`: Older encoding for different versions of GPT-3.

"""


def get_encoder_by_name(encoder_name):
    # getting encoder by name
    encoder = tiktoken.get_encoding(encoder_name)

    assert encoder.decode(encoder.encode("Hello World")) == "Hello World"

    # print("encoder output : ")
    # print(encoder.encode("Hello World"))
    return encoder


def get_encoder_by_model(model_name):
    # To get the tokeniser corresponding to a specific model in the OpenAI API:
    enc = tiktoken.encoding_for_model("gpt-4o")
    return enc


def get_encoder_details(enc):
    detail = [
        "name",
        "_pat_str",
        "_special_tokens",
        "max_token_value",
        "special_tokens_set",
    ]
    info = dict(enc.__dict__)
    for d in detail:
        print(f"<<< {d} >>> : {info[d]}")


enc = get_encoder_by_name("o200k_base")
get_encoder_details(enc)
