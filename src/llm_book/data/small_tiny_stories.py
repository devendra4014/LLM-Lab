from datasets import load_dataset, load_from_disk


def get_data_from_repo(repo="roneneldan/TinyStories"):
    ds = load_dataset(repo, split="train[:%]")
    return ds


def load_from_local():
    dataset = load_dataset(
        "parquet",
        data_files="E:\code\DL\LLM-Lab\data\small_tiny_stories\part-00000-221bb1a7-7aa9-4850-bf6f-a9fdb712fa71-c000.snappy.parquet",
    )
    print(dataset["train"][0:5])
