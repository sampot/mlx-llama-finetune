from datasets import load_dataset

system_message = """You are an text to SQL query translator. Users will ask you questions in English and you will generate a SQL query based on the provided SCHEMA.
SCHEMA:
{schema}"""


def create_prompt(sample):
    return {
        "prompt": f"""schema:\n{sample["sql_context"]}"\n{sample["sql_prompt"]}""",
        "completion": sample["sql"],
    }


dataset = load_dataset("gretelai/synthetic_text_to_sql", split="train")
dataset = dataset.shuffle().select(range(1000))
dataset = dataset.map(create_prompt, remove_columns=dataset.features, batched=False)
dataset = dataset.train_test_split(test_size=30 / 100)
print(dataset["train"][45]["prompt"])
dataset["train"].to_json("./data/train.jsonl", orient="records")
valid_and_test_data = dataset["test"].train_test_split(30 / 100)
valid_and_test_data["train"].to_json("./data/valid.jsonl", orient="records")
valid_and_test_data["test"].to_json("./data/test.jsonl", orient="records")
