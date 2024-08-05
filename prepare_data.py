from datasets import load_dataset

system_message = """You are an text to SQL query translator. Users will ask you questions in English and you will generate a SQL query based on the provided SCHEMA."""


def create_prompt(sample):
    prompt = f"""SCHEMA: {sample["sql_context"]}\n{sample["sql_prompt"]}"""
    return {
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": sample["sql"]},
        ]
    }


dataset = load_dataset("gretelai/synthetic_text_to_sql", split="train")
dataset = dataset.shuffle().select(range(1000))
dataset = dataset.map(create_prompt, remove_columns=dataset.features, batched=False)
dataset = dataset.train_test_split(test_size=20 / 100)
print(dataset["train"][45]["messages"])
dataset["train"].to_json("./data/train.jsonl", orient="records")
valid_and_test_data = dataset["test"].train_test_split(50 / 100)
valid_and_test_data["train"].to_json("./data/valid.jsonl", orient="records")
valid_and_test_data["test"].to_json("./data/test.jsonl", orient="records")
