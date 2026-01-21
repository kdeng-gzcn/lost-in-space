from datasets import load_dataset

data = load_dataset("kdeng03/VRRPI-Bench", split="train")
subset = data.shuffle(seed=42).select(range(300))
subset.push_to_hub("kdeng03/VRRPI-Bench", config_name="consistency")