from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
from datasets import Dataset

model_name = "distilgpt2"
data = Dataset.from_list([
    {
        "text": "Question: How do I reset my password?\nAnswer: Go to Settings > Security > Reset Password."
    },
    {
        "text": "Question: How do I update my email?\nAnswer: Open Profile Settings and edit your email address."
    }
])

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

def tokenize(example):
    tokens = tokenizer(example["text"], padding="max_length", truncation=True, max_length=128)
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

tokenized_data = data.map(tokenize)

base_model = AutoModelForCausalLM.from_pretrained(model_name)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["c_attn"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM"
)

model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()

training_args = TrainingArguments(
    output_dir="./lora-finetuned-model",
    per_device_train_batch_size=2,
    num_train_epochs=3,
    logging_steps=1,
    save_strategy="epoch"
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data
)
trainer.train()