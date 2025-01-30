#Load pretrained model, handle padding and truncation
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling,Trainer, TrainingArguments

from datasets import Dataset #Load Dataset (JSON file)

# Preprocess the dataset
def preprocess_data(example):
    # Tokenize the input and response
    input_encoding = tokenizer(example["input"], truncation=True, padding="max_length", max_length=128)
    response_encoding = tokenizer(example["response"], truncation=True, padding="max_length", max_length=128)

    # Return tokenized inputs and labels
    return {
        "input_ids": input_encoding["input_ids"],
        "attention_mask": input_encoding["attention_mask"],
        "labels": response_encoding["input_ids"]  # Labels are the tokenized responses
    }

#load the pretrained model
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

# Load a conversational model
chatbot = pipeline('text-generation', model='microsoft/DialoGPT-medium')

#Tokenize Data, pad tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
tokenizer.pad_token = tokenizer.eos_token

#Load Your Dataset
data = Dataset.from_json("me.json")
# Apply preprocessing to the dataset
tokenized_dataset = data.map(preprocess_data, batched=True)

print(tokenized_dataset[0])

#Data Collator to handle padding and truncation(creates rectangular tensors from batches of varying lengths)
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # mlm = Masked Language Model, set to False for GPT models
)

#Setting up the Trainer
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="no",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    num_train_epochs=3,
    save_strategy="epoch",
    logging_dir="./logs",
    remove_unused_columns=False  # Ensure all dataset columns are passed to the model
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset, #Pass the preprocessed dataset
    tokenizer=tokenizer,
    data_collator=data_collator
)

trainer.train()
model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")

