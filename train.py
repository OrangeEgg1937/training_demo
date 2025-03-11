import gc
import torch
import wget
from modelscope import AutoModelForCausalLM, AutoTokenizer
from transformers import TrainingArguments, Trainer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model

# Step 1: 下载并导入数据集
dataset_url = 'https://oss-pai-psfimcan9ur6rixdqe-cn-shanghai.oss-cn-shanghai.aliyuncs.com/zh2en_cleaned.csv?Expires=1741673814&OSSAccessKeyId=TMP.3Kks4VucAp3CdMmPsHPsfvziTSQDTGZiqUeJ219NWjcBbB136sTSzXsZ246FPxCbLvScQRuHXQe8xMgH2NLMwyPwo44efL&Signature=z7H0Xx3PmdXdd9Cvb8AJFTIF%2B20%3D'
wget.download(dataset_url)
dataset = load_dataset('csv', data_files='zh2en_cleaned.csv')
dataset = dataset['train'].train_test_split(test_size=0.2, seed=42)
train_dataset = dataset['train']
val_dataset = dataset['test']

# Step 2: 加载tokenizer和格式data
model_name = 'ZhipuAI/glm-4-9b-chat-hf'
tokenizer = AutoTokenizer.from_pretrained(model_name)

def format_example(example):
    zh_text = example['zh_text']
    en_text = example['en_text']
    prompt = f"[SYSTEM] You are a game localization expert. Translate the following Chinese game text into English: [USER] {zh_text} [OUTPUT] {en_text}"
    return {'text': prompt}

train_dataset = train_dataset.map(format_example)
val_dataset = val_dataset.map(format_example)

def tokenize_function(example):
    return tokenizer(example['text'], truncation=True, max_length=512, padding='max_length')

tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
tokenized_val_dataset = val_dataset.map(tokenize_function, batched=True)

tokenized_train_dataset = tokenized_train_dataset.remove_columns(['zh_text', 'en_text', 'text'])
tokenized_val_dataset = tokenized_val_dataset.remove_columns(['zh_text', 'en_text', 'text'])

tokenized_train_dataset.set_format('torch')
tokenized_val_dataset.set_format('torch')

# Step 3: Load the model with LoRA
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    load_in_8bit=True
)

lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Step 4: Set up training arguments
training_args = TrainingArguments(
    output_dir='./fine_tuned_model',
    num_train_epochs=3,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    fp16=True,
    logging_dir='./logs',
    logging_steps=10,
    load_best_model_at_end=True,
)

# Step 5: Initialize the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    tokenizer=tokenizer,
)

# Step 6: Train the model
trainer.train()

# Step 7: Save the fine-tuned model
model.save_pretrained('./fine_tuned_model')
tokenizer.save_pretrained('./fine_tuned_model')

# Step 8: Cleanup
del model
del tokenizer
if torch.cuda.is_available():
    torch.cuda.empty_cache()
gc.collect()
print("Model and tokenizer have been released from memory.")