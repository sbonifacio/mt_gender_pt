# fine tune with optuna parameters
from peft import LoraConfig, get_peft_model,TaskType

lora_r=8
lora_alpha=32
lora_dropout=0.1

LEARNING_RATE = 5.6e-5
NUM_TRAIN_EPOCHS = 3
WEIGHT_DECAY=0.09

import os, torch
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {torch.cuda.device_count()} GPUs!")




#############################################
#               LORA CONFIG                 #
#############################################

# Define adapter configuration
config = LoraConfig(
    r=lora_r,               # Dimension of low-rank matrices (smaller = more compression)
    lora_alpha=lora_alpha,      # Scaling factor for low rank matrices (higher = stronger adaptation) 
    lora_dropout=lora_dropout, #Dropout probability of the LoRa layers 
    inference_mode=False,
    task_type=TaskType.SEQ_2_SEQ_LM,
    target_modules=["q_proj", "v_proj"]

)

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
model_name = "facebook/nllb-200-1.3B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer.tgt_lang="por_Latn"
peft_model = get_peft_model(model, config)
peft_model.print_trainable_parameters()



#############################################
#                 DATASET                   #
#############################################


from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import load_dataset


def tokenize_function(example):
    model_inputs = tokenizer(example["en"], text_target=example["pt"], max_length = 512, truncation=True)
    return model_inputs


# Load custom dataset
dataset_name= "handcrafted-pt.csv"
raw_dataset = load_dataset("csv", data_files=dataset_name)

# Create validation set
split_datasets = raw_dataset["train"].train_test_split(train_size=0.9, seed=20)
split_datasets["validation"] = split_datasets.pop("test")


tokenized_datasets = split_datasets.map(tokenize_function, 
                                        batched=True,
                                        remove_columns=raw_dataset["train"].column_names)

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)


#############################################
#                METRICS                   #
#############################################


from evaluate import load
bleu = load("sacrebleu")

def compute_metrics(eval_preds):
    preds, labels = eval_preds

    labels = [[(token if token != -100 else tokenizer.pad_token_id) for token in label] for label in labels]

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    return bleu.compute(predictions=decoded_preds, references=[[label] for label in decoded_labels])


#############################################
#                TRAINING                   #
#############################################


training_args = Seq2SeqTrainingArguments(
    output_dir="./trainer_overlap",
    weight_decay=WEIGHT_DECAY,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=LEARNING_RATE,
    num_train_epochs=NUM_TRAIN_EPOCHS, 
    evaluation_strategy="epoch",
    save_strategy = "epoch",
    predict_with_generate=True,
    fp16=True
)

trainer = Seq2SeqTrainer(
    model=peft_model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
peft_model.save_pretrained("./overlap_model")


