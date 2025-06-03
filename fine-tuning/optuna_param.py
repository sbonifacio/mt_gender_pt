import optuna
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType
from evaluate import load

import os, torch
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {torch.cuda.device_count()} GPUs!")


# TOKENIZER #######################################################
model_name = "facebook/nllb-200-1.3B"
#model_name="facebook/nllb-200-3.3B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.tgt_lang="por_Latn"


# DATASET #########################################################
def tokenize_function(example):
    model_inputs = tokenizer(example["en"], text_target=example["pt"], max_length = 512, truncation=True)
    return model_inputs

dataset_name= "handcrafted-pt.csv"
raw_dataset = load_dataset("csv", data_files=dataset_name)

# Create validation set
split_datasets = raw_dataset["train"].train_test_split(train_size=0.9, seed=20)
split_datasets["validation"] = split_datasets.pop("test")
tokenized_datasets = split_datasets.map(tokenize_function, 
                                        batched=True,
                                        remove_columns=raw_dataset["train"].column_names)


# METRICS #############################################################

bleu = load("sacrebleu")

def compute_metrics(eval_preds):
    preds, labels = eval_preds

    labels = [[(token if token != -100 else tokenizer.pad_token_id) for token in label] for label in labels]

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    return bleu.compute(predictions=decoded_preds, references=[[label] for label in decoded_labels])

# Optuna objective function
def objective(trial):
    # Suggest hyperparameters
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True)
    num_train_epochs = trial.suggest_categorical("num_train_epochs", [3,5,8,10])
    weight_decay = trial.suggest_float("weight_decay", 0.0, 0.1)
    lora_r = trial.suggest_categorical("r", [4, 6, 8])
    lora_alpha = trial.suggest_categorical("alpha", [8, 16, 32])
    lora_dropout = trial.suggest_float("dropout", 0.0, 0.1)

    # Load model
    base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Apply LoRA
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        task_type=TaskType.SEQ_2_SEQ_LM,
        target_modules=["q_proj", "v_proj"],
    )
    model = get_peft_model(base_model, lora_config)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    # Training args
    training_args = Seq2SeqTrainingArguments(
        output_dir="./trainer-optuna",
        weight_decay=weight_decay,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        evaluation_strategy="epoch",
        save_strategy = "no",
        logging_strategy = "epoch",
        predict_with_generate=True,
        fp16=True,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],  
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    result = trainer.train()
    metrics = trainer.evaluate()
    print(metrics)
    return metrics["eval_score"]  



# OPTUNA #########################################################
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=8)

print("Best hyperparameters:", study.best_params)
print("Best BLEU score:", study.best_value)

for trial in study.best_trials:
    print("Trial:", trial.number)
    print("  BLEU:", trial.value)
    print("  Params:", trial.params)


