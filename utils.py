from datasets import load_dataset, DatasetDict
from transformers import TrainingArguments, Trainer, AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import matthews_corrcoef, precision_recall_fscore_support, accuracy_score, classification_report
from peft import LoraConfig, get_peft_model, TaskType
import pandas as pd
import numpy as np


## load model & tokenizer
def load_model_and_tokenizer(model_name, num_labels=3):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels, ignore_mismatched_sizes=True
    )
    return model, tokenizer


def tokenize_function(examples, tokenizer):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)


## load & prepare dataset
def load_data(tokenizer):
    df = load_dataset("cardiffnlp/tweet_eval", "sentiment")
    tokenized_df = df.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    return tokenized_df["train"], tokenized_df["validation"], tokenized_df["test"]


## full fine-tuning
def full_finetuning(model, train_df, val_df, tokenizer, compute_metrics):
    
    ## training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir="./logs",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_df,
        eval_dataset=val_df,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )

    return trainer


## lora fine-tuning
def lora(model, target_modules, train_df, val_df, tokenizer, compute_metrics):

    ## training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir="./logs",
    )

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=8,                            ## rank
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=target_modules
    )

    lora_model = get_peft_model(model, lora_config)

    trainer_lora = Trainer(
        model=lora_model,
        args=training_args,
        train_dataset = train_df,
        eval_dataset=val_df,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )

    return trainer_lora


## metrics (evaluation)
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    mcc = matthews_corrcoef(labels, preds)
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mcc": mcc
    }

## error analysis
def error_analysis(test_df, labels, preds_full, preds_lora, save_path="error_analysis.csv", num_examples=5):
    if isinstance(test_df, dict) or hasattr(test_df, "to_dict"):
        texts = test_df["text"]
    else:
        raise ValueError("Expected Hugging Face dataset with a 'text' column.")

    df = pd.DataFrame({
        "text": texts,
        "label": labels,
        "pred_full": preds_full,
        "pred_lora": preds_lora
    })

    df["full_correct"] = df["label"] == df["pred_full"]
    df["lora_correct"] = df["label"] == df["pred_lora"]

    # error categories
    df["error_type"] = "both_correct"
    df.loc[(~df["full_correct"]) & (~df["lora_correct"]), "error_type"] = "both_wrong"
    df.loc[(df["full_correct"]) & (~df["lora_correct"]), "error_type"] = "full_better"
    df.loc[(~df["full_correct"]) & (df["lora_correct"]), "error_type"] = "lora_better"

    # save all errors
    df.to_csv(save_path, index=False)
    print(f"\nError analysis saved to: {save_path}")

    # print classification reports
    print("\nFull Fine-tuning Classification Report:")
    print(classification_report(df["label"], df["pred_full"]))

    print("\nLoRA Fine-tuning Classification Report:")
    print(classification_report(df["label"], df["pred_lora"]))

    # print a few example misclassifications
    print("\nExamples Where LoRA is correct but Full is wrong")
    lora_better = df[df["error_type"] == "lora_better"].head(num_examples)
    for _, row in lora_better.iterrows():
        print(f"\nText: {row['text']}")
        print(f"Label: {row['label']} | Full: {row['pred_full']} | LoRA: {row['pred_lora']}")

    print("\nExamples Where Full is correct but LoRA is wrong")
    full_better = df[df["error_type"] == "full_better"].head(num_examples)
    for _, row in full_better.iterrows():
        print(f"\nText: {row['text']}")
        print(f"Label: {row['label']} | Full: {row['pred_full']} | LoRA: {row['pred_lora']}")
