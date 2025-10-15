from utils import load_model_and_tokenizer, load_data, full_finetuning, lora, compute_metrics

def run(model_name, target_modules):
    print(f"\nRunning experiments for: {model_name}\n")
    
    model, tokenizer = load_model_and_tokenizer(model_name)
    train_df, val_df, test_df = load_data(tokenizer)

    ## full fine-tuning
    trainer_full = full_finetuning(model, train_df, val_df, tokenizer, compute_metrics)
    trainer_full.train()
    results_full = trainer_full.evaluate(test_df)
    print("Full fine-tuning results:", results_full)

    # lora fine-tuning
    model, tokenizer = load_model_and_tokenizer(model_name)  # reload clean model
    trainer_lora = lora(model, target_modules, train_df, val_df, tokenizer, compute_metrics)
    trainer_lora.train()
    results_lora = trainer_lora.evaluate(test_df)
    print("LoRA results:", results_lora)


if __name__ == "__main__":
    # define models + corresponding LoRA target modules
    experiments = [
        {
            "model_name": "cardiffnlp/twitter-roberta-base-sentiment",
            "target_modules": ["query", "value"]
        },
        {
            "model_name": "bhadresh-savani/distilbert-base-uncased-emotion",
            "target_modules": ["q_lin", "v_lin"]
        }
    ]

    for exp in experiments:
        run(exp["model_name"], exp["target_modules"])
