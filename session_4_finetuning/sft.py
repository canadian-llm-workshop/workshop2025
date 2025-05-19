import warnings
warnings.filterwarnings('ignore')
import os
import json
import re
from tqdm import tqdm
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, TaskType
from trl import SFTTrainer, SFTConfig


BASE_DIR = "./llm_instruct_demo"
os.makedirs(BASE_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
trained_model_path = f"{BASE_DIR}/models"


def load_model(model_name):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",  # Automatically place the model on available GPU
        torch_dtype=torch.float16  # Force the model to load in FP16 precision
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

def instruction_tune_model(base_model, tokenizer, finetuning_dataset, epochs=3):
    """Fine-tune the model on instruction data."""
    # LoRA configuration    
    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        # TinyLlama ≈ Llama-2 layer names
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",      # attention
            "gate_proj", "up_proj", "down_proj"          # feed-forward
        ]
    )
    
    # Training configuration
    training_args = SFTConfig(
        output_dir=trained_model_path,
        num_train_epochs=epochs,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        report_to="tensorboard",
        dataset_text_field="text",
        max_seq_length=512,
        packing=False,
    )
    
    trainer = SFTTrainer(
        model=base_model,
        train_dataset=finetuning_dataset,
        args=training_args,
        peft_config=lora_config
    )
    trainer.train()
    
    # Save the final model
    trainer.model.save_pretrained(f"{trained_model_path}/final")
    tokenizer.save_pretrained(f"{trained_model_path}/final")
    return trainer.model


def extract_date(text: str) -> str:
    """
    Return the first ISO date in the first JSON object found in `text`.
    If no valid {"date": "..."} object exists, return an empty string.
    """
    # look for the first {...} block (non-greedy)
    for match in re.finditer(r'\{[^{}]*\}', text):
        try:
            obj = json.loads(match.group())
            if isinstance(obj, dict) and "date" in obj:
                return obj["date"][:10]
        except json.JSONDecodeError:
            continue   # keep scanning for the next brace block
    return ""


def call_llm(model, tokenizer, prompt: str,
             max_new_tokens: int = 256, temperature: float = 0.0) -> str:
    with torch.no_grad():
        out = model.generate(
            **tokenizer(prompt, return_tensors="pt").to("cuda"),
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
        )
    model_output = tokenizer.decode(out[0], skip_special_tokens=True)
    return model_output.split('<|assistant|>')[-1].strip()


def run_inference(
    model,
    tokenizer,
    hf_dataset,
    *,
    system_prompt: str | None = None,
    max_samples: int | None = None,
):
    model.eval()
    records = []
    dataset_iter = hf_dataset if max_samples is None else hf_dataset.select(range(max_samples))

    for row in tqdm(dataset_iter, desc="running"):
        text = row["text"]

        try:
            before, after = text.split("OUTPUT:")
        except ValueError:
            continue  # skip malformed rows

        base_prompt = f"{before}OUTPUT:".strip()
        groundtruth = after.strip()

        # ---- build TinyLlama-style prompt ----------------------------------
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt.strip()})
        messages.append({"role": "user", "content": base_prompt})
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        # --------------------------------------------------------------------

        prediction = call_llm(model, tokenizer, prompt, max_new_tokens=32, temperature=0.0)

        records.append(
            {
                "input": prompt,
                "groundtruth": groundtruth,
                "prediction": prediction,
            }
        )

    return clean_outputs(records)


def clean_outputs(records):
    df = pd.DataFrame(records)
    df['groundtruth_date'] = df['groundtruth'].apply(extract_date)
    df['predicted_date'] = df['prediction'].apply(extract_date)
    return df

if __name__ == '__main__':
    from datasets import load_dataset
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    base_model, base_tokenizer = load_model(model_name)
    ds_test = load_dataset("najeebk/date-format-dataset-v0", split="test")
    
    prompt = 'Extract the event date; return JSON {"date":"YYYY-MM-DD"} only.'
    df = run_inference(
        base_model, base_tokenizer, ds_test, system_prompt=prompt, max_samples=500)
    error_df = df[df.predicted_date != df.groundtruth_date]

    print(f"Base model error: {100*len(error_df)/len(df)}%")
    
    ds_train = load_dataset("najeebk/date-format-dataset-v0", split="train")
    instruction_model = instruction_tune_model(base_model, base_tokenizer, ds_train)
    
    df_ft = run_inference(
        instruction_model, base_tokenizer, ds_test, system_prompt='', max_samples=500)
    
    error_df_ft = df_ft[df_ft.predicted_date != df_ft.groundtruth_date]
    
    print(f"Fine tuned model error: {100*len(error_df_ft)/len(df_ft)}%")
    
