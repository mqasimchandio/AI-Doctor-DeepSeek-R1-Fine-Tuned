# AI Doctor DeepSeek R1 Fine Tuning
## 🏥 Overview

This project fine‑tunes the `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` model on a supervised medical Q&A dataset with chain‑of‑thought prompts. The resulting model acts as an AI‑Doctor chatbot capable of providing clear, cautious medical guidance while reminding users to seek professional help.

---

## ✨ Features

- **PEFT/LoRA Integration**  
  Efficient parameter‑efficient fine‑tuning of large models using LoRA adapters.  
- **Chain‑of‑Thought SFT**  
  Trained with CoT (Complex_CoT) to improve reasoning transparency.  
- **4‑bit Quantized Inference**  
  Reduced GPU memory footprint via 4‑bit quantization.  
- **W&B Logging**  
  Real‑time training metrics and experiment tracking.  

---

## 🛠 Requirements

- Python ≥ 3.8  
- CUDA‑enabled GPU (≥16 GB VRAM recommended)  
- [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/quick-start#1-log-in) credentials  
- W&B account & API token  

---

### Python Dependencies

```text
torch
transformers
trl
unsloth
datasets
huggingface_hub
wandb
````

Install via:

```bash
pip install torch transformers trl unsloth datasets huggingface_hub wandb
```

---

## 🔧 Installation

1. **Clone the repo**

   ```bash
   git clone https://github.com/your-username/AI-Doctor-DeepSeek-R1-FineTuned.git
   cd AI-Doctor-DeepSeek-R1-FineTuned
   ```

2. **Set up environment variables**

   ```bash
   export HF_TOKEN="<your-hf-token>"
   export WANDB_API_TOKEN="<your-wandb-token>"
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

---

## 📚 Dataset

I have used the [FreedomIntelligence/medical-o1-reasoning-SFT](https://huggingface.co/datasets/FreedomIntelligence/medical-o1-reasoning-SFT) dataset (train\[0:500]) for proof‑of‑concept. It contains:

* `Question`
* `Complex_CoT` (chain‑of‑thought rationale)
* `Response`

---

## ⚙️ Training

Launch fine‑tuning via a Jupyter/Colab notebook or script:

```python
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset
import wandb

# 1. Login to HF & W&B
from huggingface_hub import login as hf_login
hf_login(token=os.getenv("HF_TOKEN"))
wandb.login(key=os.getenv("WANDB_API_TOKEN"))

# 2. Load model & tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    max_seq_length=2048,
    load_in_4bit=True,
    token=os.getenv("HF_TOKEN"),
)

# 3. Prepare dataset & prompts
ds = load_dataset("FreedomIntelligence/medical-o1-reasoning-SFT", split="train[:500]", trust_remote_code=True)
# (apply preprocess_input_data from the notebook to map `Question`, `Complex_CoT`, `Response` → formatted text)

# 4. Add LoRA adapters
model_lora = FastLanguageModel.get_peft_model(
    model=model,
    r=16,
    target_modules=[ "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj" ],
    lora_alpha=16,
    lora_dropout=0.0,
    bias="none",
)

# 5. Configure Trainer
trainer = SFTTrainer(
    model=model_lora,
    tokenizer=tokenizer,
    train_dataset=finetune_dataset,
    dataset_text_field="text",
    max_seq_length=2048,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=1,
        max_steps=60,
        learning_rate=2e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
        output_dir="outputs",
    ),
)

# 6. Train
trainer.train()
wandb.finish()
```

---

## 🚀 Usage

### Inference Demo

Run a quick test to see your fine‑tuned model in action:

```python
from unsloth import FastLanguageModel

# Switch to inference mode
FastLanguageModel.for_inference(model_lora)

prompt = """
You are a friendly and helpful medical assistant chatbot. Provide clear
and simple explanations about common health questions and symptoms.
Always remind users that your advice is general and does not replace
a visit to a doctor. Encourage users to seek professional medical help
for serious or urgent issues.

### Query:
What could be the cause of a persistent cough lasting more than two weeks?

### Answer:
<think>
"""

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model_lora.generate(
    input_ids=inputs.input_ids,
    attention_mask=inputs.attention_mask,
    max_new_tokens=300,
)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## 📊 Logging & Monitoring

* **Weights & Biases**
  Tracks training loss, learning rate, and gradients.
* **Colab Notebooks**
  Handy for GPU access & rapid prototyping.

---

## 🤝 Contributing

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/XYZ`)
3. Commit your changes (`git commit -m "Add XYZ"`)
4. Push to your branch (`git push origin feature/XYZ`)
5. Open a Pull Request

---

*Built with ❤️ using [unsloth](https://github.com/unslothai/unsloth) + [TRL](https://github.com/huggingface/trl).*

```
```
