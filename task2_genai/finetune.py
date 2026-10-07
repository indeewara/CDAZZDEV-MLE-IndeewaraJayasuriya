"""Task 2B - QLoRA fine-tuning of Qwen2.5-1.5B-Instruct on the financial news impact dataset.

Every hyperparameter is a named constant with its justification next to it. The notebook imports this
module, shows the same justifications as a table, and runs train() on a Colab T4 GPU.
"""
import json
import math
from pathlib import Path

import torch
from datasets import Dataset

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"

# ---- Model -----------------------------------------------------------------------------
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
# Why: instruction-tuned (already follows JSON instructions, so fine-tuning teaches the task, not chat);
# 1.5B fits a free T4 (16 GB) comfortably in 4-bit with room for batch 4 x 1024 tokens; Apache-2.0 and
# not gated (no HF licence click-through); a different model family from the teacher (OpenAI gpt-oss).

# ---- 4-bit quantization (QLoRA) ----------------------------------------------------------
LOAD_IN_4BIT = True                 # QLoRA: frozen base weights in 4-bit, LoRA adapters trained in 16-bit
BNB_4BIT_QUANT_TYPE = "nf4"         # NormalFloat4: optimal 4-bit grid for normally distributed weights (QLoRA paper)
BNB_4BIT_DOUBLE_QUANT = True        # also quantize the quantization constants: ~0.4 bits/param saved, no quality loss
COMPUTE_DTYPE = torch.float16       # T4 (Turing) has no bfloat16 support, so matmuls run in fp16
QUANT_STORAGE = torch.uint8         # how packed 4-bit weights are stored; only matters for multi-GPU (FSDP) sharding

# ---- LoRA ------------------------------------------------------------------------------
LORA_R = 16
# Why: the task is narrow (fixed taxonomy + JSON format + short rationale) and the data small (149 examples);
# r=8 risks underfitting 11-way classification plus generation, r=64 adds capacity we cannot fill and
# overfits faster. r=16 gives ~18M trainable params (~1.2% of 1.5B), printed when the model is built.
LORA_ALPHA = 32
# Why: alpha/r = 2 is the common scaling for r=16; it keeps the update magnitude stable so the learning
# rate below can be reasoned about independently of r.
LORA_DROPOUT = 0.05
# Why: light regularisation on a small dataset; higher values slow learning in a 30-step run.
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
# Why: all linear layers, attention and MLP. The QLoRA paper found adapting every linear layer is needed
# to match full fine-tuning; format and taxonomy knowledge is not confined to attention.
LORA_BIAS = "none"                  # biases stay frozen: standard for LoRA, keeps the merged model identical in shape
USE_RSLORA = False
# Why: rank-stabilised LoRA scales updates by alpha/sqrt(r) instead of alpha/r; it helps at high ranks (64+).
# At r=16 the standard scaling is well-behaved and keeps the setup comparable to the QLoRA paper.
USE_DORA = False
# Why: DoRA adds a learned magnitude vector per layer - more memory and ~2x slower steps on a T4 - and is aimed
# at closing the gap to full fine-tuning on harder tasks; plain LoRA is enough for format + taxonomy learning.
INIT_LORA_WEIGHTS = True
# Why: A random (Kaiming-uniform), B zero, so the adapter starts as an exact no-op and training begins from the
# base model's behaviour (alternatives such as PiSSA/LoftQ initialise from the weights and change the start point).
LORA_LAYERS = None                  # None = all 28 transformer layers: format and labelling behaviour spans the depth
MODULES_TO_SAVE = None              # embeddings and LM head stay frozen: no new tokens are added, the vocabulary is unchanged

# ---- Training --------------------------------------------------------------------------
NUM_EPOCHS = 3
# Why: 149 examples / effective batch 16 = 10 optimizer steps per epoch. 1 epoch (10 steps) is too few to
# learn an 11-type taxonomy; beyond 3 the model starts memorising the teacher's phrasing. Validation loss
# per epoch is the check - it must still be falling at epoch 3.
LEARNING_RATE = 2e-4
# Why: the QLoRA paper's rate for small models; LoRA adapters start at zero (B=0), so a higher rate than
# full fine-tuning (~1e-5) is needed to move them meaningfully in only 30 steps.
LR_SCHEDULER = "cosine"
# Why: decays smoothly to ~0 by the last step, so the final epoch makes small, careful updates - this
# limits late overfitting on a small dataset better than a constant rate.
WARMUP_RATIO = 0.1
# Why: ~3 warm-up steps so Adam's moment estimates settle before the full learning rate is applied.
PER_DEVICE_TRAIN_BATCH_SIZE = 4
# Why: largest that fits a T4 at ~800 tokens per sequence with 4-bit weights and gradient checkpointing.
GRADIENT_ACCUMULATION_STEPS = 4
# Why: effective batch 4 x 4 = 16 - smoother gradients than 4, while keeping 10 steps per epoch.
PER_DEVICE_EVAL_BATCH_SIZE = 8      # no gradients at eval, so a larger batch fits; does not affect results
MAX_LENGTH = 1024
# Why: the longest training sequence is 810 tokens (measured with Qwen's tokenizer in Task 2A), so 1024
# truncates nothing and leaves headroom; anything larger only wastes memory.
OPTIMIZER = "paged_adamw_8bit"
# Why: QLoRA's paged optimizer moves optimizer state to CPU RAM on memory spikes instead of crashing with
# OOM; 8-bit states cut optimizer memory ~4x.
ADAM_BETA1, ADAM_BETA2, ADAM_EPSILON = 0.9, 0.999, 1e-8
# Why: the standard AdamW values used by the QLoRA paper. Adam's bias correction makes beta2=0.999 behave
# sensibly even over only 30 steps, and nothing in a short LoRA run gives a reason to deviate.
LABEL_SMOOTHING = 0.0
# Why: the targets are exact JSON labels from a fixed set; smoothing would reward spreading probability onto
# other tokens, which works against producing the one valid label and valid JSON.
NEFTUNE_NOISE_ALPHA = None
# Why: NEFTune adds noise to embeddings to make open-ended chat answers more varied. This task needs exact,
# consistent structured output, so the noise would work against format fidelity. Off.
TORCH_COMPILE = False               # compilation overhead exceeds its gain over 30 steps, and is fragile with 4-bit layers
DATALOADER_DROP_LAST = False        # 149 is not a multiple of 4: keep the last short batch so every example is seen each epoch
WEIGHT_DECAY = 0.0
# Why: LoRA dropout already regularises, and decaying a zero-initialised adapter over 30 steps mostly
# fights the learning signal.
MAX_GRAD_NORM = 0.3                 # QLoRA paper value: clips the occasional large gradient from 4-bit noise
GRADIENT_CHECKPOINTING = True       # recompute activations in backward: ~40% less memory for ~20% more time
COMPLETION_ONLY_LOSS = True
# Why: loss only on the assistant's JSON answer. The ~600-token system prompt is identical in every example;
# counting it would make the loss mostly "memorise the prompt" and hide whether the task is being learned.
PACKING = False                     # examples stay separate: packing would mix unrelated news items in one sequence
SEED = 42                           # reproducible shuffling, LoRA init and dropout
EVAL_STRATEGY = SAVE_STRATEGY = LOGGING_STRATEGY = "epoch"
# Why: the assessment asks for train and validation loss per epoch; saving per epoch lets us keep the best.
LOAD_BEST_MODEL_AT_END = True       # restore the epoch with the lowest validation loss before saving
METRIC_FOR_BEST_MODEL = "eval_loss"
SAVE_TOTAL_LIMIT = 2                # keep disk use small on Colab


HYPERPARAMETERS = {  # single source for the notebook's justification table
    "base model": BASE_MODEL, "quantization": f"4-bit {BNB_4BIT_QUANT_TYPE.upper()}, double quant, fp16 compute",
    "LoRA r": LORA_R, "LoRA alpha": LORA_ALPHA, "LoRA dropout": LORA_DROPOUT,
    "target modules": ", ".join(LORA_TARGET_MODULES), "learning rate": LEARNING_RATE,
    "LR scheduler": f"{LR_SCHEDULER}, warmup ratio {WARMUP_RATIO}", "epochs": NUM_EPOCHS,
    "batch size": PER_DEVICE_TRAIN_BATCH_SIZE, "gradient accumulation": GRADIENT_ACCUMULATION_STEPS,
    "effective batch": PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS, "max sequence length": MAX_LENGTH,
    "optimizer": OPTIMIZER, "weight decay": WEIGHT_DECAY, "max grad norm": MAX_GRAD_NORM,
    "gradient checkpointing": GRADIENT_CHECKPOINTING, "loss on": "assistant tokens only", "packing": PACKING,
    "seed": SEED, "rsLoRA": USE_RSLORA, "DoRA": USE_DORA, "LoRA init": "A random, B zero (no-op start)",
    "LoRA layers": "all 28", "modules to save": "none (embeddings/LM head frozen)",
    "Adam betas / epsilon": f"{ADAM_BETA1}, {ADAM_BETA2} / {ADAM_EPSILON}", "label smoothing": LABEL_SMOOTHING,
    "NEFTune noise": "off", "torch.compile": TORCH_COMPILE, "drop last batch": DATALOADER_DROP_LAST,
    "4-bit storage dtype": "uint8",
}


# ======================================================================================
# Data
# ======================================================================================
def load_split(name: str, data_dir: Path = DATA_DIR) -> Dataset:
    """Chat-format JSONL -> TRL's conversational prompt/completion format: prompt = system + user turns,
    completion = assistant turn. With this format SFTTrainer masks the prompt out of the loss."""
    rows = [json.loads(line) for line in (data_dir / f"{name}.jsonl").read_text().splitlines() if line.strip()]
    return Dataset.from_list([{"prompt": r["messages"][:2], "completion": r["messages"][2:]} for r in rows])


# ======================================================================================
# Model
# ======================================================================================
def bnb_config():
    from transformers import BitsAndBytesConfig
    return BitsAndBytesConfig(load_in_4bit=LOAD_IN_4BIT, bnb_4bit_quant_type=BNB_4BIT_QUANT_TYPE,
                              bnb_4bit_use_double_quant=BNB_4BIT_DOUBLE_QUANT, bnb_4bit_compute_dtype=COMPUTE_DTYPE,
                              bnb_4bit_quant_storage=QUANT_STORAGE)


def lora_config():
    from peft import LoraConfig
    return LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT, bias=LORA_BIAS,
                      target_modules=LORA_TARGET_MODULES, task_type="CAUSAL_LM", use_rslora=USE_RSLORA,
                      use_dora=USE_DORA, init_lora_weights=INIT_LORA_WEIGHTS, layers_to_transform=LORA_LAYERS,
                      modules_to_save=MODULES_TO_SAVE)


def load_base(model_name: str = BASE_MODEL, quantize: bool = True):
    """Base model + tokenizer. quantize=False is only for the CPU smoke test (bitsandbytes needs CUDA)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    kwargs = {"quantization_config": bnb_config(), "device_map": "auto", "dtype": COMPUTE_DTYPE} if quantize else {}
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.config.use_cache = False  # the KV cache is useless in training and conflicts with gradient checkpointing
    return model, tokenizer


def sft_config(output_dir: str, **overrides):
    from trl import SFTConfig
    args = dict(
        output_dir=output_dir, num_train_epochs=NUM_EPOCHS, learning_rate=LEARNING_RATE,
        lr_scheduler_type=LR_SCHEDULER, warmup_steps=WARMUP_RATIO,  # transformers 5: a float < 1 is a ratio of total steps
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE, per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS, max_length=MAX_LENGTH, optim=OPTIMIZER,
        weight_decay=WEIGHT_DECAY, max_grad_norm=MAX_GRAD_NORM, gradient_checkpointing=GRADIENT_CHECKPOINTING,
        adam_beta1=ADAM_BETA1, adam_beta2=ADAM_BETA2, adam_epsilon=ADAM_EPSILON,
        label_smoothing_factor=LABEL_SMOOTHING, neftune_noise_alpha=NEFTUNE_NOISE_ALPHA,
        torch_compile=TORCH_COMPILE, dataloader_drop_last=DATALOADER_DROP_LAST,
        gradient_checkpointing_kwargs={"use_reentrant": False},  # non-reentrant is required with PEFT adapters
        completion_only_loss=COMPLETION_ONLY_LOSS, packing=PACKING, seed=SEED,
        eval_strategy=EVAL_STRATEGY, save_strategy=SAVE_STRATEGY, logging_strategy=LOGGING_STRATEGY,
        load_best_model_at_end=LOAD_BEST_MODEL_AT_END, metric_for_best_model=METRIC_FOR_BEST_MODEL,
        greater_is_better=False, save_total_limit=SAVE_TOTAL_LIMIT,
        fp16=torch.cuda.is_available(), bf16=False,  # fp16 mixed precision on the T4; plain fp32 on CPU
        report_to="none",  # losses are logged manually below (table + plot); W&B is optional, not required
    )
    args.update(overrides)
    return SFTConfig(**args)


def build_trainer(model, tokenizer, train_ds, val_ds, output_dir: str, **config_overrides):
    from trl import SFTTrainer
    trainer = SFTTrainer(model=model, args=sft_config(output_dir, **config_overrides), train_dataset=train_ds,
                         eval_dataset=val_ds, processing_class=tokenizer, peft_config=lora_config())
    cast_trainable_to_fp32(trainer.model)
    return trainer


def cast_trainable_to_fp32(model) -> int:
    """fp16 mixed precision keeps fp32 'master' weights: the GradScaler unscales gradients and only supports
    fp32 there. Qwen2.5's checkpoint is stored in bfloat16, and adapters created next to bf16 layers can
    inherit it, which crashes the first optimizer step on a T4 ('..._unscale_cuda not implemented for
    BFloat16'). The trainable parameters are only the LoRA adapters (~18M), so fp32 costs ~70 MB."""
    n = 0
    for p in model.parameters():
        if p.requires_grad and p.dtype != torch.float32:
            p.data = p.data.float()
            n += 1
    return n


def trainable_dtypes(model) -> dict[str, int]:
    """{dtype: number of trainable tensors} - should be {'torch.float32': ...} only."""
    counts: dict[str, int] = {}
    for p in model.parameters():
        if p.requires_grad:
            counts[str(p.dtype)] = counts.get(str(p.dtype), 0) + 1
    return counts


# ======================================================================================
# Loss logging
# ======================================================================================
def epoch_losses(log_history: list[dict]) -> list[dict]:
    """Per-epoch train and validation loss from Trainer.state.log_history."""
    rows = {}
    for entry in log_history:
        if "epoch" not in entry:
            continue
        ep = round(entry["epoch"])
        if "loss" in entry:
            rows.setdefault(ep, {})["train_loss"] = entry["loss"]
        if "eval_loss" in entry:
            rows.setdefault(ep, {})["val_loss"] = entry["eval_loss"]
    return [{"epoch": ep, **vals} for ep, vals in sorted(rows.items())]


def val_loss_decreasing(rows: list[dict]) -> bool:
    vals = [r["val_loss"] for r in rows if "val_loss" in r and not math.isnan(r["val_loss"])]
    return len(vals) >= 2 and all(b < a for a, b in zip(vals, vals[1:]))


# ======================================================================================
# Merge
# ======================================================================================
def merge_adapter(adapter_dir: str, merged_dir: str, base_model: str = BASE_MODEL, dtype=torch.float16):
    """Reload the base in 16-bit (merging into 4-bit weights would bake in quantization error), apply the
    trained LoRA adapter, fold it into the weights with merge_and_unload(), and save a standalone model."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    base = AutoModelForCausalLM.from_pretrained(base_model, dtype=dtype)
    merged = PeftModel.from_pretrained(base, adapter_dir).merge_and_unload()
    merged.save_pretrained(merged_dir)
    AutoTokenizer.from_pretrained(adapter_dir).save_pretrained(merged_dir)
    return merged
