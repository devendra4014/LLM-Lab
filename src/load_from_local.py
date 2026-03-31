from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import GPT2LMHeadModel

model_name = "gpt2"
download_dir = r"E:\code\DL\LLM-Lab\.model"
# local_dir = r"E:\code\DL\LLM-Lab\GPTmodel"

# Download model and tokenizer into your folder
model = GPT2LMHeadModel.from_pretrained("gpt2", cache_dir=download_dir)
# model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=download_dir)
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=download_dir)

print("Model loaded successfully from local folder!")

config = model.config

print("Model config :: ")
"""
GPT2Config {
  "activation_function": "gelu_new",
  "architectures": [
    "GPT2LMHeadModel"
  ],
  "attn_pdrop": 0.1,
  "bos_token_id": 50256,
  "dtype": "float32",
  "embd_pdrop": 0.1,
  "eos_token_id": 50256,
  "initializer_range": 0.02,
  "layer_norm_epsilon": 1e-05,
  "model_type": "gpt2",
  "n_ctx": 1024,
  "n_embd": 768,
  "n_head": 12,
  "n_inner": null,
  "n_layer": 12,
  "n_positions": 1024,
  "reorder_and_upcast_attn": false,
  "resid_pdrop": 0.1,
  "scale_attn_by_inverse_layer_idx": false,
  "scale_attn_weights": true,
  "summary_activation": null,
  "summary_first_dropout": 0.1,
  "summary_proj_to_labels": true,
  "summary_type": "cls_index",
  "summary_use_proj": true,
  "task_specific_params": {
    "text-generation": {
      "do_sample": true,
      "max_length": 50
    }
  },
  "transformers_version": "4.57.3",
  "use_cache": true,
  "vocab_size": 50257
}
"""
print(config)
print("==============================")
# Step 4: Print state dict keys and shapes
for name, param in model.state_dict().items():
    print(f"{name}: {param.shape}")
