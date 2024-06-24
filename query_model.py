import torch
import transformers
import pyreft

# Define the prompt template
prompt_no_input_template = """<s>[INST] <<SYS>>
You are a helpful assistant.
<</SYS>>

%s [/INST]
"""

# Load the model
model_name_or_path = "meta-llama/Meta-Llama-3-8B"
device = "cuda" if torch.cuda.is_available() else "cpu"
model = transformers.AutoModelForCausalLM.from_pretrained(
    model_name_or_path, torch_dtype=torch.bfloat16, device_map=device)

# Load the tokenizer
tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_name_or_path, model_max_length=2048, 
    padding_side="right", use_fast=False)
tokenizer.pad_token = tokenizer.unk_token

# Set up the ReFT config
reft_config = pyreft.ReftConfig(representations={
    "layer": 15, "component": "block_output",
    "low_rank_dimension": 4,
    "intervention": pyreft.LoreftIntervention(embed_dim=model.config.hidden_size,
    low_rank_dimension=4)})
reft_model = pyreft.get_reft_model(model, reft_config)
reft_model.set_device(device)

# Function to perform inference
def perform_inference(instruction):
    prompt = prompt_no_input_template % instruction
    prompt = tokenizer(prompt, return_tensors="pt").to(device)
    
    base_unit_location = prompt["input_ids"].shape[-1] - 1  # last position
    _, reft_response = reft_model.generate(
        prompt, unit_locations={"sources->base": (None, [[[base_unit_location]]])},
        intervene_on_prompt=True, max_new_tokens=512, do_sample=True, 
        eos_token_id=tokenizer.eos_token_id, early_stopping=True
    )
    return tokenizer.decode(reft_response[0], skip_special_tokens=True)

# Example instruction
instruction = "Which dog breed do people think is cuter, poodle or doodle?"
response = perform_inference(instruction)
print(response)
