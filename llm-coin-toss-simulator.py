from transformers import AutoModelForCausalLM, AutoTokenizer, utils
import torch.nn.functional as F
import torch
import tqdm
import pickle
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

"""
Set up these values before running the script
"""
num_of_coin_tosses = 100
model_name = "meta-llama/Llama-3.2-3B-Instruct"
prompts = {
    "system": """You are a helpful assistant""",
    "simple": """What is the outcome of a coin toss given that it is a biased coin with 70% bias towards Heads? Reply with a 'H' for Heads and 'T' for Tails and no other characters.""",
    "simple_emotion": """What is the outcome of a coin toss given that it is a biased coin with 70% bias towards Heads? Your answer greatly affects my chances of getting a job, please do your best and help me. Reply with a 'H' for Heads and 'T' for Tails and no other characters.""",
    "simple_next_toss": """What is the outcome if the coin is tossed again? Reply with a 'H' for Heads and 'T' for Tails and no other characters.""",
    "simulate": """You are given a biased coin with 70% bias towards Heads. Simulate a coin toss. What is the outcome? Reply with a 'H' for Heads and 'T' for Tails and no other characters.""",
    "simulate_emotion": """You are given a biased coin with 70% bias towards Heads. Simulate a coin toss. What is the outcome? Your answer greatly affects my chances of getting a job, please do your best and help me. Reply with a 'H' for Heads and 'T' for Tails and no other characters.""",
    "simulate_next_toss": """Simulate a coin toss again. What is the outcome? Reply with a 'H' for Heads and 'T' for Tails and no other characters.""",
}
"""
"""

# Load model & tokenizer
bf_16_available = utils.import_utils.is_torch_bf16_gpu_available()
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.bfloat16 if bf_16_available else torch.float16,
)
model.eval()

for prompt_type in ["simple", "simple_emotion", "simulate", "simulate_emotion"]:

    if "simple" in prompt_type:
        next_prompt_type = "simple_next_toss"
    elif "simulate" in prompt_type:
        next_prompt_type = "simulate_next_toss"

    # token ids for characters H and T
    H_T_ids = {
        "H": tokenizer.encode("H", add_special_tokens=False)[0],
        "T": tokenizer.encode("T", add_special_tokens=False)[0],
    }

    # Prepare input
    messages = [
        {"role": "system", "content": prompts["system"]},
        {"role": "user", "content": prompts[prompt_type]},
    ]
    assistant_response = ""

    probabilities_H = []
    probabilities_T = []
    outcomes = []

    pbar = tqdm.tqdm(total=num_of_coin_tosses, desc="Num of Coin Tosses")

    while len(outcomes) < num_of_coin_tosses:

        if len(assistant_response) > 0:
            messages.append({"role": "assistant", "content": assistant_response})
            messages.append({"role": "user", "content": prompts[next_prompt_type]})

        # TODO: can be made more efficient without having to tokenize all chat history each time
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        # print("Inputs: ", prompt)

        # Generate response - keep rerunning inference until output constraints are met
        # Note that do_sample is True in model.generate, so each time it is called, the outcome may be different
        inference_output_token = ""
        while inference_output_token not in ["H", "T"]:
            with torch.no_grad():
                # TODO: can be more efficient without having to do inference on prev chat history each time by maintaining a KV cache
                output = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    return_dict_in_generate=True,
                    output_scores=True,
                    do_sample=True,
                    temperature=0.6,
                    top_p=0.9,
                )
            inference_output_token = tokenizer.decode(
                output.sequences[0][inputs["input_ids"].shape[-1] :],
                skip_special_tokens=True,
            )

        # Decode response
        assistant_response = inference_output_token
        # print(f"LLM Response: {assistant_response}")

        # Get the generated token IDs (excluding the prompt)
        generated_ids = output.sequences[0][
            inputs["input_ids"].shape[-1] :
        ]  # shape: [num_generated_tokens]

        for i, scores in enumerate(output.scores):
            probs_i = F.softmax(scores[0], dim=-1)  # shape: [vocab_size]
            token_id = generated_ids[i].item()
            prob_H = probs_i[H_T_ids["H"]].item()
            prob_T = probs_i[H_T_ids["T"]].item()
            break

        outcomes.append(assistant_response)
        probabilities_H.append(prob_H)
        probabilities_T.append(prob_T)
        pbar.update(1)

    results = [outcomes, probabilities_H, probabilities_T]
    os.system("mkdir -p ./results/")
    with open("./results/{}_results.pkl".format(prompt_type), "wb") as f:
        pickle.dump(results, f)
