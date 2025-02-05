# generate.py
import torch
import torch.nn.functional as F

def generate_text_temperature(model, tokenizer, prompt, max_length=50, temperature=1.0, device="cpu"):
    """
    Generates text using temperature sampling.
    """
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    generated = input_ids
    
    for _ in range(max_length):
        if generated.size(1) > model.max_seq_len:
            input_ids = generated[:, -model.max_seq_len:]  # Trim input to model's max sequence length
        else:
            input_ids = generated

        logits, _ = model(input_ids)  # Get logits from model
        next_token_logits = logits[:, -1, :] / temperature  # Apply temperature scaling
        probabilities = F.softmax(next_token_logits, dim=-1)
        
        next_token = torch.multinomial(probabilities, num_samples=1)  # Sample next token
        generated = torch.cat((generated, next_token), dim=1)
        
        if next_token.item() == tokenizer.eos_token_id:  # Stop at EOS token
            break
    
    return tokenizer.decode(generated[0].tolist())

def generate_text_beam(model, tokenizer, prompt, max_length=50, beam_width=3, device="cpu"):
    """
    Generates text using beam search.
    """
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    beams = [(input_ids, 0)]  # (sequence, cumulative log probability)
    
    for _ in range(max_length):
        new_beams = []
        
        for seq, score in beams:
            if seq.size(1) > model.max_seq_len:
                seq_input = seq[:, -model.max_seq_len:]  # Ensure sequence fits within model's limits
            else:
                seq_input = seq
            
            logits, _ = model(seq_input)  # Get logits
            next_token_logits = logits[:, -1, :]
            log_probs = F.log_softmax(next_token_logits, dim=-1)
            
            topk_log_probs, topk_indices = torch.topk(log_probs, beam_width, dim=-1)
            
            for i in range(beam_width):
                next_token = topk_indices[0, i].unsqueeze(0).unsqueeze(0)  # Reshape for concatenation
                new_seq = torch.cat([seq, next_token], dim=1)
                new_score = score + topk_log_probs[0, i].item()
                new_beams.append((new_seq, new_score))
        
        new_beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]  # Keep top-k beams
        beams = new_beams
        
        if all(seq[0, -1].item() == tokenizer.eos_token_id for seq, _ in beams):  # Stop if all beams hit EOS
            break

    best_seq = beams[0][0]  # Select the best sequence
    return tokenizer.decode(best_seq[0].tolist())

def evaluate_generation(model, tokenizer, test_dataset, device="cpu", num_prompts=10):
    """
    Evaluates text generation using sample prompts.
    """
    import random
    prompts = []
    indices = random.sample(range(len(test_dataset)), num_prompts)
    
    for idx in indices:
        example = test_dataset[idx]["input_ids"]
        prompt_text = tokenizer.decode(example.tolist()[:50])  # Use first 50 tokens as prompt
        prompts.append(prompt_text)

    print("=== Temperature-based Generation ===")
    for prompt in prompts:
        print("Prompt:", prompt)
        generated = generate_text_temperature(model, tokenizer, prompt, device=device)
        print("Generated:", generated)
        print("-" * 50)

    print("\n=== Beam Search-based Generation ===")
    for prompt in prompts:
        print("Prompt:", prompt)
        generated = generate_text_beam(model, tokenizer, prompt, device=device)
        print("Generated:", generated)
        print("-" * 50)
