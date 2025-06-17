import torch
import torch.nn as nn
import math
import contextlib
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM
)
from datasets import load_dataset
from transformers.models.llama.modeling_llama import LlamaAttention

from typing import Callable, Optional, Tuple, Union
from transformers.cache_utils import Cache, DynamicCache, EncoderDecoderCache, StaticCache


# ----------------------------------------
# 1. Custom Attention Forward Implementation
# ----------------------------------------

def sub_matmul(A, B, k: int):
    # A is a .. X m X n tensor
    # B is a .. X p X n tensor
    # k integer, assume k < 8
    # Assume 2^k divides n
    if A.size(-1) % (2 ** k) != 0:
        raise ValueError("A.size(-1) must be divisible by 2^k")

    currentA = A
    currentB = B
    current_proj_dim = A.size(-1) // 2

    for t in range(k):
        rand_signs = torch.randint(0, 2, (current_proj_dim,), device=A.device, dtype=A.dtype) * 2 - 1
        A1, A2 = currentA.split(current_proj_dim, dim=-1)
        nextA = A1 + A2 * rand_signs
        currentA = nextA

        B1, B2 = currentB.split(current_proj_dim, dim=-1)
        nextB = B1 + B2 * rand_signs
        currentB = nextB
        current_proj_dim = current_proj_dim // 2

    C = torch.matmul(currentA, currentB.transpose(-1, -2))
    return C

# Unit test for sub_matmul
def test_sub_matmul():
    torch.manual_seed(42)
    # Test parameters
    batch = 2
    m = 3
    p = 4
    n = 8  # n must be divisible by 2^k
    k = 2  # 2^2 = 4 divides 8

    # Create random tensors
    A = torch.randn(batch, m, n)
    B = torch.randn(batch, p, n)

    # Check output shape
    C = sub_matmul(A, B, k)
    assert C.shape == (batch, m, p), f"Output shape mismatch: got {C.shape}, expected {(batch, m, p)}"

    # Check error is raised for invalid n
    try:
        A_bad = torch.randn(batch, m, 10)
        B_bad = torch.randn(batch, p, 10)
        sub_matmul(A_bad, B_bad, k)
        assert False, "Expected ValueError for invalid n"
    except ValueError:
        pass

    # Check that k=0 gives standard matmul
    A0 = torch.randn(batch, m, n)
    B0 = torch.randn(batch, p, n)
    C0 = sub_matmul(A0, B0, 0)
    C0_ref = torch.matmul(A0, B0.transpose(-1, -2))
    assert torch.allclose(C0, C0_ref, atol=1e-5), "k=0 should match standard matmul"

    # Generate two moderate sized random matrices
    torch.manual_seed(123)
    batch = 1
    m = 16
    p = 16
    n = 64  # n must be divisible by 2^k
    k = 3   # 2^3 = 8 divides 64

    A = torch.randn(batch, m, n)
    B = torch.randn(batch, p, n)

    # Run sub_matmul
    C_sub = sub_matmul(A, B, k)
    # Run standard matmul
    C_ref = torch.matmul(A, B.transpose(-1, -2))

    # Compute total squared difference
    average_sq_value = torch.sum(C_ref ** 2).item()/(batch*m*p)
    average_sq_diff = torch.sum((C_sub - C_ref) ** 2).item()/(batch*m * p)
    print(f"Average squared magnitude of the matrix multiplication: {average_sq_value:.6f}")
    print(f"Average squared difference between sub_matmul and standard matmul: {average_sq_diff:.6f}")
    print(f"Relative squared difference between sub_matmul and standard matmul: {(average_sq_diff/average_sq_value):6f}")


    print("test_sub_matmul passed.")

if __name__ == "__main__":
    test_sub_matmul()


     



def randomized_matmul(query, key):
    # query is .. X Q_seq_len X hidden_dim
    # key is .. X K_seq_len X hidden_dim
    return sub_matmul(query,key,1)

    
    # hid_dim = query.size(-1)
    # if hid_dim % 2 == 0:
    #     proj_dim = hid_dim // 2
    # else:
    #     raise ValueError("query.size(-1) must be even")

    # rand_signs = torch.randint(0, 2, (proj_dim,), device=query.device, dtype=query.dtype) * 2 - 1

    # key1, key2 = key.split(proj_dim, dim=-1)
    # proj_key = key1 + key2 * rand_signs
   
    # query1, query2 = query.split(proj_dim, dim=-1)
    # proj_query = query1 + query2 * rand_signs

    # attn_weights = torch.matmul(proj_query, proj_key.transpose(-1, -2))
    # return attn_weights


def my_eager_attention_forward(module, query, key, value, attention_mask, head_mask=None, **kwargs):
    
    # This is the original matrix multiplication
    # ----------------------------------------------------------------
    #     attn_weights = torch.matmul(query, key)  
    # ----------------------------------------------------------------
    # Replacing it with this custom code.
    # 
    attn_weights = randomized_matmul(query, key)

    # query is batchsize X numHeads X Q_seq_len X hidden_dimension
    # key is batchsize X numHeads X k_seq_len X hidden_dimension
    
    if module.scale_attn_weights:
        attn_weights = attn_weights / torch.full(
            [], value.size(-1) ** 0.5, dtype=attn_weights.dtype, device=attn_weights.device
        )

    # Layer-wise attention scaling
    if module.scale_attn_by_inverse_layer_idx:
        attn_weights = attn_weights / float(module.layer_idx + 1)

    if not module.is_cross_attention:
        # if only "normal" attention layer implements causal mask
        query_length, key_length = query.size(-2), key.size(-2)
        causal_mask = module.bias[:, :, key_length - query_length : key_length, :key_length]
        mask_value = torch.finfo(attn_weights.dtype).min
        # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
        # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
        mask_value = torch.full([], mask_value, dtype=attn_weights.dtype, device=attn_weights.device)
        attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)

    if attention_mask is not None:
        # Apply the attention mask
        causal_mask = attention_mask[:, :, :, : key.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1)

    # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op otherwise
    attn_weights = attn_weights.type(value.dtype)
    attn_weights = module.attn_dropout(attn_weights)

    # Mask heads if we want to
    if head_mask is not None:
        attn_weights = attn_weights * head_mask

    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2)

    return attn_output, attn_weights



# Copied forward from GPT2Attention, exactly one change in this whole function.
# Replace eager_attention by my_eager_attention

def my_custom_attention_forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
        is_cross_attention = encoder_hidden_states is not None
        if is_cross_attention:
            if not hasattr(self, "q_attn"):
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have to be defined. "
                    "Please make sure to instantiate class with `GPT2Attention(..., is_cross_attention=True)`."
                )

            query_states = self.q_attn(hidden_states)
            key_states, value_states = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
            attention_mask = encoder_attention_mask
        else:
            query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)

        shape_q = (*query_states.shape[:-1], -1, self.head_dim)
        shape_kv = (*key_states.shape[:-1], -1, self.head_dim)

        query_states = query_states.view(shape_q).transpose(1, 2)
        key_states = key_states.view(shape_kv).transpose(1, 2)
        value_states = value_states.view(shape_kv).transpose(1, 2)

        if past_key_value is not None:
            if isinstance(past_key_value, EncoderDecoderCache):
                if is_cross_attention:
                    past_key_value = past_key_value.cross_attention_cache
                else:
                    past_key_value = past_key_value.self_attention_cache
            cache_kwargs = {"cache_position": cache_position}
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs=cache_kwargs
            )

        is_causal = attention_mask is None and query_states.shape[-2] > 1 and not is_cross_attention

        using_eager = self.config._attn_implementation == "eager"
        attention_interface: Callable = my_eager_attention_forward
        #if self.config._attn_implementation != "eager":
        #    if self.config._attn_implementation == "sdpa" and (output_attentions or head_mask is not None):
        #        using_eager = True
        #        logger.warning_once(
        #            "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
        #            'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
        #        )
        #    else:
                # Attention functions are consistent with previous equivalent attention classes, however they do not support some options
                # (e.g. layer scaling, head mask) that eager supports. These implementations are thus equivalent to previous code, but
                # not necessarily to eager (if mentioned options are provided).
        #        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        #if using_eager and self.reorder_and_upcast_attn:
        #    attn_output, attn_weights = self._upcast_and_reordered_attn(
        #        query_states, key_states, value_states, attention_mask, head_mask
        #    )
        #else:
        attn_output, attn_weights = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                head_mask=head_mask,
                dropout=self.attn_dropout.p if self.training else 0.0,
                is_causal=is_causal,
                **kwargs,
            )

        attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        return attn_output, attn_weights


# ----------------------------------------
# 2. Context Manager to Temporarily Swap Attention
# ----------------------------------------
@contextlib.contextmanager
def use_custom_attention_gpt(model, my_custom_forward_fn):
    """
    Context manager to temporarily swap the attention forward method for DistillGPT models.
    This will patch all attention modules in the model.
    """
    backups = []
    # Import TinyGPTAttention only inside the function to avoid import errors if not available
    from transformers.models.gpt2.modeling_gpt2 import GPT2Attention

    for module in model.modules():
        if isinstance(module, GPT2Attention):
            module.original_forward = module.forward
            module.forward = my_custom_forward_fn.__get__(module, type(module))
            backups.append(module)
    try:
        yield
    finally:
        for module in backups:
            module.forward = module.original_forward
            del module.original_forward


# ----------------------------------------
# 3. Load Model and Tokenizer
# ----------------------------------------
# Expects a Decoder only model
def load_model(model_name):
    config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return model, tokenizer, device

def load_tinyllama():
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
     # Step 1: Load config and patch it
    config = AutoConfig.from_pretrained(model_name)
    if not hasattr(config, "parallelization") or config.parallelization is None:
        config.parallelization = "none"  # or "" or some safe default

    # Step 2: Load model using patched config
    model = AutoModelForCausalLM.from_pretrained(model_name, config=config)

 
    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return model, tokenizer, device

# ----------------------------------------
#  Load Data
# ----------------------------------------

def load_data(tokenizer):
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    text = "\n\n".join(dataset["text"])
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings["input_ids"][0]
    return input_ids 


# ---------------------------------------
# Run the model
# ----------------------------------------

def run_Model(model, tokenizer, prompt, use_custom=False):
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    context = use_custom_attention_gpt(model, my_custom_attention_forward) if use_custom else contextlib.nullcontext()
    with context:
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=50)
        #    output_ids = model.generate(**inputs, max_new_tokens=50,
        #                               do_sample=True,        # Turn on sampling
        #                               top_k=50,              # Only sample from top 50 tokens
        #                              temperature=0.9)        # Smooth out logits
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=False)
    print(output_text)


# ----------------------------------------
# 4. Perplexity Evaluation Function
# ----------------------------------------


def evaluate_perplexity(model, device, input_ids,use_custom=False):
    max_length = 512
    stride = 512
    n_chunks = (input_ids.size(0) - 1) // max_length
    losses = []

    context = use_custom_attention(model, my_custom_attention_forward) if use_custom else contextlib.nullcontext()

    with context:
        with torch.no_grad():
            for i in range(n_chunks):
                start = i * stride
                end = start + max_length
                input_chunk = input_ids[start:end].unsqueeze(0).to(device)
                outputs = model(input_chunk, labels=input_chunk)
                losses.append(outputs.loss.item())

    avg_loss = sum(losses) / len(losses)
    perplexity = math.exp(avg_loss)
    return avg_loss, perplexity

# ----------------------------------------
# 5. Main Comparison Script
# ----------------------------------------

def main():

    model_name = "distilbert/distilgpt2"
    model, tokenizer, device = load_model(model_name)

    prompt = "Let me tell you a story of a cat."
    print("Generation with Native Inference:")
    run_Model(model, tokenizer,prompt,use_custom=False)
    print("Generation with approximate inference:")
    run_Model(model, tokenizer,prompt,use_custom=True)

#    input_ids = load_data(tokenizer)

#    print("\n🔹 Evaluating Perplexity Score under native inference...")
#    loss_std, ppl_std = evaluate_perplexity(model, device, input_ids, False)
#    print(f"Standard Loss: {loss_std:.4f}, Perplexity: {ppl_std:.2f}")

#    print("\n🔹 Evaluating TinyLlama with custom attention...")
#    loss_custom, ppl_custom = evaluate_perplexity(model, tokenizer, device, input_ids, use_custom=True)
#    print(f"Custom Loss:   {loss_custom:.4f}, Perplexity: {ppl_custom:.2f}")

#if __name__ == "__main__":
#    main()
