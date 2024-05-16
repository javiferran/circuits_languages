import torch
import math
from typing import List, Union, Optional
from jaxtyping import Float
from fancy_einsum import einsum
import einops
from transformer_lens import HookedTransformer, HookedTransformerConfig, FactoredMatrix, ActivationCache

def read_files(model, file_name: str) -> list:
    '''
    Read file of tuples and get only words associated with
    a single token by the model's tokenizr.

    Args:
        model: The model used for tokenization.
        file_name: The name of the file to read.

    Returns:
        A list of valid tuples containing words associated with a single token.
    '''
    f = open(file_name, "r")
    list_str = f.read()
    words_list =list_str.split('\n')[1:]
    clean_pair_list = [words_pair[words_pair.find('.')+2:] for words_pair in words_list]
    clean_pair_list_tuples = [(words_pair.split('/')[0], words_pair.split('/')[1]) for words_pair in clean_pair_list]
    # Get only valid tuples
    valid_tuples = []
    for pair_tuple in clean_pair_list_tuples:
        toks_sing = model.to_tokens(f' {pair_tuple[0]}')[0].shape[0]
        toks_plur = model.to_tokens(f' {pair_tuple[1]}')[0].shape[0]
        if toks_sing==2 and toks_plur==2:
            valid_tuples.append(pair_tuple)
    return valid_tuples


def get_batched_dataset(model, base_list, src_list, answers, batch_size=20):
    """
    Creates a batched dataset (list of batches).

    Args:
        base_list (list): A list of strings representing the questions.
        src_list (list): A list of lists, where each inner list contains the choices for a question.
        answers (list): A list of integers representing the correct answers for each question.
    Returns:
        tuple: A tuple containing two lists:
            - batches_src_tokens (List[[batch_size, seq_len]]): A list of batches of prompts tokens ids.
            - batches_base_tokens (List[[batch_size, seq_len]]): A list of batches of correct answers (ints).
            - batches_answer_token_indices (List[[batch_size, 2]]): A list of batches of answers token indices.
    """
    num_total_samples = len(base_list)
    batches = math.floor(num_total_samples/batch_size)

    def chunks_fn(xs, n):
        n = max(1, n)
        return (xs[i:i+n] for i in range(0, len(xs), n))

    batches_src_list = list(chunks_fn(src_list, batch_size))
    batches_base_list = list(chunks_fn(base_list, batch_size))
    #batches_src_label = list(chunks_fn(src_label_list, batch_size))
    #batches_base_label = list(chunks_fn(base_label_list, batch_size))
    batches_answers = list(chunks_fn(answers, batch_size))

    batches_src_tokens = []
    batches_base_tokens = []
    batches_answer_token_indices = []
    for chunk in range(batches):
        src_tokens = model.to_tokens(batches_src_list[chunk])
        base_tokens = model.to_tokens(batches_base_list[chunk])
        answers_in_batch = batches_answers[chunk]
        
        answer_token_indices = torch.tensor([[model.to_single_token(answers_in_batch[i][j]) for j in range(2)] for i in range(len(answers_in_batch))], device=model.cfg.device)

        batches_src_tokens.append(src_tokens)
        batches_base_tokens.append(base_tokens)
        batches_answer_token_indices.append(answer_token_indices)
        
    return batches_src_tokens, batches_base_tokens, batches_answer_token_indices

def get_logit_diff(logits, answer_token_indices, mean=True):
    answer_token_indices = answer_token_indices.to(logits.device)
    if len(logits.shape)==3:
        # Get final logits only
        logits = logits[:, -1, :]
    correct_logits = logits.gather(1, answer_token_indices[:, 0].unsqueeze(1))
    incorrect_logits = logits.gather(1, answer_token_indices[:, 1].unsqueeze(1))
    if mean:
        return (correct_logits - incorrect_logits).mean()
    else:
        return (correct_logits - incorrect_logits)

def residual_stack_to_logit_diff(
    model: HookedTransformer,
    residual_stack: Float[torch.Tensor, "components batch d_model"],
    cache: ActivationCache,
    logit_diff_directions : Float[torch.Tensor, "batch d_model"],
    mean : bool = True
) -> float:
    scaled_residual_stack = cache.apply_ln_to_stack(
        residual_stack, layer=-1, pos_slice=-1
    )
    res_to_logit_diff = einsum(
        "... batch d_model, batch d_model -> batch ...",
        scaled_residual_stack * model.ln_final.w,
        logit_diff_directions,
    )
    if mean==True:
        return res_to_logit_diff.mean(0)
    else:
        return res_to_logit_diff



def clean_blocks_labels(label: str) -> str:
    """
    Convert model block label into a more readable format.

    Args:
        label (str): The model component label to be cleaned.

    Returns:
        str: The cleaned label.

    """
    if label == 'embed':
        return 'Emb'
    else:
        component = label.split('_')[1]
        layer = label.split('_')[0]
        if component == 'mlp':
            label = f'MLP{layer}'
        else:
            label = f'Attn{layer}'
        return label

def paper_plot(fig, tickangle=60):
    """
    Applies styling to the given plotly figure object targeting paper plot quality.
    """
    fig.update_layout({
        'template': 'plotly_white',
    })
    fig.update_xaxes(showline=True, linewidth=2, linecolor='black', tickangle=tickangle,
                    gridcolor='rgb(200,200,200)', griddash='dash', zeroline=False)
    fig.update_yaxes(showline=True, linewidth=2, linecolor='black',
                    gridcolor='rgb(200,200,200)', griddash='dash', zeroline=False)
    return fig