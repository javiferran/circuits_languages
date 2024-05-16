import torch
import math
from typing import List, Union, Optional
from jaxtyping import Float
from fancy_einsum import einsum
import einops
from transformer_lens import HookedTransformer, HookedTransformerConfig, FactoredMatrix, ActivationCache

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