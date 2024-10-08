import torch
import math
from typing import List, Union, Optional
from jaxtyping import Float
from fancy_einsum import einsum
import einops
from transformer_lens import HookedTransformer, HookedTransformerConfig, FactoredMatrix, ActivationCache
import transformer_lens.patching as patching
from PIL import ImageColor
from functools import partial

html_colors = {
    'darkgreen' : '#138808',
    'green_drawio' : '#82B366',
    'dark_green_drawio' : '#557543',
    'dark_red_drawio' : '#990000',
    'blue_drawio' : '#6C8EBF',
    'orange_drawio' : '#D79B00',
    'red_drawio' : '#FF9999',
    'grey_drawio' : '#303030',
    'brown_D3' : '#8C564B',}

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
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')

    return fig

def to_group(number_val, lang_val, alpha):
    if number_val=='Singular' and lang_val=='Spanish':
        return ('Spanish Singular', 'rgba' + str(tuple(list(ImageColor.getcolor(html_colors['dark_green_drawio'], "RGB")) + [alpha])))
    
    elif number_val=='Plural' and lang_val=='Spanish':
        return ('Spanish Plural', 'rgba' + str(tuple(list(ImageColor.getcolor(html_colors['green_drawio'], "RGB")) + [alpha])))
    
    elif number_val=='Singular' and lang_val=='English':
        return ('English Singular', 'rgba' + str(tuple(list(ImageColor.getcolor(html_colors['dark_red_drawio'], "RGB")) + [alpha]))
    )
    elif number_val=='Plural' and lang_val=='English':
        return ('English Plural', 'rgba' + str(tuple(list(ImageColor.getcolor(html_colors['red_drawio'], "RGB")) + [alpha])))
    else:
        print('ERROR!')

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def run_pca(X, n_components):
    # Standardize data before applying PCA
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    pca = PCA(n_components=n_components)
    X_embedded = pca.fit_transform(X)
    return X_embedded, pca, scaler

def compute_diff_means(dataset_rep, class_list, class_1, class_2):
    # Get representations for each class
    indices_1 = [i for i, x in enumerate(class_list) if x == class_1]
    indices_2 = [i for i, x in enumerate(class_list) if x == class_2]
    rep_class_1 = dataset_rep[indices_1,:]
    rep_class_2 = dataset_rep[indices_2,:]
    # Average across batch dimension
    mean_class_1 = rep_class_1.mean(0)
    mean_class_2 = rep_class_2.mean(0)

    return mean_class_1 - mean_class_2


def compute_act_patching(model: HookedTransformer,
                         metric: callable,
                         patching_type: str,
                         batches_base_tokens: list,
                         batches_src_tokens: list,
                         batches_answer_token_indices: list,
                         batches: int):
    # resid_streams
    # heads_all_pos : attn heads all positions at the same time
    # heads_last_pos: attn heads last position
    # full: (resid streams, attn block outs and mlp outs)

    list_resid_pre_act_patch_results = []
    for batch in range(batches):
        base_tokens = batches_base_tokens[batch]
        src_tokens = batches_src_tokens[batch]
        base_logits, base_cache = model.run_with_cache(base_tokens)
        src_logits, _ = model.run_with_cache(src_tokens)
        answer_token_indices = batches_answer_token_indices[batch]
        
        metric_fn = partial(metric, answer_token_indices=answer_token_indices)
        if patching_type=='resid_streams':
            # resid_pre_act_patch_results -> [n_layers, pos]
            patch_results = patching.get_act_patch_resid_pre(model, src_tokens, base_cache, metric_fn)
        elif patching_type=='heads_all_pos':
            patch_results = patching.get_act_patch_attn_head_out_all_pos(model, src_tokens, base_cache, metric_fn)
        elif patching_type=='heads_last_pos':
            # Activation patching per position
            attn_head_out_per_pos_patch_results = patching.get_act_patch_attn_head_out_by_pos(model, src_tokens, base_cache, metric_fn)
            # Select last position
            patch_results = attn_head_out_per_pos_patch_results[:,-1]
        elif patching_type=='full':
            patch_results = patching.get_act_patch_block_every(model, src_tokens, base_cache, metric_fn)
        
        list_resid_pre_act_patch_results.append(patch_results)

    total_resid_pre_act_patch_results = torch.stack(list_resid_pre_act_patch_results, 0).mean(0)

    return total_resid_pre_act_patch_results