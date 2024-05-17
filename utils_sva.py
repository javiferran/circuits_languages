import torch
import math
from typing import List, Union, Optional
from jaxtyping import Float
from fancy_einsum import einsum
import einops
from transformer_lens import HookedTransformer, HookedTransformerConfig, FactoredMatrix, ActivationCache
from PIL import ImageColor


html_colors = {
    'darkgreen' : '#138808',
    'green_drawio' : '#82B366',
    'dark_green_drawio' : '#557543',
    'dark_red_drawio' : '#990000',
    'blue_drawio' : '#6C8EBF',
    'orange_drawio' : '#D79B00',
    'red_drawio' : '#FF9999',
    'grey_drawio' : '#303030'}

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