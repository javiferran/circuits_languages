
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


def clean_blocks_labels(label: str) -> str:
    """
    Convert model block label into a more readable format.

    Args:
        label (str): The model component label to be cleaned.

    Returns:
        str: The cleaned label.

    """
    if label == 'embed':
        return 'Embedding'
    else:
        component = label.split('_')[1]
        layer = label.split('_')[0]
        if component == 'mlp':
            label = f'MLP{layer}'
        else:
            label = f'Attn{layer}'
        return label

def paper_plot(fig):
    """
    Applies styling to the given plotly figure object targeting paper plot quality.
    """
    fig.update_layout({
        'template': 'plotly_white',
    })
    fig.update_xaxes(showline=True, linewidth=2, linecolor='black', tickangle=60,
                    gridcolor='rgb(200,200,200)', griddash='dash', zeroline=False)
    fig.update_yaxes(showline=True, linewidth=2, linecolor='black',
                    gridcolor='rgb(200,200,200)', griddash='dash', zeroline=False)
    return fig