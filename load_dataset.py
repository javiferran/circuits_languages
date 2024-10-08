import numpy as np
from datasets import load_dataset
import random
import torch
import math
import os
import json
from sklearn.model_selection import train_test_split

# seeds
seed_number = 10
random.seed(seed_number)
np.random.seed(seed_number)

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

def get_valid_spanish_verbs_nouns(model):
    # Get suitable list of verbs and nouns in singular and plural
    examples_valid_verbs_tuples_pred = [('es', 'son'), ('era', 'eran'), ('fue', 'fueron'), ('tuvo', 'tuvieron'), ('tiene', 'tienen')]
    examples_valid_verbs_tuples = [('tuvo', 'tuvieron')]
    examples_valid_nouns = [('cantante', 'cantantes'), ('ingeniero', 'ingenieros'), ('ministro', 'ministros'), ('piloto', 'pilotos')]
    verb_list_tuples = list(set(examples_valid_verbs_tuples + read_files(model, "datasets/plausible_spa_singular_plural_past_verbs.txt")))
    noun_list_tuples = list(set(examples_valid_nouns + read_files(model, "datasets/spa_singular_plural_nouns.txt")))
    return verb_list_tuples, noun_list_tuples, examples_valid_verbs_tuples_pred


def load_sva_dataset(model, language, subject_number='both', split='train', num_samples=100):

    dataset_path = './datasets/final_datasets'

    # Construct the filename based on language and split
    filename = f'{language}_{split}_sva_dataset.json'
    file_path = os.path.join(dataset_path, filename)

    # Load the dataset from the JSON file
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # Initialize lists to store the data
    base_list = []
    src_list = []
    base_label_list = []
    src_label_list = []
    answers = []
    ex_lang_list = []
    ex_number_list = []

    # Process the loaded data
    for i, item in enumerate(data.values()):
        if i < num_samples:
            if subject_number.lower() != 'both':
                if subject_number.lower() == item['base_subject_number'].lower():
                    base_list.append(item['base'])
                    src_list.append(item['src'])
                    base_label_list.append(item['base_label'])
                    src_label_list.append(item['src_label'])
                    answers.append((item['base_label'], item['src_label']))
                    ex_lang_list.append(language)
                    ex_number_list.append(item['base_subject_number'])
        
            else:
                # assert num_samples % 2 == 0 , "num_samples must be even"
                base_list.append(item['base'])
                src_list.append(item['src'])
                base_label_list.append(item['base_label'])
                src_label_list.append(item['src_label'])
                answers.append((item['base_label'], item['src_label']))
                ex_lang_list.append(language)
                ex_number_list.append(item['base_subject_number'])
    
    print((np.array(ex_number_list)=='singular').sum())
    print((np.array(ex_number_list)=='plural').sum())
    return {'base_list': base_list,
            'src_list': src_list,
            'base_label_list': base_label_list,
            'src_label_list': src_label_list,
            'answers': answers,
            'ex_lang_list': ex_lang_list,
            'ex_number_list': ex_number_list}


def get_batched_dataset(model, dataset, batch_size=16):
    """
    Creates a batched dataset (list of batches).

    Args:
        model (list):
        dataset (dict):
        batch_size (int)
    Returns:
        tuple: A tuple containing two lists:
            - batches_src_tokens (List[[batch_size, seq_len]]): A list of batches of prompts tokens ids.
            - batches_base_tokens (List[[batch_size, seq_len]]): A list of batches of correct answers (ints).
            - batches_answer_token_indices (List[[batch_size, 2]]): A list of batches of answers token indices.
    """
    base_list = dataset['base_list']
    src_list = dataset['src_list']
    base_label_list = dataset['base_label_list']
    src_label_list = dataset['src_label_list']
    answers = dataset['answers']
    ex_lang_list = dataset['ex_lang_list']
    ex_number_list = dataset['ex_number_list']

    num_total_samples = len(base_list)
    batches = math.floor(num_total_samples/batch_size) + 1

    def chunks_fn(xs, n):
        n = max(1, n)
        return (xs[i:i+n] for i in range(0, len(xs), n))

    batches_base_list = list(chunks_fn(base_list, batch_size))
    batches_src_list = list(chunks_fn(src_list, batch_size))
    batches_base_label_list = list(chunks_fn(base_label_list, batch_size))
    batches_src_label_list = list(chunks_fn(src_label_list, batch_size))
    batches_answers = list(chunks_fn(answers, batch_size))
    batches_ex_lang_list= list(chunks_fn(ex_lang_list, batch_size))
    batches_ex_number_list = list(chunks_fn(ex_number_list, batch_size))

    batches_src_tokens = []
    batches_base_tokens = []
    batches_answer_token_indices = []
    for chunk in range(batches):
        src_tokens = model.to_tokens(batches_src_list[chunk])
        base_tokens = model.to_tokens(batches_base_list[chunk])
        answers_in_batch = batches_answers[chunk]
        
        answer_token_indices = torch.tensor([[model.to_single_token(answers_in_batch[i][j]) for j in range(2)] for i in range(len(answers_in_batch))], device=model.cfg.device)

        batches_base_tokens.append(base_tokens)
        batches_src_tokens.append(src_tokens)
        batches_answer_token_indices.append(answer_token_indices)
        
    return {'batches_base_tokens': batches_base_tokens,
            'batches_src_tokens': batches_src_tokens,
            'batches_base_label_list': batches_base_label_list,
            'batches_src_label_list': batches_src_label_list,
            'batches_answer_token_indices': batches_answer_token_indices,
            'batches_ex_lang_list': batches_ex_lang_list,
            'batches_ex_number_list': batches_ex_number_list}


def create_english_dataset():
    len_sv_num = 6 # sentences should have 6 tokens

    for split in ['train', 'validation', 'test']:
        # Create English dataset
        final_dict = {}

        # English dataset
        hf_dataset = load_dataset("aryaman/causalgym", split=split)
        hf_dataset = hf_dataset.filter(lambda example: example['task']=='agr_sv_num_subj-relc')#agr_sv_num_pp
        dataset = hf_dataset

        valid_counter = 0
        for i in range(len(dataset)):
            for type_sentence in ['src', 'base']:
                for word in dataset[i][type_sentence]:
                    if len(word.split())>1: # eliminate compound words like ' taxi driver'
                        break
        
            # Replace <|endoftext|> with empty space (we will add BOS later)
            src = ''.join(dataset[i]['src']).replace('<|endoftext|>','')
            base = ''.join(dataset[i]['base']).replace('<|endoftext|>','')

            # Add to dataset only if sentences have the correct number of tokens
            if len(src.split())==len_sv_num and len(base.split())==len_sv_num:
                final_dict[valid_counter] = {}
                final_dict[valid_counter]['src'] = src
                final_dict[valid_counter]['base'] = base
                final_dict[valid_counter]['src_label'] = dataset[i]['src_label']
                final_dict[valid_counter]['base_label'] = dataset[i]['base_label']
                if base.split()[1].endswith('s'):
                    # plural
                    final_dict[valid_counter]['base_subject_number'] = 'plural'
                else:
                    # singular
                    final_dict[valid_counter]['base_subject_number'] = 'singular'

                valid_counter += 1
        
        # Create the directory if it doesn't exist
        final_datsets_dir = './datasets/final_datasets'
        os.makedirs(final_datsets_dir, exist_ok=True)

        # Save final_dict as a JSON file
        output_path = f'{final_datsets_dir}/english_{split}_sva_dataset.json'
        with open(output_path, 'w') as f:
            json.dump(final_dict, f, indent=4)

        print(f"Dataset saved to {output_path}")

def create_spanish_dataset(model):
    # Create Spanish dataset
    verb_list_tuples, noun_list_tuples, examples_valid_verbs_tuples_pred = get_valid_spanish_verbs_nouns(model)
    noun_list_sing = [f' {noun_tuple[0]}' for noun_tuple in noun_list_tuples]
    noun_list_plural = [f' {noun_tuple[1]}' for noun_tuple in noun_list_tuples]

    verb_1_list_sing = [f' {verb_tuple[0]}' for verb_tuple in verb_list_tuples]
    verb_1_list_plural = [f' {verb_tuple[1]}' for verb_tuple in verb_list_tuples]

    verb_2_list_sing = [f' {verb_tuple[0]}' for verb_tuple in examples_valid_verbs_tuples_pred]
    verb_2_list_plural = [f' {verb_tuple[1]}' for verb_tuple in examples_valid_verbs_tuples_pred]

    permutations_list = [[i,j,k] for i in range(len(noun_list_sing)) for j in range(len(verb_1_list_sing)) for k in range(len(noun_list_sing)) if k!=i]
    permutations_array = np.array(permutations_list)
    
    np.random.seed(seed_number)
    np.random.shuffle(permutations_array)

    final_dict = {}

    for valid_counter, (i,j,k) in enumerate(permutations_array):
        final_dict[valid_counter] = {}

        sent_1 = f'Los{noun_list_plural[k]} que{verb_1_list_plural[j]} al{noun_list_sing[i]}'
        sent_2 = f'El{noun_list_sing[k]} que{verb_1_list_sing[j]} al{noun_list_sing[i]}'

        # Avoid verb repetition
        verbs_indices = list(range(0,len(verb_2_list_sing)))
        already_verb = j
        available_verb_indices = verbs_indices[:already_verb] + verbs_indices[already_verb+1:]
        ver_2_idx = np.random.choice(available_verb_indices)

        # Get random number [0,1] to decide if we use the sentence with the plural base or the one with the singular base
        rdm_num = int(round(random.uniform(0, 1), 0))

        if rdm_num== 0:
            src = sent_1
            base = sent_2
            src_label = verb_2_list_plural[ver_2_idx]
            base_label = verb_2_list_sing[ver_2_idx]
        else:
            src = sent_2
            base = sent_1
            src_label = verb_2_list_sing[ver_2_idx]
            base_label = verb_2_list_plural[ver_2_idx]
        
        final_dict[valid_counter]['src'] = src
        final_dict[valid_counter]['base'] = base
        final_dict[valid_counter]['src_label'] = src_label
        final_dict[valid_counter]['base_label'] = base_label

        if base_label[-1] == 'n':
            # plural
            final_dict[valid_counter]['base_subject_number'] = 'plural'
        else:
            # singular
            final_dict[valid_counter]['base_subject_number'] = 'singular'

    
    len_dataset = len(final_dict)

    train_end_idx = len_dataset*70//100
    val_end_idx = len_dataset*85//100
    split_dicts = {}
    split_dicts['train'] = {k: final_dict[k] for k in list(final_dict)[:train_end_idx]}
    split_dicts['validation'] = {k: final_dict[k] for k in list(final_dict)[train_end_idx:val_end_idx]}
    split_dicts['test'] = {k: final_dict[k] for k in list(final_dict)[val_end_idx:]}

    for split in ['train', 'validation', 'test']:
        # Create the directory if it doesn't exist
        final_datsets_dir = './datasets/final_datasets'
        os.makedirs(final_datsets_dir, exist_ok=True)

        # Save final_dict as a JSON file
        output_path = f'{final_datsets_dir}/spanish_{split}_sva_dataset.json'
        with open(output_path, 'w') as f:
            json.dump(split_dicts[split], f, indent=4)

        print(f"Dataset saved to {output_path}")


def create_sva_datasets(model):
    create_english_dataset()
    create_spanish_dataset(model)
