import numpy as np
from datasets import load_dataset
import random
import torch
import math

# seeds
random.seed(10)
np.random.seed(10)

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
    #examples_valid_verbs_tuples = [('es', 'son'), ('tiene', 'tienen'), ('fue', 'fueron'), ('era', 'eran')]
    examples_valid_verbs_tuples_pred = [('fue', 'fueron'),('era', 'eran')]
    examples_valid_verbs_tuples = [('tuvo', 'tuvieron')]
    examples_valid_nouns = [('cantante', 'cantantes'), ('ingeniero', 'ingenieros'), ('ministro', 'ministros'), ('piloto', 'pilotos')]
    verb_list_tuples = list(set(examples_valid_verbs_tuples + read_files(model, "datasets/plausible_spa_singular_plural_past_verbs.txt")))
    noun_list_tuples = list(set(examples_valid_nouns + read_files(model, "datasets/spa_singular_plural_nouns.txt")))

    return verb_list_tuples, noun_list_tuples, examples_valid_verbs_tuples_pred

def load_sva_dataset(model, language, dataset_type, num_samples, start_at=0):

    answers = []
    src_list = []
    base_list = []
    src_label_list = []
    base_label_list = []
    ex_number_list = []
    ex_lang_list = []

    if language=='english' or language=='both':
        len_sv_num = 6 # sentences should have 6 tokens

        hf_dataset = load_dataset("aryaman/causalgym", split='train')
        hf_dataset = hf_dataset.filter(lambda example: example['task']=='agr_sv_num_subj-relc')#agr_sv_num_pp

        if dataset_type=='singular':
            dataset = hf_dataset.filter(lambda example: example["base_type"]=='singular')
        elif dataset_type=='plural':
            dataset = hf_dataset.filter(lambda example: example["base_type"]=='plural')
        else:
            dataset = hf_dataset

        match_counter = start_at
        i=start_at
        while match_counter<num_samples:
            print('hey')
            for type_sentence in ['src','base']:
                for word in dataset[i][type_sentence]:
                    if len(word.split())>1: # eliminate compound words like ' taxi driver'
                        break
            
            src = ''.join(dataset[i]['src']).replace('<|endoftext|>','')
            base = ''.join(dataset[i]['base']).replace('<|endoftext|>','')
            if len(src.split())==len_sv_num and len(base.split())==len_sv_num:
                src_list.append(src)
                base_list.append(base)
                src_label = dataset[i]['src_label']
                base_label = dataset[i]['base_label']
                src_label_list.append(src_label)
                base_label_list.append(base_label)
                answers.append((base_label, src_label))
                ex_lang_list.append('English')
                print(base.split()[1])
                print(f'{base} {base_label}\n{src} {src_label}')
                if base.split()[1].endswith('s'):
                    # Plural
                    ex_number_list.append('Plural')
                else:
                    # Singular
                    ex_number_list.append('Singular')
                match_counter += 1
            i += 1
            

    if language=='spanish' or language=='both':
        verb_list_tuples, noun_list_tuples, examples_valid_verbs_tuples_pred = get_valid_spanish_verbs_nouns(model)
        noun_list_sing = [f' {noun_tuple[0]}' for noun_tuple in noun_list_tuples]
        noun_list_plural = [f' {noun_tuple[1]}' for noun_tuple in noun_list_tuples]

        verb_1_list_sing = [f' {verb_tuple[0]}' for verb_tuple in verb_list_tuples]
        verb_1_list_plural = [f' {verb_tuple[1]}' for verb_tuple in verb_list_tuples]

        verb_2_list_sing = [f' {verb_tuple[0]}' for verb_tuple in examples_valid_verbs_tuples_pred]
        verb_2_list_plural = [f' {verb_tuple[1]}' for verb_tuple in examples_valid_verbs_tuples_pred]

        permutations_list = [[i,j,k] for i in range(len(noun_list_sing)) for j in range(len(verb_1_list_sing)) for k in range(len(noun_list_sing)) if k!=i]
        permutations_array = np.array(permutations_list)
        np.random.shuffle(permutations_array)
        counter = 0

        for i,j,k in permutations_array:
            counter += 1
            sent_1 = f'Los{noun_list_plural[k]} que{verb_1_list_plural[j]} al{noun_list_sing[i]}'
            sent_2 = f'El{noun_list_sing[k]} que{verb_1_list_sing[j]} al{noun_list_sing[i]}'

            if dataset_type=='singular':
                rdm_num = 0
            elif dataset_type=='plural':
                rdm_num = 1
            else:
                rdm_num = int(round(random.uniform(0, 1), 0))

            # Avoid verb repetition
            verbs_indices = list(range(0,len(verb_2_list_sing)))
            already_verb = j
            available_verb_indices = verbs_indices[:already_verb] + verbs_indices[already_verb+1:]
            ver_2_idx = np.random.choice(available_verb_indices)
            
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
            if base_label[-1] == 'n':
                # Plural
                ex_number_list.append('Plural')
            else:
                # Singular
                ex_number_list.append('Singular')

            src_list.append(src)
            base_list.append(base)
            src_label_list.append(src_label)
            base_label_list.append(base_label)
            answers.append((base_label, src_label))
            ex_lang_list.append('Spanish')

            print(f'{base} {base_label}\n{src} {src_label}\n')
            
        
            if counter >=num_samples:
                break

    return {'base_list': base_list,
            'src_list': src_list,
            'base_label_list': base_label_list,
            'src_label_list': src_label_list,
            'answers': answers,
            'ex_lang_list': ex_lang_list}


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

        batches_base_tokens.append(base_tokens)
        batches_src_tokens.append(src_tokens)
        batches_answer_token_indices.append(answer_token_indices)
        
    return batches_base_tokens, batches_src_tokens, batches_answer_token_indices