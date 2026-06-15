import torch
from itertools import combinations
import numpy as np
import argparse
import pandas as pd
import os
import shutil

main_colunas = ['model', 'pooling', 'type_pooling','agg', 'layer', 'epochs', 'out_vec_size', 'qtd_layers', 'nhid', 'params', 'best_layers']

def get_agg_base():
    list_sum_agg = []
    list_avg_agg = []

    ranges = list(range(1, 13))

    slices = {}
    for size in range(2, 13):  # De 2 até 12
        slices[size] = [f"SUM-{group[0]}-{group[-1]}" for group in (ranges[i:i+size] for i in range(len(ranges) - size + 1))]

    for size, groups in slices.items():
        list_sum_agg+=groups

    slices = {}
    for size in range(2, 13):  # De 2 até 12
        slices[size] = [f"AVG-{group[0]}-{group[-1]}" for group in (ranges[i:i+size] for i in range(len(ranges) - size + 1))]

    for size, groups in slices.items():
        list_avg_agg+=groups
    
    return list_sum_agg, list_avg_agg

def get_agg_large():
    list_sum_agg = []
    list_avg_agg = []

    ranges = list(range(1, 25))

    slices = {}
    for size in range(2, 25):  # De 2 até 12
        slices[size] = [f"SUM-{group[0]}-{group[-1]}" for group in (ranges[i:i+size] for i in range(len(ranges) - size + 1))]

    for size, groups in slices.items():
        list_sum_agg+=groups

    slices = {}
    for size in range(2, 25):  # De 2 até 12
        slices[size] = [f"AVG-{group[0]}-{group[-1]}" for group in (ranges[i:i+size] for i in range(len(ranges) - size + 1))]

    for size, groups in slices.items():
        list_avg_agg+=groups
    
    return list_sum_agg, list_avg_agg

def get_pooling_techniques(poolings_args, name_agg):

    simple_ns_poolings = ['AVG-NS', 'SUM-NS', 'MAX-NS'] 
    simple_poolings = ['CLS', 'AVG', 'SUM', 'MAX']    

    all_poolings_individuals = simple_poolings + simple_ns_poolings

    two_tokens_poolings = [f"{a}+{b}" for a, b in combinations(all_poolings_individuals, 2)]
    three_tokens_poolings = [f"{a}+{b}+{c}" for a, b, c in combinations(all_poolings_individuals, 3)]
    #four_tokens_poolings = [f"{a}+{b}+{c}+{d}" for a, b, c, d in combinations(all_poolings_individuals, 4)]

    pooling_prefixs = []
    
    if poolings_args[0] == 'all':
        pooling_prefixs = all_poolings_individuals + two_tokens_poolings + three_tokens_poolings# + four_tokens_poolings
        return pooling_prefixs
    
    if poolings_args[0] == 'best':
        pooling_prefixs = two_tokens_poolings + three_tokens_poolings# + four_tokens_poolings
        return pooling_prefixs
    
    if 'simple' in poolings_args:
        pooling_prefixs += simple_poolings
        #return pooling_prefixs
    if 'simple_all' in poolings_args:
        pooling_prefixs += all_poolings_individuals
        #return pooling_prefixs
    if 'simple-ns' in poolings_args:
        pooling_prefixs += simple_ns_poolings
        #return pooling_prefixs
    if 'two' in poolings_args:
        pooling_prefixs += two_tokens_poolings
        #return pooling_prefixs
    if 'three' in poolings_args:
        pooling_prefixs += three_tokens_poolings
        #return pooling_prefixs  
    #if 'four' in poolings_args:
    #    pooling_prefixs += four_tokens_poolings
    #    #return pooling_prefixs     
    
    
    if len(pooling_prefixs) > 0:
        return pooling_prefixs
    else:
        return poolings_args

def get_list_layers(final_layer, initial_layer, agg_layers_args):

    #if final_layer == 12:
    #    list_lyrs_agg_sum, list_lyrs_agg_avg = get_agg_base()
    #if final_layer == 24:
    #    list_lyrs_agg_sum, list_lyrs_agg_avg = get_agg_large()

    list_lyrs_agg_sum, list_lyrs_agg_avg = get_agg_base()
    list_lyrs_agg = list_lyrs_agg_sum + list_lyrs_agg_avg

    lyrs = []
        
    if agg_layers_args[0] == 'ALL':
        for i in range(initial_layer, final_layer):
            lyrs.append(f"LYR-{i+1}")
        lyrs += list_lyrs_agg
        return lyrs
    
    if agg_layers_args[0] == 'SUMAGGLAYERS':
        return list_lyrs_agg_sum
    if agg_layers_args[0] == 'SUMAGGLAYERS1':
        return list_lyrs_agg_sum[:15]
    if agg_layers_args[0] == 'SUMAGGLAYERS2':
        return list_lyrs_agg_sum[15:30]
    if agg_layers_args[0] == 'SUMAGGLAYERS3':
        return list_lyrs_agg_sum[30:45]
    if agg_layers_args[0] == 'SUMAGGLAYERS4':
        return list_lyrs_agg_sum[45:]
    
    if agg_layers_args[0] == 'AVGAGGLAYERS':
        return list_lyrs_agg_avg
    if agg_layers_args[0] == 'AVGAGGLAYERS1':
        return list_lyrs_agg_avg[:15]
    if agg_layers_args[0] == 'AVGAGGLAYERS2':
        return list_lyrs_agg_avg[15:30]
    if agg_layers_args[0] == 'AVGAGGLAYERS3':
        return list_lyrs_agg_avg[30:45]
    if agg_layers_args[0] == 'AVGAGGLAYERS4':
        return list_lyrs_agg_avg[45:]
    
    if agg_layers_args[0] == 'LYR':
        for i in range(initial_layer, final_layer):
            lyrs.append(f"LYR-{i+1}")
        return lyrs
    
    if agg_layers_args[0] == 'BEST':
        return ["BEST"]
    
    else:
        return agg_layers_args

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

def batcher(params, batch):
    # batch é uma lista de listas de palavras, ex: [['bom', 'dia'], ['frase', 'maior']]
    
    # --- NOVO: Lógica de ordenação ---
    # Mantém o registro dos índices originais
    original_indices = np.arange(len(batch))
    
    # Calcula o comprimento de cada sentença
    lengths = np.array([len(sent) for sent in batch])
    
    # Obtém os índices que ordenariam o batch por comprimento
    sorted_indices = np.argsort(lengths)
    
    # Ordena o batch e os índices originais
    sorted_batch = [batch[i] for i in sorted_indices]
    original_indices_sorted = [original_indices[i] for i in sorted_indices]
    
    # --- Fim da lógica de ordenação ---

    # Converte para strings e chama o _encode com o batch ordenado
    sentences = [' '.join(sent) for sent in sorted_batch]
    embeddings = params['encoder']._encode(sentences, params.current_task)
    
    # --- NOVO: Reordenar os embeddings para a ordem original ---
    # Cria um array para os embeddings na ordem correta
    restored_order_embeddings = np.zeros_like(embeddings)
    
    # Usa os índices ordenados para colocar cada embedding de volta em sua posição original
    restored_order_embeddings[original_indices_sorted] = embeddings
    
    return restored_order_embeddings

def strategies_pooling_list (args, qtd_layers):
        initial_layer_args = args.initial_layer
        final_layer_args = args.final_layer
        poolings_args = args.poolings
        agg_layers_args = args.agg_layers
        
        #POOLING
        pooling_techniques = get_pooling_techniques(poolings_args, agg_layers_args)
        
        #LAYERS
        if initial_layer_args is not None:
            initial_layer = initial_layer_args
        else:
            initial_layer = int(qtd_layers / 2)

        if final_layer_args is not None:
            final_layer = final_layer_args
        else:
            final_layer = int(qtd_layers)

        list_lyrs = get_list_layers(final_layer, initial_layer, agg_layers_args)

        #STRATEGIES CONCAT
        pooling_strategies = []
        for l in list_lyrs:
            for p in pooling_techniques:
                pooling_strategies.append(p + "_" + l) 

        #RETURN
        return pooling_strategies, pooling_techniques, list_lyrs

def parse_dict_with_eval(value):
    try:
        if isinstance(value, str):
            value = value.replace('np.float64', 'float')
            return eval(value)
        return {}
    except Exception as e:
        return {}

def parse_dict_with_eval_other(value):
    try:
        if isinstance(value, str):
            value = value.replace('np.float64', 'float')
            value = ','.join(value.split(',')[:3]) + '}'
            return eval(value)
        return {}
    except Exception as e:
        return {}
    
def get_type_pooling(pooling_str):

    simple_ns_poolings = ['AVG-NS', 'SUM-NS', 'MAX-NS'] 
    simple_poolings = ['CLS', 'AVG', 'SUM', 'MAX']    

    all_poolings_individuals = simple_poolings + simple_ns_poolings

    two_tokens_poolings = [f"{a}+{b}" for a, b in combinations(all_poolings_individuals, 2)]
    three_tokens_poolings = [f"{a}+{b}+{c}" for a, b, c in combinations(all_poolings_individuals, 3)]
    
    if pooling_str in simple_poolings:
        return "simple"
    if pooling_str in simple_ns_poolings:
        return "simple-ns"
    if pooling_str in two_tokens_poolings:
        return "two-tokens"
    if pooling_str in three_tokens_poolings:
        return "three-tokens"
    else:
        return "not categorized"
    
def tables_process(data, columns_tasks, type_task, path_cl, filename_task):

    ordem_colunas = main_colunas + columns_tasks

    devacc_data = {'model': data['model'], 'pooling': data['pooling'], 'epochs': data['epochs'], 'out_vec_size': data['out_vec_size'], 'qtd_layers': data['qtd_layers'], 'nhid': data['nhid'], 'best_layers': data['best_layers']}
    acc_data = {'model': data['model'], 'pooling': data['pooling'], 'epochs': data['epochs'], 'out_vec_size': data['out_vec_size'], 'qtd_layers': data['qtd_layers'], 'nhid': data['nhid'], 'best_layers': data['best_layers']}

    if type_task == 'cl':
        for task in columns_tasks:
            devacc_data[task] = data[task].apply(lambda x:x.get('devacc', None))
            acc_data[task] = data[task].apply(lambda x: x.get('acc', None))

    elif type_task == 'si':
        for task in columns_tasks:
            if task in columns_tasks[:5]:
                devacc_data[task] = data[task].apply(lambda x: (x.get('pearson', None).get('mean', None)) * 100)
                acc_data[task] = data[task].apply(lambda x: (x.get('spearman', None).get('mean', None)) * 100)
            if task in columns_tasks[5:]:
                devacc_data[task] = data[task].apply(lambda x: (x.get('pearson', None)) * 100)
                acc_data[task] = data[task].apply(lambda x: (x.get('spearman', None)) * 100)


    devacc_table = pd.DataFrame(devacc_data)
    acc_table = pd.DataFrame(acc_data)        

    devacc_table[['agg', 'layer']] = devacc_table['pooling'].str.split('_', expand=True)
    acc_table[['agg', 'layer']] = acc_table['pooling'].str.split('_', expand=True)

    devacc_table['params'] = "-".join(filename_task.split('_')[5:11])
    acc_table['params'] = "-".join(filename_task.split('_')[5:11])

    devacc_table['type_pooling'] = devacc_table['agg'].apply(get_type_pooling)
    acc_table['type_pooling'] = acc_table['agg'].apply(get_type_pooling)

    devacc_table = devacc_table[ordem_colunas]
    acc_table = acc_table[ordem_colunas]  

    devacc_table['avg_tasks'] = devacc_table[columns_tasks].mean(axis=1)
    acc_table['avg_tasks'] = acc_table[columns_tasks].mean(axis=1)

    if type_task == 'cl':
        devacc_table.to_csv(os.path.join(path_cl, filename_task + '_processado_devacc.csv'))
        acc_table.to_csv(os.path.join(path_cl, filename_task + '_processado_acc.csv'))

    elif type_task == 'si':
        devacc_table.to_csv(os.path.join(path_cl, filename_task + '_processado_pearson.csv'))
        acc_table.to_csv(os.path.join(path_cl, filename_task + '_processado_spearman.csv'))

def main_evaluate(final_df, type_task, path_for_save, filename_task, tasks_list):
    if type_task == "cl":
        tables_process(final_df, tasks_list, type_task, path_for_save, filename_task)

    elif type_task == "si":
        tables_process(final_df, tasks_list, type_task, path_for_save, filename_task)
