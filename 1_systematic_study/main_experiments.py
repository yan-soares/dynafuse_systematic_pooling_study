import senteval
from transformers import AutoTokenizer, DebertaV2Model, DebertaV2Tokenizer, BertTokenizer, BertModel, RobertaTokenizer, RobertaModel, AutoModel
import torch
import argparse
import pandas as pd
import logging
import os
import functions_code
import functions_code_testes
from nltk.corpus import stopwords
import json
import time
import subprocess
import numpy as np

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

class SentenceEncoder:
    def __init__(self, model_name, device):
        self.device = device
        self.size_embedding = None
        self.pooling_strategy = None
        self.print_best_layers = None
        
        self.stopwords_set_ids = None
        self.cls_token_id = None
        self.sep_token_id = None

        self.general_embeddings = {}
        self.list_poolings = None
        self.list_layers = None
        self.actual_layer = None

        if model_name == 'bert-base' or  model_name == 'bert-large':
            if model_name == 'bert-base':
                self.name_model = 'google-bert/bert-base-uncased'
                self.qtd_layers = 12
            if model_name == 'bert-large':
                self.name_model = 'google-bert/bert-large-uncased'
                self.qtd_layers = 24
            self.tokenizer = BertTokenizer.from_pretrained(self.name_model)
            try:
                self.model = BertModel.from_pretrained(
                    self.name_model, output_hidden_states=True, attn_implementation="flash_attention_2"
                ).to(self.device)
                print(f"Modelo {model_name} carregado com sucesso usando Flash Attention 2.")
            except (ValueError, ImportError) as e:
                print(f"AVISO: Flash Attention 2 não suportado para {model_name}. Carregando modelo padrão. Erro: {e}")
                self.model = BertModel.from_pretrained(self.name_model, output_hidden_states=True).to(self.device)

        if model_name == 'roberta-base' or  model_name == 'roberta-large':
            if model_name == 'roberta-base':
                self.name_model = 'FacebookAI/roberta-base'
                self.qtd_layers = 12
            if model_name == 'roberta-large':
                self.name_model = 'FacebookAI/roberta-large'
                self.qtd_layers = 24
            self.tokenizer = RobertaTokenizer.from_pretrained(self.name_model)
            try:
                self.model = RobertaModel.from_pretrained(
                    self.name_model, output_hidden_states=True, attn_implementation="flash_attention_2"
                ).to(self.device)
                print(f"Modelo {model_name} carregado com sucesso usando Flash Attention 2.")
            except (ValueError, ImportError) as e:
                print(f"AVISO: Flash Attention 2 não suportado para {model_name}. Carregando modelo padrão. Erro: {e}")
                self.model = RobertaModel.from_pretrained(self.name_model, output_hidden_states=True).to(self.device)

        if model_name == 'deberta-base' or model_name == 'deberta-large':
            if model_name == 'deberta-base':
                self.name_model = 'microsoft/deberta-v3-base'
                self.qtd_layers = 12
            if model_name == 'deberta-large':
                self.name_model = 'microsoft/deberta-v3-large'
                self.qtd_layers = 24
            self.tokenizer = DebertaV2Tokenizer.from_pretrained(self.name_model)
            try:
                self.model = DebertaV2Model.from_pretrained(
                    self.name_model, output_hidden_states=True, attn_implementation="flash_attention_2"
                ).to(self.device)
                print(f"Modelo {model_name} carregado com sucesso usando Flash Attention 2.")
            except (ValueError, ImportError) as e:
                print(f"AVISO: Flash Attention 2 não suportado para {model_name}. Carregando modelo padrão. Erro: {e}")
                self.model = DebertaV2Model.from_pretrained(self.name_model, output_hidden_states=True).to(self.device)

        if model_name == 'angle-base' or model_name == 'angle-large':       
            if model_name == 'angle-base':
                self.name_model = 'SeanLee97/angle-bert-base-uncased-nli-en-v1'
                self.qtd_layers = 12
            if model_name == 'angle-large':
                self.name_model = 'WhereIsAI/UAE-Large-V1'
                self.qtd_layers = 24            
            self.tokenizer = AutoTokenizer.from_pretrained(self.name_model)
            try:
                self.model = AutoModel.from_pretrained(
                    self.name_model, output_hidden_states=True, attn_implementation="flash_attention_2", trust_remote_code=True
                ).to(self.device)
                print(f"Modelo {model_name} carregado com sucesso usando Flash Attention 2.")
            except (ValueError, ImportError) as e:
                print(f"AVISO: Flash Attention 2 não suportado para {model_name}. Carregando modelo padrão. Erro: {e}")
                self.model = AutoModel.from_pretrained(self.name_model, output_hidden_states=True, trust_remote_code=True).to(self.device) 

        if model_name == 'allmpnet':
            self.name_model = 'sentence-transformers/all-mpnet-base-v2'
            self.qtd_layers = 12
            self.tokenizer = AutoTokenizer.from_pretrained(self.name_model)
            try:
                self.model = AutoModel.from_pretrained(
                    self.name_model, output_hidden_states=True, attn_implementation="flash_attention_2"
                ).to(self.device)
                print(f"Modelo {model_name} carregado com sucesso usando Flash Attention 2.")
            except (ValueError, ImportError) as e:
                print(f"AVISO: Flash Attention 2 não suportado para {model_name}. Carregando modelo padrão. Erro: {e}")
                self.model = AutoModel.from_pretrained(self.name_model, output_hidden_states=True).to(self.device)

        #try:
        #    self.model = torch.compile(self.model)
        #    print("Modelo compilado com torch.compile() para maior performance.")
        #except Exception:
        #    print("torch.compile() não disponível. Rodando sem compilação.")

        self._prepare_special_token_ids()

    def _prepare_special_token_ids(self):
        # Converte a lista de stopwords (strings) para uma lista de IDs de tokens
        stopwords_list = stopwords.words('english')
        stopword_ids = self.tokenizer.convert_tokens_to_ids(stopwords_list)
        
        # Filtra IDs desconhecidos e cria um tensor na GPU para comparações rápidas
        stopword_ids_filtered = [id for id in stopword_ids if id != self.tokenizer.unk_token_id]
        self.stopwords_set_ids = torch.tensor(stopword_ids_filtered, device=self.device)
        
        # Armazena os IDs dos tokens CLS e SEP para fácil acesso
        self.cls_token_id = self.tokenizer.cls_token_id
        self.sep_token_id = self.tokenizer.sep_token_id

    def _encode(self, sentences, current_task, batch_size=16384): 
        tokens = self.tokenizer(
            sentences,
            padding="longest",
            truncation=True,
            return_tensors="pt",
            max_length=self.model.config.max_position_embeddings
        )

        # Envia tokens para GPU de forma mais eficiente
        tokens = {key: val.to(self.device, non_blocking=True) for key, val in tokens.items()}
        all_embeddings = []

        for i in range(0, len(sentences), batch_size):
            batch_tokens = {key: val[i:i+batch_size] for key, val in tokens.items()}

            # Substituí no_grad() → inference_mode() (mais otimizado para inferência)
            with torch.inference_mode(), torch.amp.autocast('cuda'):
                output = self.model(**batch_tokens)
                embeddings = self._apply_pooling(output, batch_tokens['attention_mask'], batch_tokens['input_ids'])

            all_embeddings.append(embeddings)

        self.size_embedding = all_embeddings[0].shape 
        final_embeddings = torch.cat(all_embeddings, dim=0).to('cpu').numpy()

        del batch_tokens, output
        torch.cuda.empty_cache()   # Libera memória de GPU ociosa
        
        return final_embeddings

    def _create_combined_mask(self, input_ids, attention_mask, exclude_stopwords=False, exclude_cls_sep=False):
        """
        OTIMIZAÇÃO: Cria uma máscara combinada de forma vetorizada na GPU.
        - `attention_mask`: máscara base para ignorar padding.
        - `exclude_stopwords`: se True, zera a posição de stopwords.
        - `exclude_cls_sep`: se True, zera a posição dos tokens [CLS] e [SEP].
        """
        combined_mask = attention_mask.clone()

        if exclude_stopwords:
            # `torch.isin` é uma operação vetorizada e muito rápida na GPU
            stopword_mask = torch.isin(input_ids, self.stopwords_set_ids, invert=True)
            combined_mask = combined_mask * stopword_mask

        if exclude_cls_sep:
            # Cria uma máscara que é False onde os tokens são CLS ou SEP
            special_tokens_mask = (input_ids != self.cls_token_id) & (input_ids != self.sep_token_id)
            combined_mask = combined_mask * special_tokens_mask
        
        return combined_mask
    
    def _mean_pooling_exclude_cls_sep_and_stopwords(self, output, attention_mask, input_ids):
        mask = self._create_combined_mask(input_ids, attention_mask, exclude_stopwords=True, exclude_cls_sep=True)
        expanded_mask = mask.unsqueeze(-1)

        masked_embeddings = output * expanded_mask
        sum_embeddings = masked_embeddings.sum(dim=1)
        valid_token_counts = expanded_mask.sum(dim=1)
        return sum_embeddings / valid_token_counts.clamp(min=1e-9)

    def _sum_pooling_exclude_cls_sep_and_stopwords(self, output, attention_mask, input_ids):
        mask = self._create_combined_mask(input_ids, attention_mask, exclude_stopwords=True, exclude_cls_sep=True)
        expanded_mask = mask.unsqueeze(-1)
        return (output * expanded_mask).sum(dim=1)
    
    def _max_pooling_exclude_cls_sep_and_stopwords(self, output, attention_mask, input_ids):
        mask = self._create_combined_mask(input_ids, attention_mask, exclude_stopwords=True, exclude_cls_sep=True)
        expanded_mask = mask.unsqueeze(-1)
        
        masked_embeddings = output.masked_fill(expanded_mask == 0, -1e9)
        return masked_embeddings.max(dim=1)[0]
    
    def _simple_pooling(self, hidden_state, attention_mask, name_pooling, input_ids):

        match name_pooling:
             
            case "CLS":
                return hidden_state[:, 0, :]
            
            case "AVG":
                return ((hidden_state * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1).clamp(min=1e-9))
            
            case "SUM":
                return (hidden_state * attention_mask.unsqueeze(-1)).sum(dim=1)
            
            case "MAX":
                return torch.max(hidden_state.masked_fill(attention_mask.unsqueeze(-1) == 0, -1e9), dim=1)[0]
             
            case "AVG-NS":
                return self._mean_pooling_exclude_cls_sep_and_stopwords(hidden_state, attention_mask, input_ids)
            
            case "SUM-NS":
                return self._sum_pooling_exclude_cls_sep_and_stopwords(hidden_state, attention_mask, input_ids)
            
            case "MAX-NS":
                return self._max_pooling_exclude_cls_sep_and_stopwords(hidden_state, attention_mask, input_ids)
             
    def _pooling_optimized(self, hidden_state, attention_mask, name_pooling, input_ids, optimized_poolings_normal, optimized_poolings_ns):

        if name_pooling in optimized_poolings_normal:

            sum_vector = (hidden_state * attention_mask.unsqueeze(-1)).sum(dim=1)
            avg_vector = (sum_vector / attention_mask.sum(-1).unsqueeze(-1).clamp(min=1e-9))

            match name_pooling:

                case 'AVG+SUM':                    
                    return torch.cat((avg_vector, sum_vector),dim=1)
                case 'CLS+AVG+SUM':                    
                    return torch.cat((hidden_state[:, 0, :], avg_vector, sum_vector),dim=1)
                case 'AVG+SUM+MAX':                    
                    return torch.cat((avg_vector, sum_vector, torch.max(hidden_state.masked_fill(attention_mask.unsqueeze(-1) == 0, -1e9), dim=1)[0]),dim=1)
                case 'AVG+SUM+AVG-NS':                    
                    return torch.cat((avg_vector, sum_vector, self._mean_pooling_exclude_cls_sep_and_stopwords(hidden_state, attention_mask, input_ids)),dim=1)
                case 'AVG+SUM+SUM-NS':                    
                    return torch.cat((avg_vector, sum_vector, self._sum_pooling_exclude_cls_sep_and_stopwords(hidden_state, attention_mask, input_ids)),dim=1)
                case 'AVG+SUM+MAX-NS':                    
                    return torch.cat((avg_vector, sum_vector, self._max_pooling_exclude_cls_sep_and_stopwords(hidden_state, attention_mask, input_ids)),dim=1)
                
        elif name_pooling in optimized_poolings_ns:

            mask = self._create_combined_mask(input_ids, attention_mask, exclude_stopwords=True, exclude_cls_sep=True)
            expanded_mask = mask.unsqueeze(-1)
            sum_ns = (hidden_state * expanded_mask).sum(dim=1)
            valid_token_counts = expanded_mask.sum(dim=1)
            avg_ns = sum_ns / valid_token_counts.clamp(min=1e-9)

            match name_pooling:    

                case 'AVG-NS+SUM-NS':
                    return torch.cat((avg_ns, sum_ns),dim=1)
                case 'CLS+AVG-NS+SUM-NS':
                    return torch.cat((hidden_state[:, 0, :], avg_ns, sum_ns),dim=1)
                case 'AVG+AVG-NS+SUM-NS':
                    return torch.cat((((hidden_state * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1).clamp(min=1e-9)), avg_ns, sum_ns),dim=1)
                case 'SUM+AVG-NS+SUM-NS':
                    return torch.cat(((hidden_state * attention_mask.unsqueeze(-1)).sum(dim=1), avg_ns, sum_ns),dim=1)
                case 'MAX+AVG-NS+SUM-NS':
                    return torch.cat((torch.max(hidden_state.masked_fill(attention_mask.unsqueeze(-1) == 0, -1e9), dim=1)[0], avg_ns, sum_ns),dim=1)
                case 'AVG-NS+SUM-NS+MAX-NS':
                    return torch.cat((avg_ns, sum_ns, self._max_pooling_exclude_cls_sep_and_stopwords(hidden_state, attention_mask, input_ids)),dim=1)
             
    def _get_pooling_result(self, hidden_state, attention_mask, name_pooling, name_agg, input_ids):

        self.print_best_layers =  "NORMAL"

        optimized_poolings_normal = ['AVG+SUM', 'CLS+AVG+SUM', 'AVG+SUM+MAX', 'AVG+SUM+AVG-NS', 'AVG+SUM+SUM-NS', 'AVG+SUM+MAX-NS']
        optimized_poolings_ns = ['AVG-NS+SUM-NS', 'CLS+AVG-NS+SUM-NS', 'AVG+AVG-NS+SUM-NS', 'SUM+AVG-NS+SUM-NS', 'MAX+AVG-NS+SUM-NS', 'AVG-NS+SUM-NS+MAX-NS']
        total_optimized_poolings = optimized_poolings_normal + optimized_poolings_ns

        if name_pooling in total_optimized_poolings:

            return self._pooling_optimized(hidden_state, attention_mask, name_pooling, input_ids, optimized_poolings_normal, optimized_poolings_ns)
        
        else:
            name_pooling_split = name_pooling.split('+')

            match len(name_pooling_split):

                case 1:
                    return self._simple_pooling(hidden_state, attention_mask, name_pooling_split[0], input_ids)
                
                case 2:
                    return torch.cat(
                    (
                        self._simple_pooling(hidden_state, attention_mask, name_pooling_split[0], input_ids),
                        self._simple_pooling(hidden_state, attention_mask, name_pooling_split[1], input_ids)
                    ), 
                    dim=1)
                
                case 3:
                    return torch.cat(
                    (
                        self._simple_pooling(hidden_state, attention_mask, name_pooling_split[0], input_ids),
                        self._simple_pooling(hidden_state, attention_mask, name_pooling_split[1], input_ids),
                        self._simple_pooling(hidden_state, attention_mask, name_pooling_split[2], input_ids)
                    ), 
                    dim=1)
                
                case 4:
                    return torch.cat(
                    (
                        self._simple_pooling(hidden_state, attention_mask, name_pooling_split[0], input_ids),
                        self._simple_pooling(hidden_state, attention_mask, name_pooling_split[1], input_ids),
                        self._simple_pooling(hidden_state, attention_mask, name_pooling_split[2], input_ids),
                        self._simple_pooling(hidden_state, attention_mask, name_pooling_split[3], input_ids)
                    ), 
                    dim=1)

    def get_best_pooling(self, hidden_states, attention_mask, name_pooling):

        match name_pooling:

            case "AVG7+AVG10":
                avg7 = ((hidden_states[7] * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1).clamp(min=1e-9))
                avg10 = ((hidden_states[10] * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1).clamp(min=1e-9))
                return torch.cat((avg7, avg10), dim=1)
            
            case "AVG7+AVG10+AVG9":
                avg7 = ((hidden_states[7] * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1).clamp(min=1e-9))
                avg10 = ((hidden_states[10] * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1).clamp(min=1e-9))
                avg9 = ((hidden_states[9] * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1).clamp(min=1e-9))
                return torch.cat((avg7, avg10, avg9), dim=1)
            
            case "AVG9+AVG10":
                avg9 = ((hidden_states[9] * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1).clamp(min=1e-9))
                avg10 = ((hidden_states[10] * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1).clamp(min=1e-9))
                return torch.cat((avg9, avg10), dim=1)
            
            case "AVG9+AVG10+AVG11":
                avg9 = ((hidden_states[9] * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1).clamp(min=1e-9))
                avg10 = ((hidden_states[10] * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1).clamp(min=1e-9))
                avg11 = ((hidden_states[11] * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1).clamp(min=1e-9))
                return torch.cat((avg9, avg10, avg11), dim=1)
            
            case "AVGdoSUM7e10":
                SUM_7e10 = torch.stack([hidden_states[i] for i in [7,10]], dim=0).sum(dim=0)
                return ((SUM_7e10 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1).clamp(min=1e-9))
            
            case "AVGdoSUM7e10e9":
                SUM_7e10e9 = torch.stack([hidden_states[i] for i in [7,10,9]], dim=0).sum(dim=0)
                return ((SUM_7e10e9 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1).clamp(min=1e-9))
            
            case "AVGdoSUM9e10":
                SUM_9e10 = torch.stack([hidden_states[i] for i in [9,10]], dim=0).sum(dim=0)
                return ((SUM_9e10 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1).clamp(min=1e-9))
            
            case "AVGdoSUM9e10e11":
                SUM_9e10e11 = torch.stack([hidden_states[i] for i in [9,10,11]], dim=0).sum(dim=0)
                return ((SUM_9e10e11 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1).clamp(min=1e-9))  
                      

    def _apply_pooling(self, output, attention_mask, input_ids):  

        hidden_states = output.hidden_states
        name_pooling = self.pooling_strategy.split("_")[0]
        name_agg = self.pooling_strategy.split("_")[-1]

        if name_agg == 'BEST':
            return self.get_best_pooling(hidden_states, attention_mask, name_pooling)

        if name_agg.startswith("LYR"):
            layer_idx = int(name_agg.split('-')[-1])   
            LYR_hidden =  hidden_states[layer_idx]            
            return self._get_pooling_result(LYR_hidden, attention_mask, name_pooling, "LYR", input_ids)        
        else:        
            name_agg_type = name_agg.split("-")[0]
            agg_initial_layer = int(name_agg.split("-")[1])
            agg_final_layer = int(name_agg.split("-")[2])
            
            match name_agg_type:  

                case "SUM":
                    return self._get_pooling_result(torch.stack(hidden_states[agg_initial_layer:agg_final_layer+1], dim=0).sum(dim=0), attention_mask, name_pooling, name_agg, input_ids)
                    
                case "AVG":
                    return self._get_pooling_result(torch.stack(hidden_states[agg_initial_layer:agg_final_layer+1], dim=0).mean(dim=0), attention_mask, name_pooling, name_agg, input_ids)                  

def run_senteval(model_name, tasks, args, type_task):
    results_general = {}

    device = functions_code.get_device()
    print(f"\nExecuting Device: {device}")
    
    encoder = SentenceEncoder(model_name, device)
    if args.agg_layers[0] == 'BEST':       
        pooling_strategies = ['AVG7+AVG10_BEST', 'AVG7+AVG10+AVG9_BEST', 'AVG9+AVG10_BEST', 'AVG9+AVG10+AVG11_BEST', 
                              'AVGdoSUM7e10_BEST', 'AVGdoSUM7e10e9_BEST', 'AVGdoSUM9e10_BEST', 'AVGdoSUM9e10e11_BEST']
        list_poolings = []
        list_layers = []
    else:
        pooling_strategies, list_poolings, list_layers = functions_code.strategies_pooling_list(args, encoder.qtd_layers)

    #GET ALL EMBEDDINGS
    print("LISTA DE POOLINGS: ", list_poolings)
    print("LISTA DE LAYERS: ", list_layers)
   
    tempos = []  
    for pooling in pooling_strategies:
        encoder.pooling_strategy = pooling
        print(f"Running: Model={encoder.name_model}, Pooling={encoder.pooling_strategy}")
        if type_task == 'cl':
            senteval_params = {
                'task_path': 'data',
                'usepytorch': False,
                'kfold': args.kfold,
                'classifier': {
                    'nhid': args.nhid,
                    'optim': args.optim,
                    'batch_size': args.batch,
                    'tenacity': 5,
                    'epoch_size': args.epochs
                },
                'encoder': encoder
            }
        else:
             senteval_params = {
                'task_path': 'data',
                'usepytorch': False,
                'kfold': 10,
                'encoder': encoder
            }
        se = senteval.engine.SE(senteval_params, functions_code.batcher)

         # --- NOVO: Medição de tempo ---
        start_time = time.time()
        results_general[pooling] = se.eval(tasks)
        end_time = time.time()
        elapsed_time = (end_time - start_time) / 60
        # --- Fim da medição ---

        tempos.append(elapsed_time)
        media_tempo = np.mean(tempos)
        tempo_faltante = (media_tempo * (len(pooling_strategies) - len(tempos))) / 60
        dias_faltante = tempo_faltante / 24

        results_general[pooling]['out_vec_size'] = encoder.size_embedding
        results_general[pooling]['qtd_layers'] = encoder.qtd_layers
        results_general[pooling]['best_layers'] = encoder.print_best_layers
        print(f"Output vector size: {encoder.size_embedding}")
        print(f"BEST LAYERS: {encoder.print_best_layers}")
        print(f"--> Time for this run: {elapsed_time:.2f} minutes")
        print("Progress: " + str(len(tempos)) + '/' + str(len(pooling_strategies)))
        print(f"--> Tempo Faltante Estimado: {tempo_faltante:.2f} horas")
        print(f"--> Dias Faltante Estimado: {dias_faltante:.2f} dias")
                              
    return results_general

def tasks_run(args, main_path, filename_task, tasks_list, type_task):
    path_created = main_path + '/' + filename_task
    os.makedirs(path_created, exist_ok=True)

    logging.basicConfig(
        filename=path_created + '/' + filename_task + '_log.txt',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    results_data = []

    config_path = os.path.join(path_created, f"config_{args.save_dir}.json")
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=4)

    for model_name in args.models:
        print(f"\nExecuting Model: {model_name}")
        results = run_senteval(model_name, tasks_list, args, type_task)
        for pooling, res in results.items():
            if type_task == 'cl':
                dict_results = [res.get(task, {}) for task in tasks_list]
            elif type_task == 'si':
                dict_results = [res.get(task, {}).get('all', 0) for task in tasks_list[:5]] + [res.get(task, {}) for task in tasks_list[-2:]]
            
            results_data.append({
                "model": model_name,
                "pooling": pooling,
                "out_vec_size": res.get('out_vec_size'),
                "best_layers": res.get('best_layers'),
                "epochs": args.epochs,
                "nhid": args.nhid,
                "qtd_layers": res.get('qtd_layers'),              
                **{task: dict_results[i] for i, task in enumerate(tasks_list)}
            })
        
            final_df1 = pd.DataFrame(results_data)
            final_df1.to_csv(path_created + '/' + filename_task + '_intermediate.csv', index=False)
                    
    final_df = pd.DataFrame(results_data)
    final_df.to_csv(path_created + '/' + filename_task + '.csv', index=False)

    os.remove(path_created + '/' + filename_task + '_intermediate.csv')

    functions_code_testes.main_evaluate(final_df, type_task, path_created, filename_task, tasks_list)

def main(args):

    args.models = args.models.split(",")
    args.poolings = args.poolings.split(",")
    args.agg_layers = args.agg_layers.split(",")  

    main_path = '../results_pooling_paper/' + str(args.save_dir)

    initial_layer_args_print = args.initial_layer if args.initial_layer is not None else "default"
    final_layer_args_print = args.final_layer if args.final_layer is not None else "default"

    #filename_task = ('_models_' + '&'.join([st for st in args.models]) + 
    #                 '_epochs_' + str(args.epochs) + 
    #                 '_batch_' + str(args.batch) +
    #                 '_nhid_' + str(args.nhid) + 
    #                 '_initiallayer_' + str(initial_layer_args_print) + 
    #                 '_finallayer_' + str(final_layer_args_print) +
    #                 '_pooling_' + '&'.join([st for st in args.poolings]) + 
    #                 '_agglayers_' + '&'.join([st for st in args.agg_layers])
    #                )

    filename_task = str(args.save_dir)
    
    if args.task_type == "classification":      
        filename_cl = "cl" + filename_task
        classification_tasks = ['MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC']        
        classification_tasks = args.tasks.split(",") if args.tasks is not None else classification_tasks
        tasks_run(args, main_path, filename_cl, classification_tasks, 'cl')

    elif args.task_type == "similarity":
        filename_si = "si" + filename_task
        similarity_tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']
        similarity_tasks = args.tasks.split(",") if args.tasks is not None else similarity_tasks
        tasks_run(args, main_path, filename_si, similarity_tasks, 'si')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SentEval Experiments")

    parser.add_argument("--task_type", type=str, default='classification', choices=['classification', 'similarity'], help="Tipo de tarefa (classification ou similarity)")
    parser.add_argument("--models", type=str, required=True, help="Modelos separados por vírgula (sem espaços)")
    parser.add_argument("--epochs", type=int, default=4, help="Número máximo de épocas do classificador linear")
    parser.add_argument("--batch", type=int, default=64, help="Batch Size do classificador")
    parser.add_argument("--kfold", type=int, default=10, help="KFold para validação")
    parser.add_argument("--optim", type=str, default='adam', help="otimizador do classificador")
    parser.add_argument("--nhid", type=int, default=0, help="Numero de camadas ocultas (0 = Logistic Regression, 1 ou mais = MLP)")
    parser.add_argument("--initial_layer", default=12, type=int, help="Camada inicial para execução dos experimentos (default metade superior)")
    parser.add_argument("--final_layer", type=int, default=12, help="Camada inicial para execução dos experimentos (default metade superior)")
    parser.add_argument("--poolings", type=str, required=True, default="all", help="Poolings separados por virgula (sem espacos) ou simple, simple-ns, two, three")
    parser.add_argument("--agg_layers", type=str, required=True, default="ALL", help="agg layers separados por virgula (sem espacos)")
    parser.add_argument("--tasks", type=str, help="tasks separados por virgula (sem espacos)")
    parser.add_argument("--save_dir", type=str, help="tasks separados por virgula (sem espacos)")

    args = parser.parse_args()
    original_save_dir = args.save_dir

    main(args)