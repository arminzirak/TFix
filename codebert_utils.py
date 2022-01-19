from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

code_bert_model = None
code_bert_tokenizer = None


def load(codebert_address):
    global code_bert_tokenizer, code_bert_model
    code_bert_model = AutoModel.from_pretrained(codebert_address)
    code_bert_model.to('cuda')

    code_bert_tokenizer = AutoTokenizer.from_pretrained(codebert_address)
    print('codebert is loaded')


def code_to_vec(code):
    if not code_bert_model or not code_bert_tokenizer:
        raise Exception('codebert is not loaded')
    code_tokens = code_bert_tokenizer.tokenize(code)
    tokens = [code_bert_tokenizer.cls_token] + code_tokens + [code_bert_tokenizer.sep_token]
    tokens_ids = code_bert_tokenizer.convert_tokens_to_ids(tokens)
    context_embeddings = code_bert_model(torch.tensor(tokens_ids).to('cuda')[None, :])[0]
    return context_embeddings[0][0].cpu().detach().numpy()


def vec_distance(code1, code2):
    n_code1 = code1 / np.linalg.norm(code1)
    n_code2 = code1 / np.linalg.norm(code2)
    return np.linalg.norm(n_code1 - n_code2)


def unload():
    del code_bert_tokenizer
    del code_bert_model