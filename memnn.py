import torch
import numpy as np


def create_dict(data_files):
    '''
    Créé le dictionnaire W
    '''
    dictionary = {}
    number = 1

    for file in data_files:
        with open(file, 'r') as f:
            for line in f:
                number, dictionary = parse_line_add_dict(number, line, dictionary)

    return dictionary

def parse_line_add_dict(number, line, dictionnary):
    '''
    Implémente le dictionnaire W pour une ligne du fichier
    '''
    words = line.split()
    if words[-1][-1] == '.':
        for i in range(1, len(words)):
            word = words[i]
            if word[-1] == '.':
                word = word[:-1]
            if word not in dictionary:
                dictionary[word] = number
                number += 1
    else:
        for i in range(1, len(words) - 1):
            word = words[i]
            if word[-1] == '?':
                word = word[:-1]
            if word not in dictionary:
                dictionary[word] = number
                number += 1

    return number, dictionary


def I(x, dictionary, atype):
    '''
    Vectorise une phrase
    '''
    words = x.split()
    feature_rep = np.zeros((len(dictionary), 1), dtype=np.float64)
    
    for w in words:
        if w[-1] == '?' or w[-1] == '.':
            w = w[:-1]
        
        onehot = word_to_onehot(w, dictionary)
        feature_rep = feature_rep + onehot
    
    feature_rep = feature_rep.astype(atype)
    return feature_rep

def word_to_onehot(word, dictionary):
    '''
    Retourne word sous sa forme onehot
    '''
    onehot = np.zeros((len(dictionary), 1), dtype=np.float64)
    
    for w in dictionary.keys():
        if w == word:
            onehot[dictionary[w], 0] = 1.0
            break
    
    return onehot


def G(feature_rep, memory):
    memory.append(feature_rep)
    return memory


def phix(feature_rep_list, atype):
    mapped_length = 3 * len(feature_rep_list[0])
    mapped = np.zeros((mapped_length, 1), dtype=np.float64)
    
    for i, feature_rep in enumerate(feature_rep_list, start=1):
        for j, value in enumerate(feature_rep, start=1):
            if i == 1:
                mapped[j - 1] = value
            else:
                mapped[len(feature_rep) + j - 1] = value
    
    mapped = mapped.astype(atype)
    return mapped

def phiy(feature_rep, atype):
    mapped_length = 3 * len(feature_rep)
    mapped = np.zeros((mapped_length, 1), dtype=np.float64)
    
    for i, value in enumerate(feature_rep, start=1):
        mapped[2 * len(feature_rep) + i - 1] = value
    
    mapped = mapped.astype(atype)
    return mapped

def s(x_feature_rep_list, y_feature_rep, u, atype):
    phi_y = phiy(y_feature_rep, atype)
    phi_x = phix(x_feature_rep_list, atype)
    score = np.sum(phi_x.T @ u.T @ u @ phi_y)
    return score


def O(x_feature_rep, memory, uo, atype):
    x_feature_rep_list = [x_feature_rep]
    score_dict1 = so(x_feature_rep_list, memory, uo, atype)
    o1 = max(score_dict1, key=score_dict1.get)
    mo1 = memory[o1]
    return [x_feature_rep, mo1]

def so(x_feature_rep_list, memory, uo, atype):
    score_dict = {}
    
    for i, mem in enumerate(memory, start=1):
        score = s(x_feature_rep_list, mem, uo, atype)
        score_dict[score] = i
    
    return score_dict


def R(input_list, vocab_dict, ur, atype):
    score_dict = sr(input_list, vocab_dict, ur, atype)
    answer = max(score_dict, key=score_dict.get)
    return answer

def sr(x_feature_rep_list, vocab_dict, ur, atype):
    score_dict = {}
    
    for k in vocab_dict.keys():
        y_feature_rep = word_to_onehot(k, vocab_dict)
        score = s(x_feature_rep_list, y_feature_rep, ur, atype)
        score_dict[score] = k
    
    return score_dict



class MemNN(torch.nn.Module):
    def __init__(self):
        super.__init__()

        #reste du modele


    def forward():
        pass