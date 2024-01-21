import torch
import numpy as np
import nltk
import string
import sklearn

nltk.download('stopwords')


def create_dict(data_files):
    '''
    Créé le dictionnaire W
    '''
    dictionary = {}
    number = 0

    for file in data_files:
        with open(file, 'r') as f:
            for line in f:
                number, dictionary = parse_line_add_dict(number, line, dictionary)

    return dictionary


def parse_line_add_dict(number, line, dictionnary):
    '''
    Implémente le dictionnaire W pour une ligne du fichier
    '''
    tokenized_line = nltk.tokenize.word_tokenize(line)
    tokenized_line = [word for word in tokenized_line if word not in string.punctuation]
    porter = nltk.stem.PorterStemmer()
    lemmed_sentence = [porter.stem(mot) for mot in tokenized_line]
    stop_words = nltk.corpus.stopwords.words('english')
    unstopworded_sentence = [mot for mot in lemmed_sentence if mot.lower() not in stop_words]
    unstopworded_sentence_sans_chiffre = [element for element in unstopworded_sentence if not str(element).isdigit()]
    
    for mot in unstopworded_sentence_sans_chiffre :
        if mot not in dictionnary:
            dictionnary[mot] = number
            number +=1
    return number, dictionnary


def init_weights(atype, feature_space, embedding_dimension, winit):
    weights = {}

    u0 = winit * torch.randn(embedding_dimension, feature_space)
    ur = winit * torch.randn(embedding_dimension, feature_space)
    weights['u0'] = u0
    weights['ur'] = ur

    # for k in weights.keys():
    #     weights[k] = weights[k].to(atype)

    return weights


def I(x, vocab, atype):
    """
    Convertit une séquence de mots en représentation one-hot.

    Args:
    - x (str): La séquence de mots à convertir.
    - vocab (dict): Le vocabulaire où chaque mot unique est associé à un index.

    Returns:
    - feature_rep (np.ndarray): La représentation one-hot de la séquence.
    """
    words = x.split()
    feature_rep = torch.zeros(len(vocab), dtype=atype)

    for w in words:
        w = w.rstrip('?.')
        onehot = word2OneHot(w, vocab)
        feature_rep += onehot

    return feature_rep


def word2OneHot(word, vocab):
    """
    Convertit un mot en représentation one-hot.

    Args:
    - word (str): Le mot à convertir.
    - vocab (dict): Le vocabulaire où chaque mot unique est associé à un index.

    Returns:
    - onehot (np.ndarray): La représentation one-hot du mot.
    """
    onehot = np.zeros(len(vocab))
    if word in vocab:
        onehot[vocab[word]] = 1.0

    return onehot.astype(np.float32)



def G(one_hot_encoding, memory):
    '''
    Ajoute element à la mémoire
    '''
    memory.append(one_hot_encoding)
    return memory


def O(x_feature_rep, memory, u0, atype):
    x_feature_rep_list = [x_feature_rep]
    score_dict_1 = so(x_feature_rep_list, memory, u0, atype)
    o1 = max(score_dict_1, key=score_dict_1.get)
    mo1 = memory[o1]
    return [x_feature_rep, mo1]

def phix(feature_rep_list, atype):
    mapped = torch.zeros(3 * len(feature_rep_list[0]), 1, dtype=atype)
    for i, feature_rep in enumerate(feature_rep_list):
        for j, value in enumerate(feature_rep):
            if i == 1:
                mapped[j] = value
            else:
                mapped[len(feature_rep) + j] = value
    return mapped

def phiy(feature_rep, atype):
    mapped = torch.zeros(3 * len(feature_rep), 1, dtype=atype)
    for i, value in enumerate(feature_rep):
        mapped[2 * len(feature_rep) + i] = torch.tensor(value, dtype=atype)
    return mapped

def s(x_feature_rep_list, y_feature_rep, u, atype):
    phi_y = phiy(y_feature_rep, atype)
    phi_x = phix(x_feature_rep_list, atype)
    score = torch.sum(torch.mm(torch.mm(phi_x.t(), u.t()), u) * phi_y)
    return score.item()

def so(x_feature_rep_list, memory, u0, atype):
    score_dict = {}
    for i, memory_item in enumerate(memory):
        score = s(x_feature_rep_list, memory_item, u0, atype)
        score_dict[score] = i
    return score_dict


def R(input_list, vocab_dict, ur, atype):
    score_dict = sr(input_list, vocab_dict, ur, atype)
    answer = max(score_dict, key=score_dict.get)
    return answer

def sr(x_feature_rep_list, vocab_dict, ur, atype):
    score_dict = {}
    for k in vocab_dict.keys():
        y_feature_rep = word2OneHot(k, vocab_dict)
        score = s(x_feature_rep_list, y_feature_rep, ur, atype)
        score_dict[score] = k
    return score_dict


def marginRankingLoss(comb, x_feature_rep, memory, vocab_dict, gold_labels, margin, atype):
    u0 = comb[0]
    ur = comb[1]
    total_loss = 0
    m1_loss = 0
    r_loss = 0

    correct_m1 = gold_labels[0]
    correct_r = gold_labels[1]

    input_1 = [x_feature_rep]
    for i, memory_item in enumerate(memory):
        if not np.array_equal(memory_item, correct_m1):
            m1l = max(0, margin - s(input_1, correct_m1, u0, atype) + s(input_1, memory_item, u0, atype))
            m1_loss += m1l

    correct_r_feature_rep = word2OneHot(correct_r, vocab_dict)
    input_r = [x_feature_rep, correct_m1]
    for k in vocab_dict.keys():
        # if not np.array_equal(k, correct_r):
        if k != correct_r:
            k_feature_rep = word2OneHot(k, vocab_dict)
            rl = max(0, margin - s(input_r, correct_r_feature_rep, ur, atype) + s(input_r, k_feature_rep, ur, atype))
            r_loss += rl

    total_loss = m1_loss + r_loss
    return total_loss


def train(data_file, u0, ur, vocab_dict, lr, margin, atype):
    total_loss = 0
    numq = 0
    memory = reset_memory()

    with open(data_file, 'r') as f:
        for line in f:
            words = line.split()

            if words[-1][-1] == '.':
                line_number = int(words[0])
                sentence = ' '.join(words[1:])
                if line_number == 1:
                    memory = reset_memory()
                sentence_feature_rep = I(sentence, vocab_dict, atype)
                G(sentence_feature_rep, memory)
            else:
                line_number = int(words[0])
                question = ' '.join(words[1:-2])
                question_feature_rep = I(question, vocab_dict, atype)
                G(question_feature_rep, memory)

                correct_r = words[-2]
                correct_m1_index = int(words[-1])
                correct_m1 = memory[correct_m1_index]

                gold_labels = [correct_m1, correct_r]
                comb = [u0, ur]
                loss = marginRankingLoss(comb, question_feature_rep, memory, vocab_dict, gold_labels, margin, atype)

                loss.backward()

                u0 -= lr * u0.grad
                ur -= lr * ur.grad

                total_loss += loss
                numq += 1

    avg_loss = total_loss / numq
    return avg_loss

def reset_memory():
    return []


def trainingAccuracy(data_file, uo, ur, vocab_dict, atype):
    numsup = 0
    numcorr = 0
    numq = 0
    memory = reset_memory()
    with open(data_file, 'r') as f:
        for line in f:
            words = line.split()
            if words[-1][-1] == '.':
                line_number = int(words[0])
                sentence = ' '.join(words[1:])
                if line_number == 1:
                    memory = reset_memory()
                sentence_feature_rep = I(sentence, vocab_dict, atype)
                G(sentence_feature_rep, memory)
            else:
                line_number = int(words[0])
                question = ' '.join(words[1:-2])
                question_feature_rep = I(question, vocab_dict, atype)
                G(question_feature_rep, memory)

                correct_r = words[-2]

                correct_m1_index = int(words[-1])
                correct_m1 = memory[correct_m1_index]

                output = O(question_feature_rep, memory, uo, atype)
                if correct_m1 in output:
                    numsup += 1

                response = R([question_feature_rep, correct_m1], vocab_dict, ur, atype)
                if response == correct_r:
                    numcorr += 1

                numq += 1

    output_accuracy = numsup / numq * 100 if numq > 0 else 0
    response_accuracy = numcorr / numq * 100 if numq > 0 else 0
    return output_accuracy, response_accuracy


def test(data_file, uo, ur, vocab_dict, margin, atype):
    numcorr = 0
    total_loss = 0
    numq = 0
    memory = reset_memory()

    with open(data_file, 'r') as f:
        for line in f:
            words = line.split()
            if words[-1][-1] == '.':
                line_number = int(words[0])
                sentence = ' '.join(words[1:])

                for i in range(2, len(words)):
                    if words[i][-1] == '?' or words[i][-1] == '.':
                        words[i] = words[i][0:-1]
                    sentence = sentence + " " + words[i]

                if line_number == 1:
                    memory = reset_memory()

                sentence_feature_rep = I(sentence, vocab_dict, atype)
                G(sentence_feature_rep, memory)
            else:
                line_number = int(words[0])
                question = ' '.join(words[1:-2])

                for i in range(2, len(words) - 2):
                    if words[i][-1] == '?' or words[i][-1] == '.':
                        words[i] = words[i][0:-1]
                    question = question + " " + words[i]

                question_feature_rep = I(question, vocab_dict, atype)
                G(question_feature_rep, memory)

                correct_r = words[-2]

                correct_m1_index = int(words[-1])
                correct_m1 = memory[correct_m1_index]

                gold_labels = [correct_m1, correct_r]
                comb = [uo, ur]
                loss = marginRankingLoss(comb, question_feature_rep, memory, vocab_dict, gold_labels, margin, atype)

                response = answer(question_feature_rep, memory, vocab_dict, uo, ur, atype)
                if response == correct_r:
                    numcorr += 1

                total_loss += loss
                numq += 1

    test_accuracy = numcorr / numq * 100 if numq > 0 else 0
    avg_loss = total_loss / numq if numq > 0 else 0
    return avg_loss, test_accuracy

def answer(x_feature_rep, memory, vocab_dict, uo, ur, atype):
    output = O(x_feature_rep, memory, uo, atype)
    answer = R(output, vocab_dict, ur, atype)
    return answer

def carre(x):
    return(x**2)