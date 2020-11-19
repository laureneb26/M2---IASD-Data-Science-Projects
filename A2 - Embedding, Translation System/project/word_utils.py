"""
Fonctions utiles pour les différents algorithmes
"""


import io
import numpy as np
import math
from scipy import spatial

def load_vector_file(fname, max = 10**5):
    """
    Renvoie sous la forme d'un dictionnaire le contenu d'un fichier .vec avec comme clé le mot et comme valeur sa représentation vectorielle

    args:
        fname (string): chemin vers le fichier .vec
        max (int): nombre maximum de mot à charger dans le dictionnaire
    output:
        dict
    """

    fichier = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    _, _ = [int(elem) for elem in fichier.readline().split()]
    data = {}
    i = 0
    for line in fichier:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = [float(elem) for elem in tokens[1:]]
        i+=1
        if i == max:
          break
    return data

def load_dict_file(fname, max = 10**5):
    """
    Renvoie sous la forme d'un dictionnaire le contenu d'un fichier .vec représentant un dictionnaire de traduction avec comme clé le mot dans la langue d'origine et comme valeur sa traduction dans la langue cible

    args:
        fname (string): chemin vers le fichier .vec
        max (int): nombre maximum de mot à charger dans le dictionnaire
    output:
        dict
    """

    data = {}
    fichier = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    fichier.readline()
    i = 0
    for line in fichier:
        tokens = line.rstrip().split(' ')
        # print(read)
        if len(tokens) > 1:
            key = tokens[0]
            value = tokens[1]
            data[key] = value
            i += 1
            if i == max:
                break
    return data

def get_translation_array(dict_trad, dict_key, dict_value, verbose = False):
    """
    Renvoie sous la forme d'un dictionnaire le contenu d'un fichier .vec représentant un dictionnaire de traduction avec comme clé le mot dans la langue d'origine et comme valeur sa traduction dans la langue cible

    args:
        dict_trad (dict): dictionnaire de traduction
        dict_trad (dict): dictionnaire d'origine
        dict_trad (dict): dictionnaire cible
        verbose (bool): Si True, print le nombre d'éléments non pris en compte
    output:
        X, Y, words_X, words_Y
    """

    liste_key, liste_value, words_X, words_Y = [], [], [], []
    elems = list(dict_trad.items())

    ##Pour infos
    not_in_translation = 0

    for elem in elems:
        key, value = None, None
        try: 
            key = dict_key[elem[0]]
            value = dict_value[elem[1]]
        except:
            not_in_translation +=1
            pass
        if key != None and value != None:
            liste_key += [key]
            liste_value += [value]
            words_X += [elem[0]]
            words_Y += [elem[1]]
    if verbose:
        print("not in translation: " + str(not_in_translation))
    return np.array(liste_key), np.array(liste_value), words_X, words_Y

def normalisation(array):
    """
    Renvoie une matrice normalisée dont les colonnes (C'_i) sont obtenues à partir des colonnes (C_i) de la manière suivante: C'_i = [C'_i - Mean(C'_i)]/Std(C'_i)
    args:
        array (NumpyArray): Array à normaliser
    output:
        NumpyArray: Array normalisé
    """
    n, p = np.shape(array)
    normalized_array = np.copy(array)
    for i in range(n):
        ligne_i = normalized_array[i,:]
        ligne_i = (ligne_i-np.mean(ligne_i))/np.std(ligne_i)
    return normalized_array

def get_W(X_train, Y_train, alpha = 1, init = 2):
    """
    Renvoie la matrice W qui maximise la quantité sum_i [W@x_i].T@z_i
    Elle est obtenue par gradient descent avec la règle W = W + alpha*DeltaW
    où DeltaW = sum_i [x_i@z_i.T]

    args:
        X (NumpyArray): Matrice des x_i
        Y (NumpyArray): Matrice des y_i
        alpha (float): learning rate
        init (int [|0,2|]): Initialisation de W {0: à 0, 1: uniform, 2: gaussian}
    output:
        NumpyArray: Matrice W
        i_f (list[int]): Indices
        val_f (list[float]): Valeurs de la fonction objectif
    """
    f = lambda W, x_i, y_i : np.linalg.norm(W@x_i - y_i)**2
    n, m = X_train.shape
    i_f = []
    val_f = []
    if init == 0:
        W = np.zeros((m,m))
    elif init == 1:
        W = np.random.uniform(low = -0.2, high = 0.2, size = (m,m))
    else:
        W = np.random.normal(loc = 0, scale = 0.1, size = (m,m))
    for i in range(n):
        # print(i)
        x_i = X_train[i].reshape((-1,1))
        y_i = Y_train[i].reshape((-1,1))
        delta_W = y_i@(x_i.T)
        W += alpha*delta_W
        if i %1000 == 0:
            i_f += [i]
            u, _ , v = np.linalg.svd(W)
            W_tilde = u@np.eye(m)@v
            val_f += [sum(f(W_tilde, X_train[i,:].reshape((m,1)), Y_train[i,:].reshape((m,1))) for i in range(len(X_train)))/len(X_train)]
    u, _ , v = np.linalg.svd(W)
    W = u@np.eye(m)@v
    return W, i_f, val_f

def recherche_minimum(W, X, Y, words_X, words_Y, dist = 1, verbose = True):
    """
    Trouve la meilleure traduction au sens de la norme entre les vecteurs pour N mots, renvoie également l'accuracy
    args:
        W (NumpyArray): Matrice W qui maximise la quantité sum_i [W@x_i].T@z_i
        X (NumpyArray): Matrice des x_i
        Y (NumpyArray): Matrice des y_i
        words_X (list[string]): Mot dans la langue de départ
        words_Y (list[string]): Mot dans la langue cible
        dist {0,1}: {0:linalg.norm, 1:cosine similarity}
        N (int): Nombre de mots à traduire
        verbose (Bool): Print l'avancement
    output:
        resutats (list[string]): Mot et leur traduction
        acc_train (float): Valeur de l'accuracy du train
    """
    n, _ = np.shape(X)
    m, _ = np.shape(Y)
    resultats = []
    i, acc_train = 0, 0
    distance = np.zeros((1,m))
    
    for i in range(n):
        if verbose:
            print(i)
        for j in range(m):
            if dist == 0:
                distance[0,j] = np.linalg.norm(W@X[i] - Y[j])
            else:
                distance[0,j] = spatial.distance.cosine(W@X[i],Y[j])

        idx_min=np.argmin(distance)
        if idx_min == i:
            acc_train += 1
        resultats += [(words_X[i], words_Y[idx_min])]

    #Calcul Accuracy
    acc_train=acc_train/n*100
    return resultats, acc_train

def traducteur(dict_key, dict_value, dict_trad, norm = True, dist = 1,alpha = 1, N = 100, init = 0, verbose = True,valid=True,validation_split=0.9):
    """
    Trouve la meilleure traduction au sens de la norme entre les vecteurs pour N mots, renvoie également l'accuracy
    args:
        dict_trad (dict): dictionnaire de traduction
        dict_trad (dict): dictionnaire d'origine
        dict_trad (dict): dictionnaire cible
        norm (Bool): Normalisation des données
        dist {0,1}: {0:linalg.norm, 1:cosine similarity}
        alpha (float): learning rate
        N (int): Nombre de mots à traduire
        init (int [|0,2|]): Initialisation de W {0: à 0, 1: uniform, 2: gaussian}
        verbose (Bool): Print l'avancement
    output:
        resutats (list[string]): Mot et leur traduction
        acc_train (float): Valeur de l'accuracy du train
    """
    vvkey, vvalue, words_X, words_Y = get_translation_array(dict_trad,dict_key,dict_value)
    if norm:
        vvkey = normalisation(vvkey)
        vvalue = normalisation(vvalue)
    
    if valid:
        N_split=math.floor(N*validation_split)
        X_train=vvkey[0:N_split,:]
        Y_train=vvalue[0:N_split,:]
        X_test=vvkey[N_split:N,:]
        Y_test=vvalue[N_split:N,:]
        words_X_train=words_X[0:N_split]
        words_Y_train=words_Y[0:N_split]
        words_X_test=words_X[N_split:N]
        words_Y_test=words_Y[N_split:N]
    else :
        X_train=vvkey[0:N,:]
        Y_train=vvalue[0:N,:]
        words_X_train=words_X[0:N]
        words_Y_train=words_Y[0:N]
    W = get_W(X_train, Y_train, alpha, init)
    resultats, acc_train = recherche_minimum(W, X_train, Y_train, words_X_train, words_Y_train, dist, verbose)
    if valid:
        resultats, acc_test = recherche_minimum(W, X_test, Y_test, words_X_test, words_Y_test, dist, verbose)
    acc=[0,0]
    acc[0]=acc_train
    if valid:
        acc[1]=acc_test
    return resultats, acc


def GD_W_not_const(X,Y, alpha = 1e-3, nb_max = -1):
    """
    Renvoie la matrice W qui maximise la quantité sum_i ||W@x_i -z_i||^2
    Elle est obtenue par gradient descent avec la règle W = W - alpha*Gradf_i
    où Gradf_i = (W@x_i - z_i)@x_i^T

    args:
        X (NumpyArray): Matrice des x_i
        Y (NumpyArray): Matrice des y_i
        alpha (float): learning rate
        nb_max (int): Nombre maximum d'exemples à utiliser (-1 = tous)

    output:
        W (NumpyArray): Matrice W
        i_f (list[int]): Indices
        val_f (list[float]): Valeurs de la fonction objectif
    """
    Gradf = lambda W, x_i, y_i: 2*(W@x_i - y_i)@(x_i.T)
    n, m = X.shape
    f = lambda W, x_i, y_i : np.linalg.norm(W@x_i - y_i)**2
    W = np.random.normal(loc = 0, scale = 0.1, size = (m,m))
    i_f = []
    val_f = []
    if nb_max == -1:
        nb_ex = n
    else:
        nb_ex = nb_max
    for i in range(nb_ex):
        x_i, y_i = X[i,:].reshape((m,1)), Y[i,:].reshape((m,1))
        W = W - alpha*Gradf(W, x_i, y_i)
        if i%1000 ==0:
            i_f += [i]
            val_f += [sum(f(W, X[i,:].reshape((m,1)), Y[i,:].reshape((m,1))) for i in range(len(X)))/len(X)]
    return W, i_f, val_f