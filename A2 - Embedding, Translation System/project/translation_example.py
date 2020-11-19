import numpy as np
import matplotlib.pyplot as plt
from word_utils import *


##On commence par load les embeddings
dict_fr = load_vector_file("wiki.fr.vec")
dict_en = load_vector_file("wiki.en.vec")
dict_fr_en = load_dict_file(fname = "fr-en.txt")


##On récupère des embeddings précédents les différents vecteurs des mots, X pour le langage source (ici le français) et Y pour le langage cible (ici l'anglais)
vector_key, vector_value, word_key, word_value = get_translation_array(dict_fr_en, dict_fr, dict_en, verbose = True)
X = normalisation(vector_key)
Y = normalisation(vector_value)

##On obtient la matrice W par la méthode contrainte
nb_ex = 50000
X_train, Y_train = X[:nb_ex,:], Y[:nb_ex,:]
W_constrained, _, _ = get_W(X_train, Y_train, alpha = 10)

##On obtient ensuite la traduction des N premiers mots
N = 10
res_const, acc_const = recherche_minimum(W_constrained, X[:N,:], Y, word_key[:N], word_value, dist = 1)


print("Mot en français à traduire: | Traduction en anglais avec la matrice orthogonale: ")
for i in range(len(res_const)):
    print("{:^27} | {:^50}".format(res_const[i][0], res_const[i][1]))
print("Accuracy de {} avec la matrice orthogonale sur les {} exemples".format(acc_const, N))


##On peut faire de même avec la matrice non contrainte TO UNCOMMENT
# W_unconstrained, _, _ = GD_W_not_const(X_train, Y_train)
# print(W_unconstrained.shape)
# res_unconst, acc_unconst = recherche_minimum(W_unconstrained, X[:N,:], Y, word_key[:N], word_value, dist = 1)

# print("Mot en français à traduire: | Traduction en anglais avec la matrice orthogonale: ")
# for i in range(len(res_unconst)):
#     print("{:^27} | {:^50} ".format(res_unconst[i][0], res_unconst[i][1]))

# print("Accuracy de {} avec la matrice orthogonale sur les {} exemples".format(acc_unconst, N))
