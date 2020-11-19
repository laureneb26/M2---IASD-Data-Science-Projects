

# !curl -Lo wiki.en.vec https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.en.vec
# !curl -Lo wiki.fr.vec https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.fr.vec
# !curl -Lo wiki.fr-en.vec https://dl.fbaipublicfiles.com/arrival/dictionaries/fr-en.txt
# !curl -Lo wiki.en-fr.vec https://dl.fbaipublicfiles.com/arrival/dictionaries/en-fr.txt

# !curl -Lo wiki.he.vec https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.he.vec
# !curl -Lo wiki.he-en.vec https://dl.fbaipublicfiles.com/arrival/dictionaries/he-en.txt
# !curl -Lo wiki.en-he.vec https://dl.fbaipublicfiles.com/arrival/dictionaries/en-he.txt

# !curl -Lo wiki.el.vec https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.el.vec
# !curl -Lo wiki.el-en.vec https://dl.fbaipublicfiles.com/arrival/dictionaries/el-en.txt
# !curl -Lo wiki.en-el.vec https://dl.fbaipublicfiles.com/arrival/dictionaries/en-el.txt

# !curl -Lo wiki.es.vec https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.es.vec
# !curl -Lo wiki.es-en.vec https://dl.fbaipublicfiles.com/arrival/dictionaries/es-en.txt
# !curl -Lo wiki.en-es.vec https://dl.fbaipublicfiles.com/arrival/dictionaries/en-es.txt

# !curl -Lo wiki.he-es.vec https://dl.fbaipublicfiles.com/arrival/dictionaries/he-es.txt
# !curl -Lo wiki.es-he.vec https://dl.fbaipublicfiles.com/arrival/dictionaries/es-he.txt

import numpy as np
import matplotlib.pyplot as plt
from word_utils import *

#Creation of the dict
taille_max = 10**5

dict_en = load_vector_file("wiki.en.vec", max = taille_max)
dict_fr = load_vector_file("wiki.fr.vec", max = taille_max)
dict_fr_en = load_dict_file(fname = "wiki.fr-en.vec", max = taille_max)

dict_es = load_vector_file("wiki.es.vec", max = taille_max)
dict_es_en = load_dict_file(fname = "wiki.es-en.vec", max = taille_max)
dict_en_es = load_dict_file(fname = "wiki.en-es.vec", max = taille_max)

dict_es_he = load_dict_file(fname = "wiki.es-he.vec", max = taille_max)
dict_he_es = load_dict_file(fname = "wiki.he-es.vec", max = taille_max)
dict_he = load_vector_file("wiki.he.vec", max = taille_max)

dict_el = load_vector_file("wiki.el.vec", max = taille_max)
dict_he_en = load_dict_file(fname = "wiki.he-en.vec", max = taille_max)
dict_el_en = load_dict_file(fname = "wiki.el-en.vec", max = taille_max)
dict_en_he = load_dict_file(fname = "wiki.en-he.vec", max = taille_max)
dict_en_el = load_dict_file(fname = "wiki.en-el.vec", max = taille_max)



from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
 
#dict_keys are french words+ embedding
#dict_vals are en words + emedding
#dict_trad is the translation dictionnary of words pairs en-fr

def graphical_rpz(dict_key, dict_value, dict_trad,nmax,norm=False):
  
  vvkey, vvalue, words_X, words_Y = get_translation_array(dict_trad,dict_key,dict_value)
  #vvkey = embedding of source language
  #vvalue = embedding of target language
  #words_X = word of source language
  #words_Y = words of trgt lang

  if norm:
    X_train = normalisation(vvkey)
    Y_train = normalisation(vvalue)
  else:
    X_train = vvkey
    Y_train = vvalue
  #calculate the translation matrix (projection of sourcec lang on trgt lang)
  W = get_W(X_train, Y_train, alpha=10, init = 2)
  source=[]
  Wsource=[]
  target=[]

  #get embedding of source language
  for i in range(nmax):
    source.append(vvkey[i,:])
    
 #W*Source: get translation of source lang
  for i in range(nmax):
    Wsource.append(W@vvkey[i,:])
    
#get embedding of trgt lang
  for i in range(nmax):
    target.append(vvalue[i,:])
#PCA applied on target language
  pca=PCA(n_components=2)#top 2 eigen val
  pca.fit(target)
  target=pca.transform(target)
  source=pca.transform(source)
  Wsource=pca.transform(Wsource)
  
  x_target = target[:, 0]#1st eigenv (PCA dim)
  y_target = target[:, 1]#2nd eigenv
  x_source = source[:, 0]
  y_source = source[:, 1]
  x_Wsource = Wsource[:, 0]
  y_Wsource = Wsource[:, 1]

  plt.figure(figsize=(20,8))
  blue = [0, 0.4470, 0.7410]
  red = [0.6350, 0.0780, 0.1840]
  grey = [0.25, 0.25, 0.25]
  plt.subplot(121)
  for i in range(nmax):
    # color =blue
    label=words_X[i]
    #print(label)
    plt.annotate(label, xy=(x_source[i], y_source[i]), xytext=(5, 5), textcoords='offset points', fontsize=15,color=blue)
    # color=red # src words in blue / tgt words in red
    label=words_Y[i]
    #print(label)
    plt.annotate(label, xy=(x_target[i], y_target[i]), xytext=(5, 5), textcoords='offset points', fontsize=15,color=red )

  plt.scatter(x_source, y_source, marker='x', color ='blue')
  plt.scatter(x_target, y_target, marker='x',color='red')
  plt.plot([x_source, x_target],[y_source, y_target],'--')
  plt.title("Visualisation Before applying W",size=18,color=grey)


  plt.subplot(122)
  for i in range(nmax):
    # color =blue
    label=words_X[i]
    plt.annotate(label, xy=(x_Wsource[i], y_Wsource[i]), xytext=(5, 5), textcoords='offset points', fontsize=15,color=blue)
    # color=red # src words in blue / tgt words in red
    label=words_Y[i]
    plt.annotate(label, xy=(x_target[i], y_target[i]), xytext=(5, 5), textcoords='offset points', fontsize=15,color=red)

    print(words_X[i],words_Y[i])

  plt.title(" Visualisation After applying W on source language",size=18,color=grey)
  plt.scatter(x_Wsource, y_Wsource, marker='x',color='blue')
  plt.scatter(x_target, y_target, marker='x',color='red')
  plt.plot([x_Wsource, x_target],[y_Wsource, y_target],'--')

  plt.show()

### PCA - king man queen woman

#SEMANTIC SINGULAR PLURAL TEST
def small_dict(dict_mono,word_arr):
  new_dict={}
  # i = 0
  for word in word_arr:
    for w in list(dict_mono.keys()):
      if (word == w):
        new_dict[w] = dict_mono.get(w)
  return new_dict
word_arr2 = ['king','man','woman']
dict_en_test =small_dict(dict_en,word_arr2)
nmax = 3


vval = [dict_en.get('king'),dict_en.get('man'),dict_en.get('woman')]
vval = np.array(vval)

words_Y = ('king','men','women','king-man+women = queen')
Y_train = vval
target=[]
#get embedding of trgt lang
for i in range(nmax):
  target.append(vval[i,:])

#testing vector operations
test_queen_emb = np.array(dict_en.get('king'))- np.array(dict_en.get('man'))+np.array(dict_en.get('woman'))
target.append(test_queen_emb)


#PCA applied on target language
pca=PCA(n_components=2)#top 2 eigen val
pca.fit(target)
target=pca.transform(target)
#plot
x_target = target[:, 0]#1st eigenv (PCA dim)
y_target = target[:, 1]#2nd eigenv
plt.figure(figsize=(7,4))
red = [0.6350, 0.0780, 0.1840]
grey = [0.25, 0.25, 0.25]
plt.scatter(x_target, y_target, marker='x',color='red')

for i in range(0,nmax):
  color=red # src words in blue / tgt words in red
  label=words_Y[i]
  plt.annotate(label, xy=(x_target[i], y_target[i]), xytext=(5,10-i*10), textcoords='offset points', fontsize=15,color=red )
plt.annotate(words_Y[nmax], xy=(x_target[nmax], y_target[nmax]), xytext=(5,10-i*10), textcoords='offset points', fontsize=15,color='black' )
plt.plot([x_target[0], x_target[3]],[y_target[0], y_target[3]],'--')
plt.plot([x_target[1], x_target[2]],[y_target[1], y_target[2]],'--')
plt.title("Semantic relation visualisation PCA: 'king' - 'man' + 'women' = 'queen'",size=16,color=grey)
plt.show()

#SYNTACTIC SINGULAR PLURAL TEST

word_arr2 = ['dog','dogs','car','cars']
dict_en_test =small_dict(dict_en,word_arr2)
nmax = 4
vvkey, vvalue, words_X, words_Y = get_translation_array(dict_fr_en,dict_fr,dict_en)
words_Y = ('dog','dogs','car','cars')
Y_train = vvalue
target=[]

#get embedding of trgt lang
for i in range(nmax):
  target.append(vvalue[i,:])
#PCA applied on target language
pca=PCA(n_components=2)#top 2 eigen val
pca.fit(target)
target=pca.transform(target)
#plot
x_target = target[:, 0]#1st eigenv (PCA dim)
y_target = target[:, 1]#2nd eigenv
plt.figure(figsize=(7,4))
red = [0.6350, 0.0780, 0.1840]
grey = [0.25, 0.25, 0.25]
plt.scatter(x_target, y_target, marker='x',color='red')
for i in range(0,nmax):
  color=red # src words in blue / tgt words in red
  label=words_Y[i]
  plt.annotate(label, xy=(x_target[i], y_target[i]), xytext=(5,10-i*10), textcoords='offset points', fontsize=15,color=red )
#plt.title("Syntactic relation visualisation PCA",size=16,color=grey)
plt.plot([x_target[0], x_target[1]],[y_target[0], y_target[1]],'--')
plt.show()

#Unconstrained Matrix
alphas = [1e-5, 1e-4, 1e-3, 5e-3, 1e-2]
res_liste = []
liste_if = []
liste_valf = []
X = normalisation(vvkey)
Y = normalisation(vvalue)
for alpha in alphas:
    _,i_f, val_f = GD_W_not_const(X, Y, alpha = alpha, nb_max = -1)
    liste_if += [i_f]
    liste_valf += [val_f]
plt.figure(figsize = (8,4))
for i in range(len(alphas)):
    plt.plot(liste_if[0], liste_valf[i], label = "alpha = {}".format(alphas[i]))
plt.grid()
plt.legend()
plt.xlabel("Number of examples")
plt.ylabel("MSE")
plt.savefig("mse_for_unconstraint.png")

#Constrained Matrix
def get_W2(X_train, Y_train, alpha = 1, init = 2):
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
        
        x_i = X_train[i].reshape((-1,1))
        y_i = Y_train[i].reshape((-1,1))
        delta_W = y_i@(x_i.T)
        W += alpha*delta_W
        if i %1000==0:
            i_f += [i]
            val_f += [sum(f(W, X_train[i,:].reshape((m,1)), Y_train[i,:].reshape((m,1))) for i in range(len(X_train)))/len(X_train)]
    u, _ , v = np.linalg.svd(W)
    W = u@np.eye(m)@v
    return W, i_f, val_f
    
    
vvkey, vvalue, str_key, str_value = get_translation_array(dict_fr_en,
                                                          dict_fr,
                                                          dict_en,
                                                          verbose = True)

X = normalisation(vvkey)
Y = normalisation(vvalue)

alphas = [1e-5, 1e-4, 1e-3, 5e-3, 1e-2]
res_liste = []
liste_if = []
liste_valf = []
for alpha in alphas:
    print(alpha)       
    _,i_f, val_f = get_W2(X, Y, alpha = alpha, init = 2)
    liste_if += [i_f]
    liste_valf += [val_f]
#   res_liste += [res]
plt.figure(figsize = (8,4))
for i in range(len(alphas)):
    plt.plot(liste_if[0], liste_valf[i], label = "alpha = {}".format(alphas[i]))
plt.grid()
plt.legend()
plt.xlabel("Number of examples")
plt.ylabel("MSE")
plt.savefig("mse_for_constraint.png")


#Supervised French-English
#accuracy on 3000 entry and 90%validation split
resultats, acc=traducteur(dict_fr, dict_en, dict_fr_en, norm = False, dist = 1,alpha = 10, N = 3000, init = 2, verbose =False,valid=True,validation_split=0.9)
print(acc)

#PCA
graphical_rpz(dict_fr, dict_en, dict_fr_en,7,norm=True)

#Initialization
alpha=[0.001,0.01,0.1,1,10,100]
Results_train_normalisation_fr_Winit0=np.zeros((6,2))
Results_train_normalisation_fr_Winit1=np.zeros((6,2))
Results_train_normalisation_fr_Winit2=np.zeros((6,2))

i=0
for a in alpha :
  resultats, acc=traducteur(dict_fr, dict_en, dict_fr_en, norm = True, dist = 0,alpha = a, N = 1000, init = 0, verbose = False,valid=True,validation_split=0.9)
  Results_train_normalisation_fr_Winit0[i,:]=acc
  resultats, acc=traducteur(dict_fr, dict_en, dict_fr_en, norm = True, dist = 0,alpha = a, N = 1000, init = 1, verbose = False,valid=True,validation_split=0.9)
  Results_train_normalisation_fr_Winit1[i,:]=acc
  resultats, acc=traducteur(dict_fr, dict_en, dict_fr_en, norm = True, dist = 0,alpha = a, N = 1000, init = 2, verbose = False,valid=True,validation_split=0.9)
  Results_train_normalisation_fr_Winit2[i,:]=acc
  i+=1

plt.figure(figsize=(10,4))
plt.subplot(121)
plt.semilogx(alpha,Results_train_normalisation_fr_Winit0[:,0],'o--',label='matrice with zeros')
plt.semilogx(alpha,Results_train_normalisation_fr_Winit1[:,0],'o--',label='uniform distribution')
plt.semilogx(alpha,Results_train_normalisation_fr_Winit2[:,0],'o--',label='gaussian distribution')
plt.legend()
plt.xlabel('alpha')
plt.ylabel('Accuracy')
plt.title('Difference in the initialization of W on training set')
plt.subplot(122)
plt.semilogx(alpha,Results_train_normalisation_fr_Winit0[:,1],'o--',label='matrice with zeros')
plt.semilogx(alpha,Results_train_normalisation_fr_Winit1[:,1],'o--',label='uniform distribution')
plt.semilogx(alpha,Results_train_normalisation_fr_Winit2[:,1],'o--',label='gaussian distribution')
plt.legend()
plt.xlabel('alpha')
plt.ylabel('Accuracy')
plt.title('Difference in the initialization of W on testing set')
plt.savefig("initialization.png")


## Normalization
#With normalization
resultats, acc=traducteur(dict_fr, dict_en, dict_fr_en, norm = True, dist = 1,alpha = 10, N = 1000, init = 2, verbose = False,valid=True,validation_split=0.9)
print(acc)
#Without Normalization
resultats, acc=traducteur(dict_fr, dict_en, dict_fr_en, norm = False, dist = 1,alpha = 10, N = 1000, init = 2, verbose =False,valid=True,validation_split=0.9)
print(acc)
##Distance
#LinalgNorm
resultats, acc=traducteur(dict_fr, dict_en, dict_fr_en, norm = True, dist = 0,alpha = 10, N = 1000, init = 2, verbose = False,valid=True,validation_split=0.9)
print(acc)
#Cosine Distance
resultats, acc=traducteur(dict_fr, dict_en, dict_fr_en, norm = True, dist = 1,alpha = 10, N = 1000, init = 2, verbose =False,valid=True,validation_split=0.9)
print(acc)

#PCA espagnol-anglais
graphical_rpz(dict_en,dict_es,dict_en_es,5)

#PCA hebreu-anglais
graphical_rpz(dict_en,dict_el,dict_en_el,5)