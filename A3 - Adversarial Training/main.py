import torch

from data_loader import *
from models import *
from monitoring import *
from trainings import *
from attacks import *

# choix du device et du type
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device ( input("device (cuda,cpu,cuda:1) :") )
settings = {'device': device, 'dtype': torch.float}
print(f"device : {settings['device']}\ndtype : {settings['dtype']}")


# on charge les deux datasets sur le device
cifar10_train = CIFAR10(True, settings)
cifar10_test = CIFAR10(False, settings)


# on charge le modèle sur le device
model = ConvNet(dropout=0.3,sigma=0.05,settings=settings).to(**settings)
# model_simple : ancien modèle avec 78% d'accuracy
# model_state : nouveau modèle de Alejandro avec 87% d'accuracy
# model_new : copie de model_state mais moins performante avec 83% d'accuracy
# model_advtraining : modèle entraîné avec l'adversarial training (efficace contre fgsm)
# model_cont : modèle entraîné avec la contrastive loss, mauvaise défense (84% d'accuracy)
# model_advtriplet : modèle entraîné avec une triplet loss custom 
model.load_state_dict ( torch.load ( 'models/model_cont.pt', map_location = settings['device'] ) )

batch_size = 500

for iter in range ( int(input("iterations :")) ) :
    train_cont ( model, cifar10_train, batch_size = batch_size, epochs=20, learning_rate=7e-3)
    print ( "Acc :", fmt(get_accuracy ( model, cifar10_test, max_n = 1000 )) )
    torch.save ( model.state_dict(), 'models/model_defcont1.pt' )

# fonction qui affiche les scores de notre modèle après différentes attaques
def fmt2 ( res, eps, w = 'SANS' ) :
    print ( '***** {} DÉFENSE *****'.format ( w ) )
    print ( "epsilon = {}".format ( eps ) )
    print ( "accuracy totale : {}".format ( fmt(res[0]/res[2]) ) )
    print ( "accuracy après aucune attaque : {}".format ( fmt(res[1][0]/res[0] ) ) )
    print ( "accuracy après fgsm : {}".format ( fmt(res[1][1]/res[0]) ) )
    print ( "accuracy après pgd : {}".format ( fmt(res[1][2]/res[0]) ) )

for eps in [0.01, 0.05] :
    fmt2 ( get_accuracy_after_attacks ( model, cifar10_test, [ no_attack, fgsm_attack, pgd_attack ], eps = eps, max_n = 500 ), eps, 'AVEC CONTRASTIVE' )
    fmt2 ( getcont_accuracy_after_attacks ( model, cifar10_test, [ no_attack, fgsm_attack, pgd_attack ], eps = eps, max_n = 500 ), eps, 'AVEC CONTRASTIVE + EL' )

model.load_state_dict ( torch.load ( 'models/model_advtraining.pt', map_location = settings['device'] ) )
for eps in [0.01, 0.05] :
    fmt2 ( get_accuracy_after_attacks ( model, cifar10_test, [ no_attack, fgsm_attack, pgd_attack ], eps = eps, max_n = 500 ), eps, 'AVEC ADV TRAINING (FGSM)' )
    fmt2 ( getcont_accuracy_after_attacks ( model, cifar10_test, [ no_attack, fgsm_attack, pgd_attack ], eps = eps, max_n = 500 ), eps, 'AVEC ADV TRAINING (FGSM) + EL' )

model.load_state_dict ( torch.load ( 'models/model_advpgd.pt', map_location = settings['device'] ) )
for eps in [0.01, 0.05] :
    fmt2 ( get_accuracy_after_attacks ( model, cifar10_test, [ no_attack, fgsm_attack, pgd_attack ], eps = eps, max_n = 500 ), eps, 'AVEC ADV TRAINING (PGD)' )
    fmt2 ( getcont_accuracy_after_attacks ( model, cifar10_test, [ no_attack, fgsm_attack, pgd_attack ], eps = eps, max_n = 500 ), eps, 'AVEC ADV TRAINING (PGD) + EL' )
















