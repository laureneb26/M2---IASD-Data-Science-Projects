# Projet 3 - Training robust neural networks

### Avancement
* code pour charger notre modèle, le dataset CIFAR-10 et lancer l'entraînement : DONE
* code pour améliorer l'accuracy du modèle (data augmentation, learning_rate decay, etc...) : DONE
* code pour mesurer l'accuracy après une attaque : DONE 
* code pour attaquer efficacement avec FGSM et PGD : DONE
* code pour défendre avec Adversarial Training : DONE (contre efficacement FGSM mais pas PGD)
* code pour défendre avec une défense à nous (dans l'espoir de contrer PGD) : EN COURS
* code pour tester d'autres attaques / d'autres normes : À VENIR

### Notre projet en quelques chiffres clefs
* Nos modèles ont une accuracy de base autour de 80-90% sur CIFAR-10.
* Sans défense on descend à moins de 8% avec FGSM et autour de 0% avec PGD. (norme l-infinie, epsilon = 0.05)
* Avec défense dans les mêmes conditions on est plutôt à 35% et 5% 