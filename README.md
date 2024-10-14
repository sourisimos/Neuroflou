# Neuro-Flou

Développements en Neuro-Flou.


=========================================
Obtenir le dossier depuis git ONERA (nécessite d'être associé dans le projet)
1) Ouvrir le terminal
2) cmd : git clone http://gitlab-dtis.onera/syd/neuro-flou/neuro-flou.git
3) git pull origin


=========================================
Préparer l'environnement et la configuration
Une fois le logiciel possédé:
Commande à effectuer dans le terminal dans l'odre
1) cd neuro-flou/code
2) python3 -m venv mon-nom-d-env # Creation de l'env virtuel
3) source mon-nom-d-env/bin/activate
4) pip install --proxy=proxy:80 --upgrade pip # maj de pip (souvent dépassé à l'ONERA....) 
5) pip install --proxy=proxy:80 -r requirements.txt # isntallation des librairies
6) La configuration est prête !


=========================================
Effectuer un entrainement du réseau:
1) Modifier dans le fichier INIT.py (à ouvrir depuis le gestionnaire de fichier) les paramètres que l'on souhaite
Commande à effectuer dans le terminal dans l'ordre
1) cd neuroflou/code
2) python3 main.py (-args) 
Des arguments peuvent être spécifiés pour modifier directement depuis le terminal

Une execution se lance, un message "Started" apparait après quelques secondes, puis une chargement s'affiche. A 100% l'entrainement est terminé. 

Une fois l'entrainement terminé des sorties sont enregistrées dans code/output_temp: 
	dossier console_log correspond aux informations numériques du réseau : paramètres, valeurs des poids après apprentissage, valeurs de la fonction cout sur les données d'apprentissage, de validation, ... 
	dossier graph : correspond à l'évolution des fonctions couts des différents réseaux ouverts et considéré comme prometeur par rapport au nombre d'itération totale (dont les itérations utilisées pour les réseaux voisins non prometteurs)
	dossier membership_func : correspond aux fonctions d'appartenances des differents descripteurs linguistique 

==========================================
Ajouter un nouveau jeu de données brute: 
1) Créer un dossier "folder_name" dans neuro-flou/datasets
2) sauvegarder le fichier de données brut dans le dossier créé
3) ouvrir un terminal 
Commande à effectuer dans le terminal dans l'ordre
4) cd neuro-flou/code
5) source -m mon-nom-d-env/bin/activate
6) python3 data_creator.py -h
7) python3 data_creator.py -args /!\ spécifier les arguments obligatoires!
8) attention il faut rajouter une entrée au dictionnaire netw_variables dans INIT.py contenant les variables du réseau manuellement et y associer la clé sous le bon format (<FOLDER_NAME>_<SAMPLE_NAME>)

Ajout d'une nouvelle version d'un jeu de donnée brute déjà présent
1) python3 data_creator.py -args /!\ Toutes les colonnes de scores ne sont pas obligatoires !!! 
2) attention il faut rajouter une entrée au dictionnaire netw_variables dans INIT.py contenant les variables du réseau manuellement et y associer la clé sous le bon format (<FOLDER_NAME>_<SAMPLE_NAME>)


Le pdf "Schéma d'arborescence" contient les arborescences décrivant la structure du code et les différentes relations entre les fichiers a partir des 3 fichiers principaux : main.py, NeuroFuzzyNetwork.py, TraingTree.py 



