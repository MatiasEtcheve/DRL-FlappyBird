# Flappy Bird Helper

## Contenu
### Agent de RL
Dossier `helper/agents`  
Agents implémentés:
- DQN + Prioritized experience replay
- PPO
- Reinforce
- A2C

### Tuning d'hyperparamètres
Dossier `helper/hyparam_tuning`

### Boucles d'entraînement
Dossier `helper/training`


## Instructions d'installation
La bibliothèque `helper` est conçus pour être installée dans un notebook Colab de la même manière
que la bibilothèque `deep-rl`. Comme le dépôt est privé, il y a une procédure un peu technique
à accomplir **une seule fois**:

### 1. Créer une clé ssh
Sur votre ordinateur local, créez une clé ssh
```bash
ssh-keygen
```
Ne mettez pas de phrase secrète. Comme nom de fichier, écrivez `./flappy_rsa`

### 2. Créer un ficher de config
Créez un fichier appelé `config` dans lequel vous écrivez ceci:
```shell
Host github.com
  HostName github.com
  User git
  ItentityFile /root/.ssh/flappy_rsa
```
### 3. Emballer le tous dans un joli dossier
Créez un dossier nommé `ssh-colab` dans lequel vous placerez les fichiers suivants:
- flappy_rsa
- config

### 4. Envoyer le dossier sur Google drive
Téléversez le dossier `ssh-colab` sur votre Google Drive dans le dossier associé au projet.

### 5. Associer votre clé publique à votre contre GitHub
Dans les réglages ssh de votre compte GitHub, ajoutez une nouvelle clé ssh et collez-y le contenu
du fichier `flappy_rsa.pub`

### 6. Ajoutez les cellules suivantes au début de votre noteboook
Une cellule pour accéder à votre Drive où se trouve votre clé privée
```Python
from google.colab import drive

drive.mount('/content/drive')
GDRIVE_PATH = "chemin/vers/le/dossier/du/projet"  # Adapter selon votre structure de Drive
```
/!\ Attention, le chemin indiqué doit être le dossier **dans lequel** se trouve `ssh-colab`

Ensuite laissez la cellule qui installe `deep-rl`, la bibliothèque des profs.

Puis ajoutez une dernière cellule qui fera l'installation de la présente bibliothèque
```iPython
# Select the repository's branch to clone from
BRANCH = 'main'

!cp /content/drive/MyDrive/{GDRIVE_PATH}/ssh-colab/* /root/.ssh
!rm ~/.ssh/known_hosts
!ssh-keyscan -t rsa github.com >> ~/.ssh/known_hosts
!chmod go-rwx /root/.ssh/flappy_bird_rsa
!pip install "git+ssh://git@github.com/MatiasEtcheve/DRL-FlappyBird.git@$BRANCH#egg=helper"
clear_output()
```
Voilà !