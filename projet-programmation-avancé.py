# projet programmation avancé
# vacances et bagage
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap
import copy

classe_importance = {"a" :2, "A" : 4 ,"b":6}
classe_sizes = {"a" : 20 , "b":30,"A":60}
espace = 50

def take_objects(obj_importance,obj_sizes,car_space):
    resultat = list()
    opt = 0
    t  = len(obj_importance)
    for i in range(t):
        objet = max(obj_importance, key = obj_importance.get)
        if car_space >= obj_sizes[objet]:
            opt+= obj_sizes[objet]
            car_space -= obj_sizes[objet]
            resultat.append(objet)
        else :
            del obj_importance[objet]
            del obj_sizes[objet] 
            continue

        del obj_importance[objet]
        del obj_sizes[objet]
        if car_space == 0 :
            break
        
    return (opt,resultat)


def take_objects2(obj_importance, obj_sizes, car_space):
  """
  Détermine les objets à inclure dans la voiture.

  Args:
    obj_importance: La liste des importances des objets.
    obj_sizes: La liste des tailles des objets.
    car_space: L'espace disponible dans la voiture.

  Returns:
    (opt, choices) tel que opt est la valeur optimale du problème et choices est
    la listes des objets pris.
  """

  n_objects = len(obj_importance)
  # Initialisation de la solution.

  opt = 0
  choices = []
  # Parcours des objets dans l'ordre décroissant de l'importance.

  for i in range(n_objects):
    # Si l'objet rentre dans la voiture, on l'ajoute à la solution.
    if obj_sizes[i] <= car_space:
      opt += obj_importance[i]
      choices.append(i)
      car_space -= obj_sizes[i]
  return (opt, choices)


obj_importance = [25, 20, 30, 40]
obj_sizes = [1, 2, 3, 6]
car_space = 11

(opt, choices) = take_objects2(obj_importance, obj_sizes, car_space)

print(opt)
print(choices)




def is_valid_move(puzzle, row, col, number):
    for i in range(9):
        if puzzle[i][col] == number or puzzle[row][i] == number:
            return False
    c_row = row - row % 3
    c_col = col - col % 3
    for i in range(3):
        for j in range(3):
            if puzzle[c_row + i][c_col + j] == number:
                return False
    return True

def solve(puzzle, row, col):
    if col == 9:
        if row == 8:
            return True
        row += 1
        col = 0
    if puzzle[row][col] > 0:
        return solve(puzzle, row, col + 1)
    for i in range(1, 10):
        if is_valid_move(puzzle, row, col, i):
            puzzle[row][col] = i
            if solve(puzzle, row, col + 1):
                return True
            puzzle[row][col] = 0
    return False

def sudoku(puzzle):
    if solve(puzzle, 0, 0):
        for i in range(9):
            for j in range(9):
                print(puzzle[i][j], end=" ")
            print()
    else:
        print("Pas de solution.")

# Exemple d'utilisation :
puzzle = [
    [5, 3, 0, 0, 7, 0, 0, 0, 0],
    [6, 0, 0, 1, 9, 5, 0, 0, 0],
    [0, 9, 8, 0, 0, 0, 0, 6, 0],
    [8, 0, 0, 0, 6, 0, 0, 0, 3],
    [4, 0, 0, 8, 0, 3, 0, 0, 1],
    [7, 0, 0, 0, 2, 0, 0, 0, 6],
    [0, 6, 0, 0, 0, 0, 2, 8, 0],
    [0, 0, 0, 4, 1, 9, 0, 0, 5],
    [0, 0, 0, 0, 8, 0, 0, 7, 9]
]

initial_puzzle = copy.deepcopy(puzzle)

sudoku(puzzle)


# Définition des couleurs
colors = ['white', 'skyblue']

# Convertir le puzzle en une matrice de couleurs
puzzle = np.array(puzzle)
colored_puzzle = np.zeros(puzzle.shape)
for i in range(puzzle.shape[0]):
    for j in range(puzzle.shape[1]):
        if puzzle[i][j] == 0:
            colored_puzzle[i][j] = 0
        elif puzzle[i][j] == initial_puzzle[i][j]:
            colored_puzzle[i][j] = 1
        else:
            colored_puzzle[i][j] = 2

x = np.arange(-0.5, 9, 1)
y = np.arange(8.5, -1, -1)

fig, ax = plt.subplots()
mesh = ax.pcolormesh(x, y, colored_puzzle, edgecolors='black', linewidth=2, cmap=ListedColormap(colors))

for i in range(9):
    for j in range(9):
        if puzzle[i][j] != 0:
            ax.text(j, 8 - i, puzzle[i][j], ha='center', va='center', fontsize=15, color='black')

plt.xlim(-0.5, 8.5)
plt.ylim(-0.5, 8.5)
plt.axis('off') 
plt.grid(color='black', linewidth=2)
plt.show()


### Arbre de Décision 
class DecisionTree:
	def __init__(self,criterion):
		self.criterion = criterion
	
	def fit(X,y):
		self.tree = self.build_tree(X, y)
	
	def predict(X):
		pass
		
	def build_tree(self, X, y):
		pass




### Foret Aléatoire
class RandomForest:
	def __init__(self,n_estimators,criterion):
		self.n_estimators = n_estimators
		self.criterion = criterion
	
	def fit(X,y):
		pass
	
	def predict(X):
		pass


#Numpy : Regression Logistique

class LogisticRegresion :
    def __init__(self,nb_iter):
	"""
    Implemente un classificateur d'arbre de décision utilisant le critère d'impureté de Gini ou d'entropie.

    Parameters:
        criterion (str): Critère d'impureté à utiliser pour la division. La valeur par défaut est "gini".
            Les critères supportés sont 'gini' et 'entropy'.

    Attributes:
        tree (dict): Le noeud racine de l'arbre de décision.
        criterion (str): Critère d'impureté utilisé pour la division.

    Methods:
        _calculate_impurity(self, counts): Calcule l'impureté d'un ensemble donné de nombres de classes.
        _split_dataset(self, X, y, feature_index, threshold): Divise l'ensemble de données en fonction d'une caractéristique et d'un seuil.
        _find_best_split(self, X, y): Détermine la meilleure caractéristique et le meilleur seuil pour diviser l'ensemble de données.
        _build_tree(self, X, y, depth=0, max_depth=5): Construit de manière récursive l'arbre de décision.
        fit(self, X, y): Ajuste l'arbre de décision aux données et aux étiquettes données.
        _predict_single(self, tree, sample): Predit l'étiquette de la classe pour un seul échantillon.
        predict(self, X): Prédit les étiquettes de classe pour un ensemble donné d'échantillons.
        plot_decision_boundary(model, X, y, title): Trace la frontière de décision pour un modèle donné.
    """
        self.nb_iter = nb_iter
        

    def fit(self,X,y):
        n,d = X.shape
        self.beta = np.linalg.pinv(x)@y

    def predict(self,X):
        pass

#   Vacances et bagage
# obj_importance et  obj_sizes sont considérés comme des dictionnaires dans ce probléme :
# car_place est une valeur représentant l'espace de la voiture



class RandomForest:
    """
    Implemente un classificateur de forêt aléatoire utilisant le critère d'impureté de Gini ou d'entropie.
    
    Parameters:
		criterion (str): Critère d'impureté à utiliser pour la division. La valeur par défaut est "gini".
            Les critères supportés sont 'gini' et 'entropy'.
            
    Attributes:  
		forest (list): La liste des arbres de la forêt.
		
    Methods:    
		fit(self, X, y): Prédire les étiquettes de classe pour un ensemble donné d'échantillons.Cette méthode s'entraîne sur les données d'entraînement données 
			et construit une forêt aléatoire de `n_estimators` arbres. Chaque arbre est construit en utilisant un sous-échantillon aléatoire des données d'entraînement.
			Le critère d'impureté spécifié est utilisé pour choisir la meilleure caractéristique et le meilleur seuil pour fractionner chaque nœud de l'arbre.
		predict(self, X):Prédire les étiquettes de classe pour un ensemble donné d'échantillons.Cette méthode prédit les étiquettes de classe pour les échantillons donnés 
			en votant pour les étiquettes de classe prédites par chaque arbre de la forêt. 
			L'étiquette de classe prédite pour un échantillon est l'étiquette la plus courante parmi les étiquettes prédites par les arbres.
		_calculate_impurity(self, counts): Calcule l'impureté d'un ensemble donné de nombres de classes.
		_split_dataset(self, X, y, feature_index, threshold): Divise l'ensemble de données en fonction d'une caractéristique et d'un seuil.
		_find_best_split(self, X, y): Détermine la meilleure caractéristique et le meilleur seuil pour diviser l'ensemble de données.
		_build_tree(self, X, y, depth=0, max_depth=5): Construit de manière récursive l'arbre de décision.
		_predict_single(self, tree, sample): Predit l'étiquette de la classe pour un seul échantillon.
		plot_decision_boundary(model, X, y, title): Trace la frontière de décision pour un modèle donné.
    """

def take_objects(obj_importance, obj_sizes, car_space):
  n=len(obj_importance)
  opt=0
  choices=[]
  for i in range(n):
    im_max=max(obj_importance)
    val_max=obj_sizes[im_max]
    if val_max <=car_space:
      choices.append(im_max)
      opt+=obj_sizes[im_max]
      del obj_importance[im_max]
      del obj_sizes[im_max]
      car_space=car_space-val_max
    else:
      del obj_importance[im_max]
      del obj_sizes[im_max]
      continue
  return (choices,opt)
"""

"""
from sklearn.datasets import make_moons, make_circles, make_blobs
datasets = [
    make_moons(noise=0.3, random_state=0),
    make_circles(noise=0.2, factor=0.5, random_state=1),
    make_blobs(n_samples=100, centers=2, n_features=2, center_box=(0, 20), random_state=0)
]
"""
