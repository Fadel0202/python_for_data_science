import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

class SVM:
    def __init__(self, kernel='linear', C=None):
        self.kernel = kernel
        self.C = C
        self.alpha = None
        self.b = None
        self.X_train = None
        self.y_train = None

    def _linear_kernel(self, x1, x2):
        return np.dot(x1, x2)

    def _rbf_kernel(self, x1, x2, gamma=0.1):
        return np.exp(-gamma * np.sum((x1 - x2)**2))

    def fit(self, X, y):
		
		# Avant la définition des variables et des contraintes
        print("Dimensions de X avant optimisation :", X.shape)
        print("Dimensions de y avant optimisation :", y.shape)

		
		
        self.X_train = X
        self.y_train = y
        m,n  = X.shape

        # Variables
        alpha = cp.Variable(n)
        xi = cp.Variable(m)
        b = cp.Variable()
        # Après la définition des variables
        print("Dimensions de alpha après définition :", alpha.shape)

        # Constraints
        constraints = [0 <= alpha, alpha <= self.C, xi >= 0, cp.sum(cp.multiply(y, X @ alpha + b)) >= 1 - xi, cp.sum(alpha) - 0.5 * cp.quad_form(cp.multiply(alpha, y), X, assume_PSD=True) >= 0]



        # Après la définition des contraintes
        print("Dimensions de X @ alpha après contraintes :", (X @ alpha).shape)

        # Objective function
        obj = cp.Minimize(0.5 * cp.norm(alpha)**2 + self.C * cp.sum(xi))

        # Problem definition
        prob = cp.Problem(obj, constraints)

        # Solve the problem
        prob.solve()

        # Store the results
        self.alpha = alpha.value
        self.b = b.value

    def predict(self, X):
        decision_function = X @ self.X_train.T @ self.alpha + self.b
        return np.sign(decision_function.flatten())

    def plot_decision_boundary(self, X, y):
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, edgecolors='k', marker='o')

        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                             np.arange(y_min, y_max, 0.1))

        Z = self.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        plt.contourf(xx, yy, Z, alpha=0.3, levels=[-1, 0, 1], cmap=plt.cm.Paired)
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title(f'Decision Boundary for {self.kernel.capitalize()} SVM')
        plt.show()

# Vos données d'entraînement
X_train = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
y_train = np.array([-1, -1, 1, 1])


# Exemple d'utilisation :
# Création d'une instance de la classe SVM avec un kernel RBF et C=1
svm_rbf = SVM(kernel='linear', C=0.025)
# Entraînement sur un jeu de données X_train, y_train
svm_rbf.fit(X_train, y_train)
# Affichage de la frontière de décision
svm_rbf.plot_decision_boundary(X_train, y_train)
