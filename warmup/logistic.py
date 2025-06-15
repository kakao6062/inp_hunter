from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import numpy as np
import optuna
from sklearn.model_selection import cross_validate

class Objective:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __call__(self, trial):
        params = {
            'solver': trial.suggest_categorical('solver', ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']),
            'max_iter': trial.suggest_int('max_iter', 100, 100000),
            'C': trial.suggest_float('C', -.0001, 10),
        }
        model = LogisticRegression(**params)

        scores = cross_validate(model, X=self.X, y=self.y, scoring='accuracy', n_jobs=-1)
        return scores['test_score'].mean()

training_data = np.genfromtxt('dataset.csv', delimiter=',', dtype=np.int32)

# Holdout split
X = training_data[:, :-1]
y = training_data[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Optuna optimization
objective = Objective(X_train, y_train)
study = optuna.create_study(direction='maximize')
study.optimize(objective, timeout=60)

model = LogisticRegression(
    solver=study.best_params['solver'],
    max_iter=study.best_params['max_iter'],
    C=study.best_params['C']
)

# Train and evaluate the model
model.fit(X_train, y_train)
pred = model.predict(X_test)
print('Accuracy:', accuracy_score(y_test, pred))
print(confusion_matrix(y_test, pred))