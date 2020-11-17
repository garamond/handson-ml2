#%%
import numpy as np
#%%
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', version=1)
# %%
X, y = mnist["data"], mnist["target"]
# %%
y = y.astype(np.uint8)
# %%
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
# %%
from sklearn.neighbors import KNeighborsClassifier
kn_clf = KNeighborsClassifier()
#kn_clf.fit(X_train, y_train)
#%%
from datetime import datetime
start = datetime.now()
from sklearn.model_selection import GridSearchCV
param_grid = [
    {'n_neighbors': [3, 5, 8], 'weights': ['uniform', 'distance']}
]
grid_search = GridSearchCV(kn_clf, param_grid, cv=3,
                           scoring='accuracy',
                           return_train_score=True, verbose=3, n_jobs=-1, pre_dispatch=2)
print(f'Start grid search {grid_search}')
grid_search.fit(X_train, y_train)
# %%
print(f'Completed in {datetime.now()-start} with best params: {grid_search.best_params_}')
# %%
import joblib
joblib.dump(grid_search, 'grid_search.pkl')
