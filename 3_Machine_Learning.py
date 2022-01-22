

# importing the main modules:
import numpy as np
import pandas as pd
import sklearn.model_selection
import sklearn.metrics
import sklearn.ensemble



print("Enter the name of the pdbqt file containing the receptor (without .pqbqt extension):")
receptor_name = str(input())

read_file = pd.read_excel ('.\\Results_'+receptor_name+'\\fragLipinski.xlsx', sheet_name='Results')
df = pd.DataFrame(read_file).dropna()

X = df.drop(["SMILES", "Ligand Name", "Lowest binding energy (kcal/mol)"], axis=1)
Y = df["Lowest binding energy (kcal/mol)"]
X = X.astype(float)
Y = Y.astype(float)
        
X_train, X_validation, y_train, y_validation = sklearn.model_selection.train_test_split(X, Y, test_size = 0.2, random_state = 0)

m = sklearn.ensemble.RandomForestRegressor(n_estimators=500,max_features = 0.6, n_jobs=-1) #Nete that the Parameters can be changed for a better adjusting, see:https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
m.fit(X_train, y_train)
y_validation_pred = m.predict(X_validation)
print('Root Mean Square Error (RMSE) and R-squared values:')

res = [[np.sqrt(sklearn.metrics.mean_squared_error(m.predict(X_train), y_train)), m.score(X_train, y_train)],
        [np.sqrt(sklearn.metrics.mean_squared_error(m.predict(X_validation), y_validation)), m.score(X_validation, y_validation)]]
    
score = pd.DataFrame(res, columns=['RMSE','R2'], index = ['Train','Validation'])

print(score)