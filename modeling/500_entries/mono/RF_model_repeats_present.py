import numpy as np
import pandas as pd
from sklearn import preprocessing as pp
from sklearn import feature_selection as fs
import scipy.stats as stats
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedKFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from scipy.stats import linregress
from pprint import pprint
import os

full_df = pd.read_csv('500_entry_altered_temp_scaled_ddG_mono_basic_compound_features.csv', index_col=0)
# full_df = pd.read_csv('p6_updated_501_altered_temp_scaled_ddG.csv', index_col=0)

#Gets B1 through temp
# X_df = full_df.iloc[:,-5:-1]
# full_df = full_df[['b1','b5','espmin','espmax','b1_min_b5','b1_dot_b5','b1_add_b5','ddG er (kcal/mol)']]
X_df = full_df.iloc[:,:-1]
# print(X_df)
X_val = X_df.values
# print(X_val)
Y_df = full_df['ddG er (kcal/mol)']
# print(Y_df)
Y_val = Y_df.values

print(X_df)
# raise ValueError()

X_train, X_test, Y_train, Y_test = train_test_split(X_val,Y_val,test_size=0.20,random_state=42)


scaler = pp.MinMaxScaler().fit(X_train)
X_train_scale = scaler.transform(X_train)
X_test_scale = scaler.transform(X_test)


thresh = fs.VarianceThreshold(threshold=0).fit(X_train_scale)
print(f'There were {np.count_nonzero(thresh.get_support())} features remaining')
X_train_var = thresh.transform(X_train_scale)
X_test_var = thresh.transform(X_test_scale)

# trees = [100]
# split_crit = ['absolute_error']
# max_depth = [None,1,5,10]
# min_samples_split = [None,]

# # Number of trees in random forest
# n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# # Number of features to consider at every split
# max_features = ['auto', 'sqrt']
# # Maximum number of levels in tree
# max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
# max_depth.append(None)
# # Minimum number of samples required to split a node
# min_samples_split = [2, 5, 10]
# # Minimum number of samples required at each leaf node
# min_samples_leaf = [1, 2, 4]
# # Method of selecting samples for training each tree
# bootstrap = [True, False]

# # Create the random grid
# random_grid = {'n_estimators': n_estimators,
#                'max_features': max_features,
#                'max_depth': max_depth,
#                'min_samples_split': min_samples_split,
#                'min_samples_leaf': min_samples_leaf,
#                'bootstrap': bootstrap}
# pprint(random_grid)

# rf = RandomForestRegressor()

# rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 5, verbose=2, random_state=42, n_jobs = -1)

# rf_random.fit(X_train_var,Y_train)

def evaluate(model, X_train, X_test, Y_train, Y_test,plot=False, name='Default Model', path_for_plot='./default_model'):

    model.fit(X_train,Y_train)
    pred_X_train = model.predict(X_train)
    fix_pred_X_train = np.reshape(pred_X_train,newshape=(pred_X_train.shape[0],))
    pred_X_test = model.predict(X_test)
    fix_pred_X_test = np.reshape(pred_X_test,newshape=(pred_X_test.shape[0],))

    train_slope,train_intercept,train_rvalue,train_pvalue,train_stderr = linregress(Y_train,fix_pred_X_train)
    test_slope,test_intercept,test_rvalue,test_pvalue,test_stderr = linregress(Y_test,fix_pred_X_test)

    train_r2 = train_rvalue**2
    test_r2 = test_rvalue**2

    train_mae = mean_absolute_error(Y_train, fix_pred_X_train)
    test_mae = mean_absolute_error(Y_test, fix_pred_X_test)

    if plot:
        fig,ax = plt.subplots()
        ax.scatter(x=Y_train,y=fix_pred_X_train, edgecolors='tab:orange', c='tab:orange',alpha=0.5)
        ax.axline((0,train_intercept), slope=train_slope, color='tab:orange', ls='dotted', label='_nolegend_')
        ax.scatter(x=Y_test,y=fix_pred_X_test, edgecolors='b',c='tab:blue',alpha=0.5)
        ax.axline((0, test_intercept), slope=test_slope, color='tab:blue', ls='dotted', label='_nolegend_')
        ax.legend(labels=['train','test'],loc='upper right')
        at = AnchoredText(
            f'Train:\ny={round(train_slope,3)}x+{round(train_intercept,3)}\nR2={round(train_r2,3)}\nMAE={round(train_mae,3)}\n\nTest:\ny={round(test_slope,3)}x+{round(test_intercept,3)}\nR2={round(test_r2,3)}\nMAE={round(test_mae,3)}',
            prop=dict(size=8),
            frameon=True,
            loc='upper left')

        at.patch.set_boxstyle('round,pad=0.,rounding_size=0.2')
        ax.add_artist(at)
        # ax.set_title(f'rfclass Model (4_feats,n_comp={n_comp}) (80:20 Train:Test)')
        ax.set_title(r'RF Model with AD-mix $\alpha$ and $\beta$ Present Mono')
        ax.set_xlabel(r'Observed $\Delta\Delta$$G^\ddag$ (kcal/mol)')
        ax.set_ylabel(r'Predicted $\Delta\Delta$$G^\ddag$ (kcal/mol)')
        ax.set_xlim((0,3.5))
        ax.set_ylim((0,3.5))
        if not os.path.isdir(path_for_plot):
            os.mkdir(path_for_plot)
        plt.savefig(f'{path_for_plot}/{name}.png')
        plt.show()
        raise ValueError()
    
    return train_mae, test_mae, train_r2, test_r2
    print('Model Performance')
    print(f'train mae is: {train_mae}, R2 is {train_rvalue**2}')
    print(f'test mae is: {test_mae}, R2 is {test_rvalue**2}')

# print(rf_random.best_params)

base_model = RandomForestRegressor(random_state=42)
base_model.fit(X_train_var,Y_train)
train_mae,test_mae,train_r2,test_r2 = evaluate(
    base_model,
    X_train=X_train_var,
    X_test=X_test_var,
    Y_train=Y_train,
    Y_test=Y_test,
    plot=True,
    name="Default Model"
    )
print('Base Model Performance')
print(f'train mae is: {train_mae}, R2 is {train_r2}')
print(f'test mae is: {test_mae}, R2 is {test_r2}')

print("Best Random Estimator Paramaters")
# pprint(rf_random.best_params_)
# best_random = rf_random.best_estimator_

####This was only necessary because I messed up the search and didn't save it
rf_params = {'bootstrap': True,
 'max_depth': 110,
 'max_features': 'sqrt',
 'min_samples_leaf': 1,
 'min_samples_split': 2,
 'n_estimators': 1000}
best_random = RandomForestRegressor(n_estimators=rf_params['n_estimators'],max_depth=rf_params['max_depth'],max_features=rf_params['max_features'],min_samples_leaf=rf_params['min_samples_leaf'],min_samples_split=rf_params['min_samples_split'],random_state=42)
best_random.fit(X_train_var,Y_train)
#####

train_mae,test_mae,train_r2,test_r2 = evaluate(
    best_random,
    X_train=X_train_var,
    X_test=X_test_var,
    Y_train=Y_train,
    Y_test=Y_test,
    plot=True,
    name='Random Model'
    )
print(f'train mae is: {train_mae}, R2 is {train_r2}')
print(f'test mae is: {test_mae}, R2 is {test_r2}')

param_grid = {
    'bootstrap': [True],
    'max_depth': [10,30,50,70,90,110],
    'max_features': [2, 3],
    'min_samples_leaf': [1,2, 3, 4],
    'min_samples_split': [1,2,3,4],
    'n_estimators': [200,500,750, 1000,1200]
}

# rf = RandomForestRegressor()

# grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, cv = 5, n_jobs = -1, verbose = 2)
# grid_search.fit(X_train_var,Y_train)

print("Best Grid Estimator Paramaters")
best_grid_rf_params = {'bootstrap': True,
 'max_depth': 50,
 'max_features': 2,
 'min_samples_leaf': 1,
 'min_samples_split': 3,
 'n_estimators': 200}

 ####This was only necessary because I messed up the search and didn't save it
best_grid = RandomForestRegressor(n_estimators=best_grid_rf_params['n_estimators'],max_depth=best_grid_rf_params['max_depth'],max_features=best_grid_rf_params['max_features'],min_samples_leaf=best_grid_rf_params['min_samples_leaf'],min_samples_split=best_grid_rf_params['min_samples_split'],random_state=42)
best_grid.fit(X_train_var,Y_train)
#####

train_mae,test_mae,train_r2,test_r2 = evaluate(
    best_grid,
    X_train=X_train_var,
    X_test=X_test_var,
    Y_train=Y_train,
    Y_test=Y_test,
    plot=True,
    name='Grid Search'
    )
print(f'train mae is: {train_mae}, R2 is {train_r2}')
print(f'test mae is: {test_mae}, R2 is {test_r2}')
# pprint(grid_search.best_params_)
# best_grid = grid_search.best_estimator_
# train_mae,test_mae,train_r2,test_r2 = evaluate(
#     best_grid,
#     X_train=X_train_var,
#     X_test=X_test_var,
#     Y_train=Y_train,
#     Y_test=Y_test,
#     plot=True,
#     name='Grid Search'
#     )

for i in [1,2,3,4,5,6,7,8,9]:
    random_testing = RandomForestRegressor(n_estimators=best_grid_rf_params['n_estimators'],max_depth=i,max_features=best_grid_rf_params['max_features'],min_samples_leaf=best_grid_rf_params['min_samples_leaf'],min_samples_split=best_grid_rf_params['min_samples_split'],random_state=42)
    random_testing.fit(X_train_var,Y_train)
    #####

    train_mae,test_mae,train_r2,test_r2 = evaluate(
        random_testing,
        X_train=X_train_var,
        X_test=X_test_var,
        Y_train=Y_train,
        Y_test=Y_test,
        plot=True,
        name=f'Max Depth {i}'
        )
    print(f'With max depth {i}')
    print(f'train mae is: {train_mae}, R2 is {train_r2}')
    print(f'test mae is: {test_mae}, R2 is {test_r2}')