import numpy as np
import pandas as pd
from sklearn import preprocessing as pp
from sklearn import feature_selection as fs
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from scipy.stats import linregress
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import Pipeline
import os

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
        ax.set_title(r'RF Model with Averaged $\Delta\Delta$$G^\ddag$ AD-mix $\alpha$ and $\beta$ Mono')
        ax.set_xlabel(r'Observed $\Delta\Delta$$G^\ddag$ (kcal/mol)')
        ax.set_ylabel(r'Predicted $\Delta\Delta$$G^\ddag$ (kcal/mol)')
        ax.set_xlim((0,3.5))
        ax.set_ylim((0,3.5))
        if not os.path.isdir(path_for_plot):
            os.mkdir(path_for_plot)
        plt.savefig(f'{path_for_plot}/repeat_remove.png')
        plt.show()
        raise ValueError()
    return train_mae, test_mae, train_r2, test_r2

full_df = pd.read_csv('500_entry_altered_temp_scaled_ddG_mono_basic_compound_features.csv', index_col=0)
# full_df = pd.read_csv('p6_updated_501_altered_temp_scaled_ddG.csv', index_col=0)

#Gets B1 through temp
# X_df = full_df.iloc[:,-5:-1]
# full_df = full_df[['b1','b5','espmin','espmax','b1_min_b5','b1_dot_b5','b1_add_b5','ddG er (kcal/mol)']]
X_df = full_df.iloc[:,:-1]
print(X_df)
X_df = X_df[~X_df.index.duplicated(keep='first')]
print(X_df)
# raise ValueError()

# print(X_df)
X_val = X_df.values
# print(X_val)
Y_df = full_df['ddG er (kcal/mol)']
print(Y_df)
Y_df = Y_df[~Y_df.index.duplicated(keep='first')]
print(Y_df)
Y_val = Y_df.values

X_train, X_test, Y_train, Y_test = train_test_split(X_val,Y_val,test_size=0.20,random_state=42)

best_grid_rf_params = {'bootstrap': True,
'max_depth': 50,
'max_features': 2,
'min_samples_leaf': 1,
'min_samples_split': 3,
'n_estimators': 200}

pipe = Pipeline([('scaler', pp.MinMaxScaler()), ('vt',fs.VarianceThreshold(threshold=0)), ('rf',RandomForestRegressor(n_estimators=best_grid_rf_params['n_estimators'],max_depth=5,max_features=best_grid_rf_params['max_features'],min_samples_leaf=best_grid_rf_params['min_samples_leaf'],min_samples_split=best_grid_rf_params['min_samples_split'],random_state=42))])

rf_model = pipe.fit(X_train, Y_train)

# print(rf_model['rf'].feature_importances_)
# print(X_df.columns)
# raise ValueError()
train_mae,test_mae,train_r2,test_r2 = evaluate(
    rf_model,
    X_train=X_train,
    X_test=X_test,
    Y_train=Y_train,
    Y_test=Y_test,
    plot=True,
    name=f'Best Grid Depth 5 Removed React (80_20 Train_Test)'
    )
print(f'train mae is: {train_mae}, R2 is {train_r2}')
print(f'test mae is: {test_mae}, R2 is {test_r2}')
