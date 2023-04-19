import numpy as np
import pandas as pd
import os
from sklearn import preprocessing as pp
from sklearn import feature_selection as fs
from sklearn.model_selection import train_test_split
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LassoCV
from sklearn.linear_model import RidgeCV
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from scipy.stats import linregress
from pprint import pprint
from sklearn.pipeline import Pipeline
from glob import glob

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
        ax.set_title(f'{name}')
        ax.set_xlabel('Observed $\Delta\Delta$$G^\ddag$ (kcal/mol)')
        ax.set_ylabel('Predicted $\Delta\Delta$$G^\ddag$ (kcal/mol)')
        ax.set_xlim((0,3.5))
        ax.set_ylim((0,3.5))
        if not os.path.isdir(path_for_plot):
            os.mkdir(path_for_plot)
        plt.savefig(f'{path_for_plot}.png')
        plt.show()
    return train_mae, test_mae, train_r2, test_r2

def check_feat_remove(pipe:Pipeline, X_train_df):
    '''
    Checks to see if features have been removed from X
    '''
    var_pipe_fit = pipe['vt'].fit(X_train_df)
    feat_in = var_pipe_fit.n_features_in_
    feat_out = np.count_nonzero(var_pipe_fit.get_support())
    feats_removed = list()
    if feat_in != feat_out:
        feats_removed = X_df.columns[~var_pipe_fit.get_support()].tolist()
        print(f'The following features were removed {feats_removed}')

def model_to_choose(model_type='GBR',random_state=42,**kwargs):

    '''
    Currently listings in this dictionary are optimized for gem disubstituted alkenes
    Current Ones Optimized: 
    GBR
    '''

    if model_type == 'GBR':
        
        _model_params = {'alpha': 0.23, #Best option if using alpha
        'criterion': 'friedman_mse', #untested
        'learning_rate': 0.38, #massive effect (Best at 0.5155999999999983 for loss of mean absolute error)
        'loss': 'absolute_error', #Huber allows for an additional param, but not worth
        'max_depth': 1, #Actually not affecting for some reason
        'max_features': 'sqrt', #minor effect with max features
        'max_leaf_nodes': 10, #no effect
        'min_samples_leaf': 10, #no effect
        'min_samples_split': 2, # No effect above 1 or in (0.0,1.0)
        'min_weight_fraction_leaf': 0, #Large effect, but gets worse when above 0 through 0.5
        'n_estimators': 420, # (best regression at 290, but bad upper data) Decent Effect, but only in the neighborhood of about 0.1 R^2, loses accuracy both higher and lower from 392-393
        'subsample': 0.35, #massive Effect
        'validation_fraction': 0.1, #no effect
        'warm_start': False} #no effect

        _model_params.update(kwargs)

        _model = GradientBoostingRegressor(
            alpha= _model_params['alpha'],
            criterion= _model_params['criterion'],
            learning_rate= _model_params['learning_rate'],
            loss= _model_params['loss'],
            max_depth= _model_params['max_depth'],
            max_features= _model_params['max_features'],
            max_leaf_nodes= _model_params['max_leaf_nodes'],
            min_samples_leaf= _model_params['min_samples_leaf'],
            min_samples_split= _model_params['min_samples_split'],
            min_weight_fraction_leaf= _model_params['min_weight_fraction_leaf'],
            n_estimators= _model_params['n_estimators'],
            subsample= _model_params['subsample'],
            validation_fraction= _model_params['validation_fraction'],
            warm_start= _model_params['warm_start'],
            random_state=random_state
        )
    elif model_type == 'RF':
        _model_params = {'n_estimators':1595,#Medium Impact
        'criterion':'friedman_mse', #currently only functional with friedman_mse
        'max_depth':3, #High impact, continuously increases to the maximum amount
        'min_samples_split':0.09, #Decent impact works between (0.0,1.0] (best at 0.09) and integers >= 1 (best at 9)
        'min_samples_leaf':1, #High but useless impact, Works between (0,0.5] and ints >= 1(best at 0)
        'min_weight_fraction_leaf':0.02,#High but useless impact, works between [0,0.5], best near 0
        'max_features':'log2', #High Impact, but for this, sqrt,log2, and 2 give the same
        'max_leaf_nodes':None, #Low Impact
        'min_impurity_decrease':1.3, #Low Impact To Modeling until above 2 which decreases
        'bootstrap':True, #Large Impact on Shape of Curve, when False underfits
        'oob_score':False,#No Impact
        'warm_start':False,#No Impact
        'ccp_alpha':0.0,#Large Impact on Shape, tends to underfit. Good for late stage optimization (0.0377)
        'max_samples':None} #No Impact

        _model_params.update(kwargs)

        _model = RandomForestRegressor(
            n_estimators=_model_params['n_estimators'],
            criterion=_model_params['criterion'],
            max_depth=_model_params['max_depth'],
            min_samples_split=_model_params['min_samples_split'],
            min_samples_leaf=_model_params['min_samples_leaf'],
            min_weight_fraction_leaf=_model_params['min_weight_fraction_leaf'],
            max_features=_model_params['max_features'],
            max_leaf_nodes=_model_params['max_leaf_nodes'],
            min_impurity_decrease=_model_params['min_impurity_decrease'],
            bootstrap=_model_params['bootstrap'],
            oob_score=_model_params['oob_score'],
            warm_start=_model_params['warm_start'],
            ccp_alpha=_model_params['ccp_alpha'],
            max_samples=_model_params['max_samples'],
            random_state=random_state
        )
    elif model_type == 'ETR':
        _model_params = {'n_estimators': 1000,
            'criterion': 'friedman_mse',
            'max_depth': 6,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'min_weight_fraction_leaf': 0.05,
            'max_features': 'log2',
            'max_leaf_nodes': None,
            'min_impurity_decrease': 0.0,
            'bootstrap': True,
            'oob_score': False,
            'warm_start': True,
            'ccp_alpha': 0.0,
            'max_samples': None}

        _model_params.update(kwargs)

        _model = ExtraTreesRegressor(n_estimators=_model_params['n_estimators'],
            criterion=_model_params['criterion'],
            max_depth=_model_params['max_depth'],
            min_samples_split=_model_params['min_samples_split'],
            min_samples_leaf=_model_params['min_samples_leaf'],
            min_weight_fraction_leaf=_model_params['min_weight_fraction_leaf'],
            max_features=_model_params['max_features'],
            max_leaf_nodes=_model_params['max_leaf_nodes'],
            min_impurity_decrease=_model_params['min_impurity_decrease'],
            bootstrap=_model_params['bootstrap'],
            oob_score=_model_params['oob_score'],
            warm_start=_model_params['warm_start'],
            ccp_alpha=_model_params['ccp_alpha'],
            max_samples=_model_params['max_samples'],
            random_state=random_state
            )
    elif model_type == 'PLS':
        _model_params = {'n_components': 4}

        _model_params.update(kwargs)

        _model = PLSRegression(
            n_components=_model_params['n_components']
        )
    elif model_type == 'LassoCV':
        _model_params = {'eps': 0.001,
        'n_alphas': 100,
        'cv': 5}

        _model_params.update(kwargs)

        _model = LassoCV(
            eps=_model_params['eps'],
            n_alphas=_model_params['n_alphas'],
            cv=_model_params['cv'],
            random_state=random_state
        )
    elif model_type == 'RidgeCV':
        _model_params = {'fit_intercept':True,
        'cv': 5,
        'gcv_mode':'auto'}

        _model_params.update(kwargs)

        _model = RidgeCV(
            fit_intercept=_model_params['fit_intercept'],
            cv=_model_params['cv'],
            gcv_mode=_model_params['gcv_mode']
        )
    elif model_type == 'SVR':
        _model_params = {'kernel':'poly',
        'degree':3,
        'gamma':'scale',
        'coef0':0.0,
        'epsilon':0.1,
        'shrinking': True}

        _model_params.update(kwargs)

        _model = SVR(
            kernel=_model_params['kernel'],
            degree=_model_params['degree'],
            gamma=_model_params['gamma'],
            coef0=_model_params['coef0'],
            epsilon=_model_params['epsilon'],
            shrinking=_model_params['shrinking']
        )
    else:
        raise ValueError(f'Not Calibrated for Model Type: {model_type}')

    return _model_params, _model

# full_df = pd.read_csv('Step_8_Altered_Temp_Scaled_ddG_Full_No_Type_Desc_Matrix_Avg.csv', index_col=0)

models_to_test = ['GBR']

# indvl_csv = ['500_entry_altered_temp_scaled_ddG_gem_di_desc_matrix_avg.csv']
# indvl_csv = ['500_entry_altered_temp_scaled_ddG_gem_di_avg_PFE_2.csv']
indvl_csv = ['500_entry_altered_temp_scaled_ddG_gem_di_avg_PFE_3.csv']

model_dict = {
    'GBR': GradientBoostingRegressor(random_state=42),
    'RF': RandomForestRegressor(random_state=42),
    'ETR': ExtraTreesRegressor(random_state=42),
    'PLS': PLSRegression(),
    'SVR': SVR(),
    'LassoCV': LassoCV(random_state=42),
    'RidgeCV': RidgeCV(),
}
title = r'GBR Model PFE Degree 3 Mono'
path_for_plot = './default_model/gbr_mono_pfe_3'
##############
for csv in indvl_csv:
    full_df = pd.read_csv(csv, index_col=0)
    file_name = '_'.join(csv.split('_')[6:8])

    csv_comment=file_name

    for _model_type in models_to_test:
        print(f'Evaluating {_model_type} on {file_name}')

        X_df = full_df.iloc[:,:-1]
        # X_val = X_df.values
        X_val = X_df.values
        # print(X_val)
        Y_df = full_df['ddG er (kcal/mol)']

        Y_val = Y_df.values
        # Y_val = Y_df.values

        X_train, X_test, Y_train, Y_test = train_test_split(X_df,Y_df, test_size=0.20,random_state=42)
        # print(X_train.shape)
        # raise ValueError()
        # for i in np.arange(0.01, 0.99, 0.01):
        model_params, model_test = model_to_choose(
            model_type=_model_type,
            random_state=42,
            #You can add any type of kwarg to alter individual parameters of interest (i.e. learning_rate=0.833 will update the dict)
            # criterion = 'poisson'
            # subsample=i
            # max_features=15
            # max_iter=100000
            # degree=2
            )
    
        # print(f'Using the following parameters')
        # pprint(model_params,sort_dicts=False)
        # print()

        pipe = Pipeline(steps=[
            ('scaler', pp.MinMaxScaler()),
            ('vt', fs.VarianceThreshold(threshold=0)),
            ('model', model_test)
        ])

        train_mae,test_mae,train_r2,test_r2 = evaluate(
            pipe,
            X_train=X_train,
            X_test=X_test,
            Y_train=Y_train,
            Y_test=Y_test,
            plot=True,
            name=title,
            path_for_plot=path_for_plot
            )
        # print(i)
        print(f'train mae is: {train_mae}, R2 is {train_r2}')
        print(f'test mae is: {test_mae}, R2 is {test_r2}')
        print()
