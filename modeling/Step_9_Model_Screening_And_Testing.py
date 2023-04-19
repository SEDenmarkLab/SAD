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
        ax.set_title(f'{name} (80:20 Train:Test)')
        ax.set_xlabel('Observed ddG (kcal/mol)')
        ax.set_ylabel('Predicted ddG (kcal/mol)')
        ax.set_xlim((0,3.5))
        ax.set_ylim((0,3.5))
        if not os.path.isdir(path_for_plot):
            os.mkdir(path_for_plot)
        plt.savefig(f'{path_for_plot}/{name}.png')
        # plt.show()
    return train_mae, test_mae, train_r2, test_r2

def prep_pipe_comment(model_type='PLS',path_comment='pipe_test',param_comment='Default',csv_comment='Default'):
    path_for_plot = f'./{model_type}_{path_comment}'
    name = f"{model_type} {param_comment} {csv_comment}"
    return name,path_for_plot

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
        _model_params = {'alpha': 0.3,
        'criterion': 'friedman_mse',
        'learning_rate': 0.060000000000000005,
        'loss': 'absolute_error',
        'max_depth': 1,
        'max_features': 'sqrt',
        'max_leaf_nodes': None,
        'min_samples_leaf': 1,
        'min_samples_split': 2,
        'min_weight_fraction_leaf': 0.3,
        'n_estimators': 300,
        'subsample': 0.2,
        'validation_fraction': 0.1,
        'warm_start': True}

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

def test_one_param(model_type, name, parameter_test:str,min_val,max_val,inc,path_for_plot):
    _model_params = dict()
    #for manual screening
    all_test_dict = dict()
    for i in np.arange(min_val,max_val,inc):
        _model_params[parameter_test] = i
        _,model_test = model_to_choose(
        model_type=_model_type,
        random_state=42,
        #You can add any type of kwarg to alter individual parameters of interest (i.e. learning_rate=0.833 will update the dict)
        **_model_params
        )

        pipe = Pipeline(steps=[
            ('scaler', pp.MinMaxScaler()),
            ('vt', fs.VarianceThreshold(threshold=0)),
            # ('selector',fs.RFE(model_test,n_features_to_select=7)),
            ('model', model_test)
        ])

        train_mae,test_mae,train_r2,test_r2 = evaluate(
            pipe,
            X_train=X_train,
            X_test=X_test,
            Y_train=Y_train,
            Y_test=Y_test,
            plot=False,
            name=name,
            path_for_plot=path_for_plot
            )

        print(i)
        print(f'train mae is: {train_mae}, R2 is {train_r2}')
        print(f'test mae is: {test_mae}, R2 is {test_r2}')
        print()
        all_test_dict[i] = test_r2

    all_key_arr = np.array(list(all_test_dict.keys()))
    all_r2_arr = np.array(list(all_test_dict.values()))
    best_r2_sort = np.argsort(all_r2_arr)[::-1][:5]

    top_key=all_key_arr[best_r2_sort]
    top_r2 = all_r2_arr[best_r2_sort]

    top_5_list = list(zip(top_key,top_r2))
    print('These are the top 5 values')
    pprint(top_5_list)

    fig = plt.figure()
    plt.plot(all_test_dict.keys(), all_test_dict.values(), 'bo-')
    plt.title(f'Plotting {model_type} {parameter_test}', fontsize=16)
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.ylabel('Total Explained Variance', fontsize=12)
    plt.xticks(np.arange(0, max_val+inc*5, inc*10))
    plt.xlabel('Number of Components', fontsize=12)
    if not os.path.isdir(path_for_plot):
        os.mkdir(path_for_plot)
    fig.savefig(f'{path_for_plot}/{model_type}_{parameter_test}_test.png')

def test_two_param(model_type, name, param_1_test, param_1_min_val,param_1_max_val,param_1_inc, param_2_test, param_2_min_val,param_2_max_val,param_2_inc,path_for_plot):
    _model_params = dict()
    #for manual screening
    all_test_dict = dict()
    for j in np.arange(param_1_min_val,param_1_max_val, param_1_inc):
        _model_params[param_1_test] = j
        for i in np.arange(param_2_min_val,param_2_max_val,param_2_inc):
            _model_params[param_2_test] = i
            _,model_test = model_to_choose(
            model_type=_model_type,
            random_state=42,
            #You can add any type of kwarg to alter individual parameters of interest (i.e. learning_rate=0.833 will update the dict)
            **_model_params
            )

            pipe = Pipeline(steps=[
                ('scaler', pp.MinMaxScaler()),
                ('vt', fs.VarianceThreshold(threshold=0)),
                # ('selector',fs.RFE(model_test,n_features_to_select=7)),
                ('model', model_test)
            ])

            train_mae,test_mae,train_r2,test_r2 = evaluate(
                pipe,
                X_train=X_train,
                X_test=X_test,
                Y_train=Y_train,
                Y_test=Y_test,
                plot=False,
                name=name,
                path_for_plot=path_for_plot
                )

            print(f'{param_1_test} = {j}, {param_2_test} = {i}')
            print(f'train mae is: {train_mae}, R2 is {train_r2}')
            print(f'test mae is: {test_mae}, R2 is {test_r2}')
            print()
            all_test_dict[(f'{param_1_test}={j},{param_2_test}={i}')] = test_r2

    all_key_arr = np.array(list(all_test_dict.keys()))
    all_r2_arr = np.array(list(all_test_dict.values()))
    best_r2_sort = np.argsort(all_r2_arr)[::-1][:10]

    top_key=all_key_arr[best_r2_sort]
    top_r2 = all_r2_arr[best_r2_sort]

    top_10_list = list(zip(top_key,top_r2))
    print('These are the top 10 values')
    pprint(top_10_list)

def prep_grid(**kwargs):
    all_grid_vals = dict()
    all_grid_vals.update(kwargs)
    return all_grid_vals

def random_search_cv(trans_X_train, Y_train, estimator, scoring, grid, n_iter, cv, verbose=0, n_procs=-1, random_state=42):
    random_search = RandomizedSearchCV(
        estimator = estimator, 
        param_distributions = grid, 
        n_iter = n_iter, 
        scoring=scoring,
        cv = cv, 
        verbose=verbose, 
        random_state=random_state, 
        n_jobs = n_procs)
    random_search.fit(trans_X_train, Y_train)

    print('After a Random Grid Search, the following parameters are the best:')
    pprint(random_search.best_params_, sort_dicts=False)

    best_random_estimator = random_search.best_estimator_

    return best_random_estimator

def grid_search_cv(trans_X_train, Y_train, estimator, scoring, grid, cv, verbose=0, n_procs=-1):
    grid_search = GridSearchCV(
        estimator = estimator, 
        param_grid = grid, 
        scoring=scoring,
        cv = cv, 
        verbose=verbose, 
        n_jobs = n_procs)
    grid_search.fit(trans_X_train, Y_train)

    print('After a Grid Search, the following parameters are the best:')
    pprint(grid_search.best_params_, sort_dicts=False)

    best_grid_estimator = grid_search.best_estimator_

    return best_grid_estimator

#One Parameter Screen
new_param='n_estimators'
min_value = 280
# max_value = full_df.columns.shape[0]-2
max_value = 320
inc=1

#Two Parameter Screen
new_param1 ='learning_rate'
param1_min = 0.1
param1_max = 1
param1_inc = 0.05

new_param2 ='subsample'
param2_min = 0.1
param2_max = 1
param2_inc = 0.05

#####################################################

# all_csv = glob('*Avg.csv')
# indvl_csv = [csv for csv in all_csv if "Full" not in csv]
full_no_type_csv = glob('*Full*.csv')
# indvl_csv = ['Step_8_Altered_Temp_Scaled_ddG_Gem_Di_Desc_Matrix_Avg.csv']
# indvl_csv = ['Step_8_Altered_Temp_Scaled_ddG_Tri_Only_Desc_Matrix_Avg.csv']
# indvl_csv = ['Step_8_Altered_Temp_Scaled_ddG_Trans_di_Desc_Matrix_Avg.csv']
indvl_csv = ['Step_8_Altered_Temp_Scaled_ddG_Cis_Di_Desc_Matrix_Avg.csv']
# full_df = pd.read_csv('Step_8_Altered_Temp_Scaled_ddG_Full_No_Type_Desc_Matrix_Avg.csv', index_col=0)
# full_df = pd.read_csv('Step_8_Altered_Temp_Scaled_ddG_Gem_Di_Desc_Matrix_Avg.csv', index_col=0)

####Deciding models and if its a parameter screen####
# models_to_test = ['GBR','RF','ETR','PLS', 'SVR', 'LassoCV','RidgeCV']
# models_to_test = ['PLS', 'SVR','RidgeCV']
# models_to_test = ['SVR']
models_to_test = ['GBR']
# models_to_test = ['PLS']

random_state_train_test = 42


#Various screen types that can be done
# screen_type = 'model_test'
screen_type = 'one_param_screen'
# screen_type = 'two_param_screen'
# screen_type = 'random_screen'
# screen_type = 'grid_screen'

folder_path=screen_type
title_comment = 'Poly Grid Search Opt'
############################################
#Example Grid Screen for GBR
screen_grid = prep_grid(
    alpha= [0.3, 0.6, 0.9],
    criterion= ['friedman_mse'],
    learning_rate= [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
    loss= ['absolute_error'],
    max_depth= [1, 2, 3, 5, 10],
    max_features= ['sqrt'],
    max_leaf_nodes= [None, 10],
    min_samples_leaf= [1, 10],
    min_samples_split= [2, 5],
    min_weight_fraction_leaf= [0, 0.3],
    subsample= [0.2, 0.4, 0.6, 0.8, 0.95],
    validation_fraction= [0.1],
    warm_start= [True, False]
)

model_dict = {
    'GBR': GradientBoostingRegressor(random_state=42),
    'RF': RandomForestRegressor(random_state=42),
    'ETR': ExtraTreesRegressor(random_state=42),
    'PLS': PLSRegression(),
    'SVR': SVR(),
    'LassoCV': LassoCV(random_state=42),
    'RidgeCV': RidgeCV(),
}

##############
for csv in indvl_csv:
    full_df = pd.read_csv(csv, index_col=0)
    file_name = '_'.join(csv.split('_')[6:8])

    csv_comment=file_name

    for _model_type in models_to_test:
        print(f'Evaluating {_model_type} on {file_name}')

        name,plot_path = prep_pipe_comment(
            model_type=_model_type,
            path_comment=folder_path,
            param_comment=title_comment,
            csv_comment=csv_comment
        )

        X_df = full_df.iloc[:,:-1]
        # X_val = X_df.values

        Y_df = full_df['ddG er (kcal/mol)']
        # Y_val = Y_df.values

        X_train, X_test, Y_train, Y_test = train_test_split(X_df,Y_df, test_size=0.20,random_state=random_state_train_test)

        if screen_type == 'one_param_screen':
            test_one_param(
                model_type=_model_type,
                name=name,
                parameter_test=new_param,
                min_val=min_value,
                max_val=max_value,
                inc=inc,
                path_for_plot=plot_path
            )
        elif screen_type == 'two_param_screen':
            test_two_param(
                model_type=_model_type,
                name=name,
                param_1_test=new_param1,
                param_1_min_val=param1_min,
                param_1_max_val=param1_max,
                param_1_inc=param1_inc,
                param_2_test=new_param2,
                param_2_min_val=param2_min,
                param_2_max_val=param2_max,
                param_2_inc=param2_inc,
                path_for_plot=plot_path
            )
        elif screen_type == 'random_screen':
            pipe = Pipeline(steps=[
                ('scaler', pp.MinMaxScaler()),
                ('vt', fs.VarianceThreshold(threshold=0)),
            ])

            estimator = model_dict[_model_type]
            vt_X_train = pipe['vt'].fit_transform(X_train)

            best_random_estimator = random_search_cv(
                trans_X_train = vt_X_train,
                Y_train=Y_train,
                estimator=estimator,
                grid=screen_grid,
                scoring='neg_mean_absolute_error',
                n_iter=100,
                cv=5,
                verbose=1,
                random_state=42,
                n_procs= 64
                )

            pipe.steps.append(['model', best_random_estimator])

            train_mae,test_mae,train_r2,test_r2 = evaluate(
                pipe,
                X_train=X_train,
                X_test=X_test,
                Y_train=Y_train,
                Y_test=Y_test,
                plot=True,
                name=name,
                path_for_plot=plot_path
                )
            
            print(f'train mae is: {train_mae}, R2 is {train_r2}')
            print(f'test mae is: {test_mae}, R2 is {test_r2}')
            print()
        elif screen_type == 'grid_screen':
            pipe = Pipeline(steps=[
                ('scaler', pp.MinMaxScaler()),
                ('vt', fs.VarianceThreshold(threshold=0)),
            ])

            estimator = model_dict[_model_type]
            vt_X_train = pipe['vt'].fit_transform(X_train)

            best_grid_estimator = grid_search_cv(
                trans_X_train = vt_X_train,
                Y_train=Y_train,
                estimator=estimator,
                grid=screen_grid,
                scoring='neg_mean_absolute_error',
                cv=5,
                verbose=1,
                n_procs= 64
                )

            pipe.steps.append(['model', best_grid_estimator])

            train_mae,test_mae,train_r2,test_r2 = evaluate(
                pipe,
                X_train=X_train,
                X_test=X_test,
                Y_train=Y_train,
                Y_test=Y_test,
                plot=True,
                name=name,
                path_for_plot=plot_path
                )
            print(f'train mae is: {train_mae}, R2 is {train_r2}')
            print(f'test mae is: {test_mae}, R2 is {test_r2}')
            print()
        elif screen_type == 'model_test':
            
            model_params, model_test = model_to_choose(
                model_type=_model_type,
                random_state=42,
                #You can add any type of kwarg to alter individual parameters of interest (i.e. learning_rate=0.833 will update the dict)
                # criterion = 'poisson'
                # max_features=15
                # max_iter=100000
                # degree=2
            )
            
            print(f'Using the following parameters')
            pprint(model_params,sort_dicts=False)
            print()

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
                name=name,
                path_for_plot=plot_path
                )
            print(f'train mae is: {train_mae}, R2 is {train_r2}')
            print(f'test mae is: {test_mae}, R2 is {test_r2}')
            print()
        else:
            raise NotImplementedError(f'Screen Type {screen_type} not implemented.')
