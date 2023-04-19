#New Mono (Train/Test Split seeded with random_state=42)
_model_params = {'alpha': 0.04,
'criterion': 'friedman_mse',
'learning_rate': 0.04,
'loss': 'absolute_error',
'max_depth': 6,
'max_features': 'sqrt',
'max_leaf_nodes': 10,
'min_samples_leaf': 1,
'min_samples_split': 5,
'min_weight_fraction_leaf': 0,
'n_estimators': 254,
'subsample': 0.66,
'validation_fraction': 0.1,
'warm_start': True}

#New Gem Di (Train/Test Split seeded with random_state=42)
_model_params = {'alpha': 0.3,
'criterion': 'friedman_mse',
'learning_rate': 0.44,
'loss': 'absolute_error',
'max_depth': 3,
'max_features': 'sqrt',
'max_leaf_nodes': 10,
'min_samples_leaf': 10,
'min_samples_split': 2,
'min_weight_fraction_leaf': 0,
'n_estimators' :411,
'subsample': 0.25,
'validation_fraction': 0.1,
'warm_start': True}

#Cis disubstituted (Also Train/Test Split seeded with random_state=81)
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
'n_estimators': 301,
'subsample': 0.2,
'validation_fraction': 0.1,
'warm_start': True}

#Trans disubstituted (Train/Test Split seeded with random_state=42)
_model_params = {'alpha': 0.3,
'criterion': 'friedman_mse',
'learning_rate': 0.9,
'loss': 'absolute_error',
'max_depth': 3,
'max_features': 'sqrt',
'max_leaf_nodes': None,
'min_samples_leaf': 1,
'min_samples_split': 5,
'min_weight_fraction_leaf': 0,
'n_estimators' : 301,
'subsample': 0.7,
'validation_fraction': 0.1,
'warm_start': True}

#Trisubstituted (Train/Test Split seeded with random_state=10)
_model_params = {'alpha': 0.3,
'criterion': 'friedman_mse',
'learning_rate': 0.67,
'loss': 'absolute_error',
'max_depth': 3,
'max_features': 'sqrt',
'max_leaf_nodes': 10,
'min_samples_leaf': 10,
'min_samples_split': 2,
'min_weight_fraction_leaf': 0,
'n_estimators': 480,
'subsample': 0.95,
'validation_fraction': 0.1,
'warm_start': True}

#Tetrasubstituted Olefin (SVR) (Train/Test Split seeded with random_state=42)
_model_params = {'C': 1,
'coef0': 0.25,
'degree': 3,
'gamma': 'scale',
'kernel': 'poly',
'shrinking': True}
