###Optimized for GBR 500_entry_altered_temp_scaled_ddG_gem_di_desc_matrix_avg.csv (No PFE, No Repeats, Average ddG Values)
_model_params = {'alpha': 0.23, #Best option if using alpha
'criterion': 'friedman_mse', #untested
'learning_rate': 0.51, #massive effect (Best at 0.5155999999999983 for loss of mean absolute error)
'loss': 'absolute_error', #Huber allows for an additional param, but not worth
'max_depth': 1, #Actually not affecting for some reason
'max_features': 'sqrt', #minor effect with max features
'max_leaf_nodes': 10, #no effect
'min_samples_leaf': 10, #no effect
'min_samples_split': 2, # No effect above 1 or in (0.0,1.0)
'min_weight_fraction_leaf': 0, #Large effect, but gets worse when above 0 through 0.5
'n_estimators': 1022, # (best regression at 1022) Decent Effect, but only in the neighborhood of about 0.1 R^2, loses accuracy both higher and lower from 392-393
'subsample': 0.34, #massive Effect
'validation_fraction': 0.1, #no effect
'warm_start': False} #no effect

###Optimized for GBR 500_entry_altered_temp_scaled_ddG_gem_di_avg_PFE_2.csv (PFE Degree 2, No Repeats, Average ddG Values)
_model_params = {'alpha': 0.23, #Best option if using alpha
'criterion': 'friedman_mse', #untested
'learning_rate': 0.27, #massive effect (Best at 0.5155999999999983 for loss of mean absolute error)
'loss': 'absolute_error', #Huber allows for an additional param, but not worth
'max_depth': 1, #Actually not affecting for some reason
'max_features': 'sqrt', #minor effect with max features
'max_leaf_nodes': 10, #no effect
'min_samples_leaf': 10, #no effect
'min_samples_split': 2, # No effect above 1 or in (0.0,1.0)
'min_weight_fraction_leaf': 0, #Large effect, but gets worse when above 0 through 0.5
'n_estimators': 420, # (best regression at 1022) Decent Effect, but only in the neighborhood of about 0.1 R^2, loses accuracy both higher and lower from 392-393
'subsample': 0.35, #massive Effect
'validation_fraction': 0.1, #no effect
'warm_start': False} #no effect

#Optimized for GBR 500_entry_altered_temp_scaled_ddG_gem_di_avg_PFE_3.csv (PFE Degree 2, No Repeats, Average ddG Values)
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