###Optimized for RF 500_entry_altered_temp_scaled_ddG_mono_basic_compound_features.csv (No PFE, Both Values Present)
_model_params = {'bootstrap': True,
 'max_depth': 110,
 'max_features': 'sqrt',
 'min_samples_leaf': 1,
 'min_samples_split': 2,
 'n_estimators': 1000}

###Optimized for RF 500_entry_altered_temp_scaled_ddG_mono_basic_compound_features.csv (No PFE, Only One Value Present)
_model_params = {'bootstrap': True,
'max_depth': 50,
'max_features': 2,
'min_samples_leaf': 1,
'min_samples_split': 3,
'n_estimators': 200}

###Optimized for GBR 500_entry_altered_temp_scaled_ddG_mono_desc_matrix_avg.csv (No PFE, No Repeats, Average ddG Values)
_model_params = {'alpha': 0.2, #no effect due to criterion
'criterion': 'friedman_mse', #untested
'learning_rate': 0.883, #massive effect
'loss': 'absolute_error', #untested
'max_depth': 1, #test r2 decreases, train r2 increases at larger
'max_features': 'sqrt', #minor effect with max features
'max_leaf_nodes': 10, #no effect
'min_samples_leaf': 10, #no effect
'min_samples_split': 2, # No effect above 1 or in (0.0,1.0)
'min_weight_fraction_leaf': 0, #Large effect, but gets worse when above 0 through 0.5
'n_estimators': 408, #Decent Effect, but only in the neighborhood of about 0.1 R^2, loses accuracy both higher and lower from 392-393
'subsample': 0.34, #massive Effect
'validation_fraction': 0.1, #no effect
'warm_start': False} #no effect

###Optimized for GBR 500_entry_altered_temp_scaled_ddG_mono_avg_PFE_2.csv (PFE Degree 2, No Repeats, Average ddG Values)
_model_params = {'alpha': 0.2, #no effect due to criterion
'criterion': 'friedman_mse', #untested
'learning_rate': 0.25, #massive effect
'loss': 'absolute_error', #untested
'max_depth': 1, #test r2 decreases, train r2 increases at larger
'max_features': 'sqrt', #minor effect with max features
'max_leaf_nodes': 10, #no effect
'min_samples_leaf': 10, #no effect
'min_samples_split': 2, # No effect above 1 or in (0.0,1.0)
'min_weight_fraction_leaf': 0, #Large effect, but gets worse when above 0 through 0.5
'n_estimators': 251, #Decent Effect, but only in the neighborhood of about 0.1 R^2, loses accuracy both higher and lower from 392-393
'subsample': 0.25, #massive Effect
'validation_fraction': 0.1, #no effect
'warm_start': False} #no effect

#Optimized for GBR 500_entry_altered_temp_scaled_ddG_mono_avg_PFE_3.csv (PFE Degree 2, No Repeats, Average ddG Values)
_model_params = {'alpha': 0.2, #no effect due to criterion
'criterion': 'friedman_mse', #untested
'learning_rate': 0.51, #massive effect
'loss': 'absolute_error', #untested
'max_depth': 1, #test r2 decreases, train r2 increases at larger
'max_features': 'sqrt', #minor effect with max features
'max_leaf_nodes': 10, #no effect
'min_samples_leaf': 10, #no effect
'min_samples_split': 2, # No effect above 1 or in (0.0,1.0)
'min_weight_fraction_leaf': 0, #Large effect, but gets worse when above 0 through 0.5
'n_estimators': 334, # (251) Decent Effect, but only in the neighborhood of about 0.1 R^2, loses accuracy both higher and lower from 392-393
'subsample': 0.27, #massive Effect
'validation_fraction': 0.1, #no effect
'warm_start': False} #no effect