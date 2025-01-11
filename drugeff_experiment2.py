
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import QuantileTransformer, quantile_transform
from sklearn.metrics import mean_absolute_error,mean_squared_error,  r2_score
from pycaret.regression import *
from pycaret.utils import check_metric

"""# Load Data"""

panCancer_train_df = pd.read_csv('../../data/panCancer_train_CGC_657_DR_Drug_features_new_df.csv')
GBM_train_df = pd.read_csv('../../data/GBM_train_CGC_657_DR_Drug_features_new_df.csv')

panCancer_df = pd.concat([panCancer_train_df,GBM_train_df],axis=0)

"""# Feature Engineeing (Step-2)"""

# Removing Features based on Feature Score less than 70
featureScore = pd.read_csv('../../supplementary_material/featureScore.csv')
featureScore = featureScore[featureScore['Score']>=70]
features = set(featureScore['Name'].to_list())

# Getting the column names
disease_cols = (pd.read_csv('../../data/diseaseNames.csv')).columns.tolist()
GE_cols =  (pd.read_csv('../../data/657_Gene_name.csv')).columns.tolist()
drug_cols =  (pd.read_csv('../../data/moleculeNames_v1.csv')).columns.tolist()

Drug_Xcols = list(features.intersection(set(drug_cols)))+disease_cols
GE_Xcols = GE_cols#list(features.intersection(set(panCancer_train_df.columns[55+2986:].to_list())))

print(f"No. of Diseases: {len(disease_cols)}")
print(f"No. of Drugs Features: {len(Drug_Xcols)}")
print(f"No. of GENEs: {len(GE_Xcols)}")

"""# Data Preparation for ML operation

### Data for Drug Repurposing Experiments
"""

GBM_test_Carmustine = panCancer_df[(panCancer_df['DRUG_NAME']=='Carmustine')
                                   &(panCancer_df['TCGA_DESC']=='GBM')].copy(deep=True)
GBM_test_Temozolomide = panCancer_df[(panCancer_df['DRUG_NAME']=='Temozolomide')
                                     &(panCancer_df['TCGA_DESC']=='GBM')].copy(deep=True)

"""### Data for Drug Screening Experiments"""

indx_4_drugs = (GBM_test_Carmustine.index.to_list()+
 GBM_test_Temozolomide.index.to_list())
panCancer_df.drop(index=indx_4_drugs,inplace=True)

#panCancer_df = pd.concat([panCancer_train_df,GBM_train_df],axis=0)
panCancer_df = panCancer_df[Drug_Xcols+GE_Xcols+['TCGA_DESC','LN_IC50','SAMPLE_ID','DRUG_NAME']]
panCancer_df.dropna(inplace=True)

train = panCancer_df.sample(frac=0.8).copy(deep=True)# Training data for Drug Screening
test = panCancer_df.drop(index=train.index).copy(deep=True) # Test Data for Drug Screening

print(f"Train data size: {train.shape[0]}")
print(f"Test data size: {test.shape[0]}")

"""# Quantile Transformation of Target (LN_IC50)"""

qt1 = QuantileTransformer(n_quantiles=2, random_state=42,output_distribution='uniform')
#qt2 = QuantileTransformer(n_quantiles=200, random_state=0)

train['LN_IC50_t'] = qt1.fit_transform(
    train['LN_IC50'].values.reshape(-1, 1))
test['LN_IC50_t'] = qt1.fit_transform(
    test['LN_IC50'].values.reshape(-1, 1))

#test_unseen['LN_IC50_t'] = qt1.fit_transform(test_unseen['LN_IC50'].values.reshape(-1, 1))

GBM_test_Carmustine['LN_IC50_t'] = qt1.fit_transform(GBM_test_Carmustine['LN_IC50'].values.reshape(-1, 1))
GBM_test_Temozolomide['LN_IC50_t'] = qt1.fit_transform(GBM_test_Temozolomide['LN_IC50'].values.reshape(-1, 1))

"""# Random Forest Model for Quantile Inverse Transformation"""

exp1 = setup(data=train[['LN_IC50_t','LN_IC50']], target='LN_IC50', test_data=test[['LN_IC50_t','LN_IC50']],numeric_features=['LN_IC50_t']
           ,transformation = False, normalize=False,use_gpu=True,polynomial_features=True,polynomial_degree=3)

best = compare_models()

et_LNIC50 = create_model('et')

et_res= pull()

et_res.to_csv('../../result/LabelModel10fold.csv')

predict_model(et_LNIC50);

final_et_LNIC50 = finalize_model(et_LNIC50)

save_model(final_et_LNIC50,'../../models/LNIC50_Model/Final_et_LNIC50_Model_25June2022')

"""# Drug Screening Model development"""

exp = setup(data=train, target='LN_IC50', test_data=test,  ignore_features=['TCGA_DESC','LN_IC50_t','SAMPLE_ID','DRUG_NAME'],numeric_features=disease_cols+Drug_Xcols+GE_Xcols
           ,transformation = False, normalize=False,use_gpu=True)

models()

mdl1= compare_models(include=['lr','knn','gbr','dt','ridge','ada','lasso','rf','lightgbm','et'])

#mdl1_1 = pull()
#mdl1_1.to_csv('../../result/modelSelection.csv')

lr = create_model('lr')

lgbm = create_model('lightgbm')

x=pd.DataFrame({'Feature': get_config('X_train').columns}).to_csv('../../result/feature.csv')

lgbm_res= pull()

lgbm_res.to_csv('../../result/LGBM10fold.csv')

final_lgbm = finalize_model(lgbm)
save_model(final_et_LNIC50,'../../models/LGBM/Final_lgbm_Model_25June2022')

"""# Drug Screening experiment with Test data"""

# Prediction
unseen_predict =predict_model(lgbm, data=test)

#saving the resutlt
unseen_predict.to_csv('../../result/unseen_predict.csv',index=False)

"""# Drug Repurposing Experiment"""

# Prediction
GBM_Carmustine_pred = predict_model(final_lgbm, data=GBM_test_Carmustine)
GBM_Temozolomide_pred = predict_model(final_lgbm, data=GBM_test_Temozolomide)

# Saving the Results of drug Repurposing
GBM_Carmustine_pred.to_csv('../../result/GBM_Carmustine_pred.csv',index=False)
GBM_Temozolomide_pred.to_csv('../../result/GBM_Temozolomide_pred.csv',index=False)