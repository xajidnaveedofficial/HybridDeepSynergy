
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

repurposing_df = pd.read_csv('../../data/Repurposing.csv')

"""# Feature Engineeing (Step-2)"""

# Removing Features based on Feature Score less than 70
featureScore = pd.read_csv('../../supplementary_material/featureScore.csv')
featureScore = featureScore[featureScore['Score']>=70]
features = set(featureScore['Name'].to_list())

# Getting the column names
disease_cols = (pd.read_csv('../../data/diseaseNames.csv')).columns.tolist()
GE_cols =  (pd.read_csv('../../data/657_Gene_name.csv')).columns.tolist()
drug_cols =  (pd.read_csv('../../data/moleculeNames_v1.csv')).columns.tolist()

Drug_Xcols = list(features.intersection(set(drug_cols)))
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
panCancer_df = panCancer_df[disease_cols+Drug_Xcols+GE_Xcols+['TCGA_DESC','SAMPLE_ID','DRUG_NAME','LN_IC50']]
panCancer_df.dropna(inplace=True)

train = panCancer_df.sample(frac=0.8).copy(deep=True)# Training data for Drug Screening
test = panCancer_df.drop(index=train.index).copy(deep=True) # Test Data for Drug Screening

print(f"Train data size: {train.shape[0]}")
print(f"Test data size: {test.shape[0]}")

train.columns[:34]

cols = Drug_Xcols+GE_Xcols+['TCGA_DESC','SAMPLE_ID','DRUG_NAME','LN_IC50']

repurposing_df = repurposing_df[Drug_Xcols+GE_Xcols+['TCGA_DESC','SAMPLE_ID','DRUG_NAME']]

"""# Drug Screening Model development"""

exp = setup(data=train, target='LN_IC50', test_data=test,  ignore_features=['TCGA_DESC','SAMPLE_ID','DRUG_NAME'],numeric_features=disease_cols+Drug_Xcols+GE_Xcols
           ,transformation = False, normalize=False,use_gpu=True)

exp = setup(data=train[train['TCGA_DESC']=='GBM'], target='LN_IC50',
            test_data=test[test['TCGA_DESC']=='GBM'],  ignore_features=['TCGA_DESC','SAMPLE_ID','DRUG_NAME'],numeric_features=Drug_Xcols+GE_Xcols+disease_cols
           ,transformation = False, normalize=False,use_gpu=True)

exp = setup(data=(train[train['TCGA_DESC']=='GBM'])[cols], target='LN_IC50',
            test_data=(test[test['TCGA_DESC']=='GBM'])[cols],  ignore_features=['TCGA_DESC','SAMPLE_ID','DRUG_NAME'],numeric_features=Drug_Xcols+GE_Xcols
           ,transformation = False, normalize=False,use_gpu=True)

#mdl1= compare_models(include=['lr','knn','gbr','dt','ridge','ada','lasso','rf','lightgbm','et'])

mdl1= compare_models()

mdl1_1 = pull()
mdl1_1.to_csv('../../result/modelSelection_KIRC.csv')

lr = create_model('lightgbm')

lgbm = create_model('lightgbm')

x=pd.DataFrame({'Feature': get_config('X_train').columns}).to_csv('../../result/feature.csv')

lgbm_res= pull()

lgbm_res.to_csv('../../result/LGBM10fold_KIRC.csv')

final_lgbm = finalize_model(lgbm)
save_model(final_lgbm,'../../models/LGBM/Final_lgbm_Model_25June2022')

"""# Drug Screening experiment with Test data"""

# Prediction
unseen_predict =predict_model(lgbm, data=test[cols])

#saving the resutlt
unseen_predict.to_csv('../result/unseen_predict.csv',index=False)

"""# Drug Repurposing Experiment"""

# Prediction
GBM_Carmustine_pred = predict_model(final_lgbm, data=GBM_test_Carmustine)
GBM_Temozolomide_pred = predict_model(final_lgbm, data=GBM_test_Temozolomide)

# Saving the Results of drug Repurposing
GBM_Carmustine_pred.to_csv('../../result/GBM_Carmustine_pred.csv',index=False)
GBM_Temozolomide_pred.to_csv('../../result/GBM_Temozolomide_pred.csv',index=False)

unseen_predict =predict_model(lgbm, data=(test[test['TCGA_DESC']=='GBM'])[cols])

def metrics_func(true_label,prediction):
    lst = []
    lst.append(check_metric(true_label, prediction,'MAE',))
    lst.append(check_metric(true_label, prediction,'MSE'))
    lst.append(check_metric(true_label, prediction,'RMSE'))
    lst.append(check_metric(true_label, prediction,'R2'))
    lst.append(check_metric(true_label,prediction,'RMSLE'))
    lst.append(check_metric(true_label, prediction,'MAPE'))
    return lst

metrics_func(unseen_predict['LN_IC50'],unseen_predict['Label'])

unseen_predict1 =predict_model(lgbm, data=repurposing_df)

unseen_predict1.to_csv('../../result/unseen_predict_repurpose.csv',index=False)

unseen_predict1

2662-657-31

unseen_predict['DRUG_CODE'] = unseen_predict['DRUG_NAME'].astype('category').cat.codes

fig, ax = plt.subplots(constrained_layout=True)
t_predx1 =unseen_predict.copy(deep=True)
t_predx1.sort_values(by=['DRUG_CODE'],inplace=True)
t_predx1.reset_index(inplace=True)
t_predx1.reset_index(inplace=True)
t_predx1['SAMPLE_ID'] = t_predx1['SAMPLE_ID'].astype('string')

t_predx1.rename(columns={'LN_IC50':'Real','Label':'Predicted'},inplace=True)
t_predx1.plot.line(ax=ax,y='Real', rot=90, marker='.',title=f'Drug Screening for KIRC')
t_predx1.plot.line(y='Predicted', ax=ax,rot=90 , marker='.')
t_predx1.plot(y='Predicted',x='level_0',kind='scatter',ax=ax)
#ax.set_xticks(t_predx1.index)
#ax.set_xticklabels(t_predx1['SAMPLE_ID'],Fontsize=5)
#ax.set_yticklabels(Fontsize=20)
ax.set_xlabel("SAMPLES", fontdict={'fontsize':20})
ax.set_ylabel("LN (IC50)",fontdict={'fontsize':20})
ax.legend(loc=2,fontsize=10)
ax.title.set_size(30)
#secax = ax.secondary_xaxis('top')
#secax.set_xticks(t_predx1.DRUG_CODE)
#secax.set_xticklabels(t_predx1.DRUG_CODE,Fontsize=5,rotation = 90)

ax.figure.savefig(f'../../plots/KIRC_screening.pdf')

unseen_predict1['DRUG_CODE'] = unseen_predict1['DRUG_NAME'].astype('category').cat.codes

samples = unseen_predict1['SAMPLE_ID'].unique()

for samp in samples:
    fig, ax = plt.subplots(constrained_layout=True)
    t_predx1 =unseen_predict1.copy(deep=True)
    t_predx1 = t_predx1[t_predx1['SAMPLE_ID']==samp]
    t_predx1.sort_values(by=['DRUG_CODE'],inplace=True)
    t_predx1.reset_index(inplace=True)
    t_predx1.reset_index(inplace=True)
    t_predx1['SAMPLE_ID'] = t_predx1['SAMPLE_ID'].astype('string')

    t_predx1.rename(columns={'Label':'Predicted'},inplace=True)
    #t_predx1.plot.line(ax=ax,y='Real', rot=90, marker='.',title=f'Drug Screening for KIRC')
    t_predx1.plot.line(y='Predicted', ax=ax,rot=90 , marker='.',title=f'Drug Screening for Sample {samp}')
    t_predx1.plot(y='Predicted',x='DRUG_NAME',kind='scatter',ax=ax,rot=90)
    #ax.set_xticks(t_predx1.index)
    #ax.set_xticklabels(t_predx1['SAMPLE_ID'],Fontsize=5)
    #ax.set_yticklabels(Fontsize=20)
    ax.set_xlabel("Drugs", fontdict={'fontsize':20})
    ax.set_ylabel("LN (IC50)",fontdict={'fontsize':20})
    ax.legend(loc=2,fontsize=10)
    ax.title.set_size(30)
    #secax = ax.secondary_xaxis('top')
    #secax.set_xticks(t_predx1.DRUG_CODE)
    #secax.set_xticklabels(t_predx1.DRUG_CODE,Fontsize=5,rotation = 90)

    ax.figure.savefig(f'../../plots/repurpose/{samp}_repurposing.pdf')

