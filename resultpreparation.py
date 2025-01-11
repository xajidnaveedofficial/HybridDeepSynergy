
import pandas as pd

from sklearn.preprocessing import QuantileTransformer, quantile_transform
import matplotlib.pyplot as plt

from pycaret.regression import *
from pycaret.utils import check_metric

def metrics_func(true_label,prediction):
    lst = []
    lst.append(check_metric(true_label, prediction,'MAE',))
    lst.append(check_metric(true_label, prediction,'MSE'))
    lst.append(check_metric(true_label, prediction,'RMSE'))
    lst.append(check_metric(true_label, prediction,'R2'))
    lst.append(check_metric(true_label,prediction,'RMSLE'))
    lst.append(check_metric(true_label, prediction,'MAPE'))
    return lst

panCancer_train_df = pd.read_csv('data/panCancer_train_CGC_657_new_drug_features_v4_df.csv')
GBM_train_df = pd.read_csv('/data/GBM_train_CGC_657_new_drug_features_v4_df.csv')

drug_screen_Result = pd.read_csv('/Project-D2GNETs/result/unseen_predict.csv')
GBM_Carmustine = pd.read_csv('/result/GBM_Carmustine_pred.csv')
GBM_Temozolomide = pd.read_csv('/result/GBM_Temozolomide_pred.csv')

IC50_transformer = load_model('/models/LNIC50_Model/Final_et_LNIC50_Model_25June2022')



info_cols =['TCGA_DESC', 'SAMPLE_ID',
       'DRUG_NAME', 'LN_IC50_t_Real']

col = {'LN_IC50_t':'LN_IC50_t_Real',
        'Label':'LN_IC50_t'}
drug_screen_Result.rename(columns=col,inplace=True)
GBM_Carmustine.rename(columns=col,inplace=True)
GBM_Temozolomide.rename(columns=col,inplace=True)

col1 = {'LN_IC50_t':'LN_IC50_t_Predicted',
        'Label':'LN_IC50_Predicted',
      'LN_IC50':'LN_IC50_Real'}

t_pred = predict_model(IC50_transformer, data=GBM_Carmustine[['LN_IC50','LN_IC50_t']])

metrics_func(t_pred.LN_IC50, t_pred.Label)

t_pred =pd.concat([GBM_Carmustine[info_cols],t_pred],axis=1)

t_pred.rename(columns=col1,inplace=True)



t_pred1 = predict_model(IC50_transformer, data=GBM_Temozolomide[['LN_IC50','LN_IC50_t']])

metrics_func(t_pred1.LN_IC50, t_pred1.Label)

t_pred1 =pd.concat([GBM_Temozolomide[info_cols],t_pred1],axis=1)
t_pred1.rename(columns=col1,inplace=True)

t_pred2 = predict_model(IC50_transformer, data=drug_screen_Result[['LN_IC50','LN_IC50_t']])

metrics_func(t_pred2.LN_IC50, t_pred2.Label)

drug_screen_Result.columns

metrics_func(drug_screen_Result.LN_IC50_t_Real, drug_screen_Result.LN_IC50_t)

t_pred2 =pd.concat([drug_screen_Result[info_cols],t_pred2],axis=1)
t_pred2.rename(columns=col1,inplace=True)



t_pred.to_csv('/GBM_Carmustine_Result_rescaled_LN_IC50.csv',index=False)
t_pred1.to_csv('/GBM_Temozolomide_Result_rescaled_LN_IC50.csv',index=False)
t_pred2.to_csv('/result/drug_screen_Result_rescaled_LN_IC50.csv',index=False)

tcga_desc = list(drug_screen_Result['TCGA_DESC'].unique())

screenRes =[['TCGA_DESC','MAE', 'MSE', 'RMSE','R2','RMSLE','MAPE']]
for x in tcga_desc:
    tdf = t_pred2[t_pred2['TCGA_DESC']==x]
    res = metrics_func(tdf.LN_IC50_Real, tdf.LN_IC50_Predicted)
    screenRes.append([x] + res)

screenResDF = pd.DataFrame(screenRes[1:],columns=screenRes[0])

screenResDF.to_csv('/TCGS_DESC_DrugSCREEN_RESULT.csv',index=False)

screenRes =[['TCGA_DESC','MAE', 'MSE', 'RMSE','R2','RMSLE','MAPE']]
for x in tcga_desc:
    tdf = t_pred2[t_pred2['TCGA_DESC']==x]
    res = metrics_func(tdf.LN_IC50_t_Real, tdf.LN_IC50_t_Predicted)
    screenRes.append([x] + res)
screenResDF = pd.DataFrame(screenRes[1:],columns=screenRes[0])
screenResDF.to_csv('/TCGS_DESC_DrugSCREEN_RESULT_on_scaled_target.csv',index=False)



repurposeRes = [['Carmustine','GBM']+metrics_func(t_pred.LN_IC50_Real, t_pred.LN_IC50_Predicted),
                ['Temozolomide','GBM']+metrics_func(t_pred1.LN_IC50_Real, t_pred1.LN_IC50_Predicted)
            ]

repurposeResDF = pd.DataFrame(repurposeRes, columns=['Drug_Name','TCGS_DESC','MAE', 'MSE', 'RMSE','R2','RMSLE','MAPE'])

repurposeResDF.to_csv('/data/user/rsharma3/nbotw/Project-D2GNETs/result/GBM_Repurposing_RESULT.csv',index=False)

repurposeRes = [['Carmustine','GBM']+metrics_func(t_pred.LN_IC50_t_Real, t_pred.LN_IC50_t_Predicted),
                ['Temozolomide','GBM']+metrics_func(t_pred1.LN_IC50_t_Real, t_pred1.LN_IC50_t_Predicted)
            ]
repurposeResDF = pd.DataFrame(repurposeRes, columns=['Drug_Name','TCGS_DESC','MAE', 'MSE', 'RMSE','R2','RMSLE','MAPE'])
repurposeResDF.to_csv('/result/GBM_Repurposing_RESULT_on_scaled_target.csv',index=False)



t_predx1 =t_pred.copy(deep=True)
t_predx1['SAMPLE_ID'] = t_predx1['SAMPLE_ID'].astype('string')
t_predx1.rename(columns={'LN_IC50_Real':'Real','LN_IC50_Predicted':'Predicted'},inplace=True)
ax = t_predx1.plot.line(y='Real', figsize=(20, 10), rot=90, marker='.',markersize=20,title='Repurposing of Carmustine for GBM')
t_predx1.plot.line(y='Predicted',figsize=(20, 10), ax=ax,rot=90 , marker='.',markersize=15)
ax.set_xticks(t_predx1.index)
ax.set_xticklabels(t_predx1['SAMPLE_ID'],Fontsize=15)
#ax.set_yticklabels(Fontsize=20)
ax.set_xlabel("SAMPLE ID", fontdict={'fontsize':20})
ax.set_ylabel("LN (IC50)",fontdict={'fontsize':20})
ax.legend(loc=2,fontsize=20)
ax.title.set_size(30)
ax.figure.savefig('/Project-D2GNETs/plots/Carmustine_repurposing.pdf')

t_predx1 =t_pred1.copy(deep=True)
t_predx1['SAMPLE_ID'] = t_predx1['SAMPLE_ID'].astype('string')
t_predx1.rename(columns={'LN_IC50_Real':'Real','LN_IC50_Predicted':'Predicted'},inplace=True)
ax = t_predx1.plot.line(y='Real', figsize=(20, 10), rot=90, marker='.',markersize=20,title='Repurposing of Temozolomide for GBM')
t_predx1.plot.line(y='Predicted',figsize=(20, 10), ax=ax,rot=90 , marker='.',markersize=15)
ax.set_xticks(t_predx1.index)
ax.set_xticklabels(t_predx1['SAMPLE_ID'],Fontsize=15)
#ax.set_yticklabels(Fontsize=20)
ax.set_xlabel("SAMPLE ID", fontdict={'fontsize':20})
ax.set_ylabel("LN (IC50)",fontdict={'fontsize':20})
ax.legend(loc=2,fontsize=20)
ax.title.set_size(30)
ax.figure.savefig('/plots/Temozolomide_repurposing.pdf')

from matplotlib import pyplot as plt

import seaborn as sns

t_predx1.columns

ctr =1
for k, v in t_predx1.iterrows():
    print(k)
    print(v[[9]])
    ctr+=1
    if ctr==4:
        break

cancer = list(t_pred2['TCGA_DESC'].unique())

t_pred2['DRUG_CODE'] = t_pred2['DRUG_NAME'].astype('category').cat.codes

for cname in cancer:
    fig, ax = plt.subplots(constrained_layout=True)
    t_predx1 =t_pred2[t_pred2['TCGA_DESC']==cname].copy(deep=True)
    t_predx1.sort_values(by=['DRUG_CODE'],inplace=True)
    t_predx1.reset_index(inplace=True)
    t_predx1.reset_index(inplace=True)
    t_predx1['SAMPLE_ID'] = t_predx1['SAMPLE_ID'].astype('string')

    t_predx1.rename(columns={'LN_IC50_Real':'Real','LN_IC50_Predicted':'Predicted'},inplace=True)
    t_predx1.plot.line(ax=ax,y='Real', rot=90, marker='.',title=f'Drug Screening for {cname}')
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

    ax.figure.savefig(f'/plots/{cname}_screening.pdf')

t_pred2.groupby('TCGA_DESC').count()

cancer_list =['COREAD','NB','LGG','LIHC','BRCA','PRAD','KIRC','ESCA','GBM']

ax = t_pred2[t_pred2['TCGA_DESC'].isin(cancer_list)].boxplot(
    column=['LN_IC50_Real','LN_IC50_Predicted'],rot=90,figsize=(50, 40),fontsize=20)

t_pred2.columns



tcga = 'GBM'
ax = sns.boxplot(data=(t_pred2[t_pred2['TCGA_DESC']==tcga])[['LN_IC50_Real','LN_IC50_Predicted']])
ax.set_ylabel('LN_IC50')
ax.set_title('GBM')
ax.set_ylim(-10,10)
ax.figure.savefig(f'/plots/result_comparison_{tcga}.pdf')

cancer = list(t_pred2['TCGA_DESC'].unique())

df_lst =[]
for x in cancer:
    dft = (t_pred2[t_pred2['TCGA_DESC']==x])[['LN_IC50_Real','LN_IC50_Predicted']]
    tdf = pd.DataFrame(dft.values,columns=[f'{x}-Real',f'{x}-Predicted'])
    df_lst.append(tdf)

new_df = pd.concat(df_lst, axis=1)

c_list = []
pal ={}
for x in cancer_list:
    c_list.append(f'{x}-Real')
    pal.update({f'{x}-Real':'g'})
    c_list.append(f'{x}-Predicted')
    pal.update({f'{x}-Predicted':'b'})

new_df.to_csv('/plots/result_comparison_.csv')

ax = sns.boxplot(data=new_df[c_list], palette=pal)
ax.set_ylabel('LN_IC50',fontsize=40)
ax.set_xlabel('Cancer Types',fontsize=40)
ax.set_title('Comparison of nine cancer type for Real and Predicted LN_IC50',fontsize=50)
ax.set_ylim(-8.5,10)
ax.set_xticklabels(ax.get_xticklabels(),rotation=70)
ax.figure.set_figwidth(35)
ax.figure.set_figheight(15)
ax.tick_params(labelsize=30)
ax.autoscale_view()
ax.figure.savefig(f'/plots/result_comparison_.pdf')



ax = t_pred2[t_pred2['TCGA_DESC']=='GBM'].boxplot(
    column=['LN_IC50_Real','LN_IC50_Predicted'],by=['DRUG_NAME'],grid=False,
                                                  layout=(2,1),rot=90,figsize=(50, 0),fontsize=20)

ax[0].title

t_pred2.columns

import seaborn as sns

sns.set_theme(style="whitegrid")

tips = sns.load_dataset("tips")

t_predx =t_pred2[t_pred2['TCGA_DESC']=='GBM'].pivot(index=['SAMPLE_ID'],columns=['DRUG_NAME'],values='LN_IC50_Predicted')

t_predx.head()

ax = sns.boxplot(data=t_predx, orient="h", palette="Set2")
ax.figure.set_figheight(40)
ax.figure.set_figwidth(15)

featureScore = pd.read_csv('/data/user/rsharma3/nbotw/Project-D2GNETs/Supplementary_Material/featureScore_v4.csv')
featureScore = featureScore[featureScore['Score']>=0]
features = set(featureScore['Name'].to_list())

disease_cols = (pd.read_csv('/data/user/rsharma3/nbotw/Project-D2GNETs/data/diseaseNames.csv')).columns.tolist()
GE_cols =  (pd.read_csv('/data/user/rsharma3/nbotw/Project-D2GNETs/data/657_Gene_name.csv')).columns.tolist()
drug_cols =  (pd.read_csv('/data/user/rsharma3/nbotw/Project-D2GNETs/data/moleculeNames.csv')).columns.tolist()

#list(features.intersection(set(panCancer_train_df.columns[55+2986:].to_list())))

Drug_Xcols = list(features.intersection(set(drug_cols)))+disease_cols
GE_Xcols = GE_cols#list(features.intersection(set(panCancer_train_df.columns[55+2986:].to_list())))

model = load_model('/data/user/rsharma3/nbotw/Project-D2GNETs/models/LGBM/Final_lgbm_Model_25June2022')

model

drug_cols = list(set(GBM_Carmustine).intersection(set(drug_cols)))
Disease_Cols = list(set(GBM_Carmustine).intersection(set(disease_cols)))
Genes = list(set(GBM_Carmustine).intersection(set(GE_Xcols)))

print(f"No. of Diseases: {len(Disease_Cols)}")
print(f"No. of Drugs Features: {len(drug_cols)}")
print(f"No. of GENEs: {len(Genes)}")



print(f"No. of Diseases: {len(disease_cols)}")
print(f"No. of Drugs Features: {len(Drug_Xcols)}")
print(f"No. of GENEs: {len(GE_Xcols)}")

GBM_test_Carmustine = GBM_train_df[GBM_train_df['DRUG_NAME']=='Carmustine'].copy(deep=True)
#GBM_test_Docetaxel = GBM_test_df[GBM_test_df['DRUG_NAME']=='Docetaxel'].copy(deep=True)
#GBM_test_Crizotinib = GBM_test_df[GBM_test_df['DRUG_NAME']=='Crizotinib'].copy(deep=True)
GBM_test_Temozolomide = GBM_train_df[GBM_train_df['DRUG_NAME']=='Temozolomide'].copy(deep=True)

indx_4_drugs = (GBM_test_Carmustine.index.to_list()+
 GBM_test_Temozolomide.index.to_list())
GBM_train_df.drop(index=indx_4_drugs,inplace=True)
panCancer_df = pd.concat([panCancer_train_df,GBM_train_df],axis=0)
panCancer_df = panCancer_df[Drug_Xcols+GE_Xcols+['TCGA_DESC','LN_IC50','SAMPLE_ID','DRUG_NAME']]
panCancer_df.dropna(inplace=True)
train = panCancer_df.sample(frac=0.8).copy(deep=True)#.values
test = panCancer_df.drop(index=train.index).copy(deep=True)

qt1 = QuantileTransformer(n_quantiles=2, random_state=42,output_distribution='uniform')
#qt2 = QuantileTransformer(n_quantiles=200, random_state=0)

train['LN_IC50_t'] = qt1.fit_transform(
    train['LN_IC50'].values.reshape(-1, 1))
test['LN_IC50_t'] = qt1.fit_transform(
    test['LN_IC50'].values.reshape(-1, 1))

#test_unseen['LN_IC50_t'] = qt1.fit_transform(test_unseen['LN_IC50'].values.reshape(-1, 1))

GBM_test_Carmustine['LN_IC50_t'] = qt1.fit_transform(GBM_test_Carmustine['LN_IC50'].values.reshape(-1, 1))
GBM_test_Temozolomide['LN_IC50_t'] = qt1.fit_transform(GBM_test_Temozolomide['LN_IC50'].values.reshape(-1, 1))

train_transform_df = train[['LN_IC50','LN_IC50_t']].copy(deep=True)

t_pred = predict_model(IC50_transformer, data=train_transform_df)

t_pred.rename(columns={'Label':'LN_IC50_inversed','LN_IC50_t':'LN_IC50_scaled'},inplace=True)

metrics_func(t_pred.LN_IC50, t_pred.Label)

f, (ax0, ax1,ax2) = plt.subplots(1, 3)

ax0.hist(t_pred['LN_IC50'].values, bins=1000, density=True)
ax0.set_xlim([-10, 10])
ax0.set_ylabel("Probability")
ax0.set_xlabel("Target")
ax0.set_title("Original Distribution")

ax1.hist(t_pred['LN_IC50_scaled'], bins=1000, density=True)
ax1.set_ylabel("Probability")
ax1.set_xlabel("Target")
ax1.set_title("Transformed distribution")

ax2.hist(t_pred['LN_IC50_inversed'], bins=1000, density=True)
ax2.set_ylabel("Probability")
ax2.set_xlabel("Target")
ax2.set_title("Inverse-tranformed distribution")


f.suptitle("GDSC Drug Perturbations training data", y=0.06, x=0.53)
f.tight_layout(rect=[0.05, 0.05, .95, .95])

f.savefig('/data/user/rsharma3/nbotw/Project-D2GNETs/plots/target_scaling_train_data.pdf')

#plotting for test data
transform_df = test[['LN_IC50','LN_IC50_t']].copy(deep=True)
t_pred = predict_model(IC50_transformer, data=transform_df)
t_pred.rename(columns={'Label':'LN_IC50_inversed','LN_IC50_t':'LN_IC50_scaled'},inplace=True)
f, (ax0, ax1,ax2) = plt.subplots(1, 3)

ax0.hist(t_pred['LN_IC50'].values, bins=1000, density=True)
ax0.set_xlim([-10, 10])
ax0.set_ylabel("Probability")
ax0.set_xlabel("Target")
ax0.set_title("Original Target Distribution")

ax1.hist(t_pred['LN_IC50_scaled'], bins=1000, density=True)
ax1.set_ylabel("Probability")
ax1.set_xlabel("Target")
ax1.set_title("Transformed target distribution")

ax2.hist(t_pred['LN_IC50_inversed'], bins=1000, density=True)
ax2.set_ylabel("Probability")
ax2.set_xlabel("Target")
ax2.set_title("Inverse-tranformed target distribution")


f.suptitle("GDSC Drug Perturbations test data", y=0.06, x=0.53)
f.tight_layout(rect=[0.05, 0.05, 1.95, 1.95])

f.savefig('/plots/target_scaling_test_data.pdf')

#plotting for test data
transform_df = GBM_test_Carmustine[['LN_IC50','LN_IC50_t']].copy(deep=True)
t_pred = predict_model(IC50_transformer, data=transform_df)
t_pred.rename(columns={'Label':'LN_IC50_inversed','LN_IC50_t':'LN_IC50_scaled'},inplace=True)
f, (ax0, ax1,ax2) = plt.subplots(1, 3)

ax0.hist(t_pred['LN_IC50'].values, bins=1000, density=True)
ax0.set_xlim([-10, 10])
ax0.set_ylabel("Probability")
ax0.set_xlabel("Target")
ax0.set_title("Original Target Distribution")

ax1.hist(t_pred['LN_IC50_scaled'], bins=1000, density=True)
ax1.set_ylabel("Probability")
ax1.set_xlabel("Target")
ax1.set_title("Transformed target distribution")

ax2.hist(t_pred['LN_IC50_inversed'], bins=1000, density=True)
ax2.set_ylabel("Probability")
ax2.set_xlabel("Target")
ax2.set_title("Inverse-tranformed target distribution")


f.suptitle("GDSC Drug Perturbations for Carmustine", y=0.06, x=0.53)
f.tight_layout(rect=[0.1, 0.1, 1.95, 1.95])

f.savefig('Project-D2GNETs/plots/target_scaling_Carmustine_data.pdf')

#plotting for test data
transform_df = GBM_test_Temozolomide[['LN_IC50','LN_IC50_t']].copy(deep=True)
t_pred = predict_model(IC50_transformer, data=transform_df)
t_pred.rename(columns={'Label':'LN_IC50_inversed','LN_IC50_t':'LN_IC50_scaled'},inplace=True)
f, (ax0, ax1,ax2) = plt.subplots(1, 3)

ax0.hist(t_pred['LN_IC50'].values, bins=1000, density=True)
ax0.set_xlim([-10, 10])
ax0.set_ylabel("Probability")
ax0.set_xlabel("Target")
ax0.set_title("Original Target Distribution")

ax1.hist(t_pred['LN_IC50_scaled'], bins=1000, density=True)
ax1.set_ylabel("Probability")
ax1.set_xlabel("Target")
ax1.set_title("Transformed target distribution")

ax2.hist(t_pred['LN_IC50_inversed'], bins=1000, density=True)
ax2.set_ylabel("Probability")
ax2.set_xlabel("Target")
ax2.set_title("Inverse-tranformed target distribution")


f.suptitle("GDSC Drug Perturbations for Temozolomide", y=0.06, x=0.53)
f.tight_layout(rect=[0.05, 0.05, 1.95, 1.95])

f.savefig('/plots/target_scaling_Temozolomide_data.pdf')

57+47+14

