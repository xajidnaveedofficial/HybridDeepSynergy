
#!pip install pubchempy

!pwd

import sys
from sklearn.preprocessing import StandardScaler, MaxAbsScaler, MinMaxScaler, QuantileTransformer, PowerTransformer,Normalizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

from rdkit import Chem

import pandas as pd
import numpy as np
import pubchempy as pcp

"""# Loading data

### Cell Line Gene Expression data from COSMIC
"""

data = pd.read_csv('/data/AllGeneExpression.csv')

"""### Drug REsponse and Drug Information data from GDSC"""

df_GDSC = pd.read_csv('/GDSC2_fitted_dose_response_25Feb20_3.csv')
df_GDSC_DrugInfo = pd.read_csv('/data/Drug_Desciption_GDSC.csv')



"""# Drug Data Preparation"""

lst = []

ctr =0
for x in df_GDSC_DrugInfo['drug_name'].unique().tolist():
  c = pcp.get_compounds(x, 'name')

  if len(c) != 0:
    tlst = []
    tlst.append(x)
    tlst.append(int(c[0].cid))
    tlst.append(c[0].isomeric_smiles)
    lst.append(tlst)
  else:
    ctr += 1
    tlst = []
    tlst.append(x)
    tlst.append(None)
    tlst.append(None)
    lst.append(tlst)

df = pd.DataFrame(lst, columns=['DRUG_NAME','PUBCHEM_ID','SMILES'])

df.dropna(inplace=True)
df.isna().sum()

df= pd.read_csv('/data/SMILES_FeatureEngineered.csv')
df = df[['DRUG_NAME', 'PUBCHEM_ID', 'SMILES']].copy(deep=True)



### Using NLP to extract features from Drug SMILES

# function to convert sequence strings into k-mer words, default size = 6 (hexamer words)
def getKmers(sequence, size=6):
  if sequence!=None:

    return [sequence[x:x+size].lower() for x in range(len(sequence) - size + 1)]
  else:
    return None

from io import StringIO
sio = sys.stderr = StringIO()

lst

# function to convert sequence strings into k-mer words, default size = 6 (hexamer words)
def getKmers(sequence, size=6):
  if sequence!=None:
    lst =[]
    X =None
    for i in [1]:
    #for i in range(1,int(len(sequence)*0.50)):
        for x in range(len(sequence) - i + 1):
            try:

                X=Chem.MolFromSmiles(sequence[x:x+i])
                sio = sys.stderr = StringIO()
            except SyntaxError:
                pass
            if X is None:
                continue
            else:
                lst.append(sequence[x:x+i])
    return lst
  else:
    return None

lst = getKmers('CCNC(=O)C1=NOC(=C1C2=CC=C(C=C2)CN3CCOCC3)C4=CC(=C(C=C4O)O)C(C)C')

df['WORDS'] = df.apply(lambda x: getKmers(x['SMILES']), axis=1)

mols = []
for x in list(df['WORDS']):
    mols = mols + x

setx = list(set(mols))

len(setx)

molName = [f'Mol{i}' for i in range(1,len(setx)+1)]

molDict = {}
ctr=1
for i in setx:
    molDict.update({i:f'Mol{ctr}'})
    ctr+=1

mol = pd.DataFrame(np.asarray([setx,molName]).T,columns=['Molecule','tokenName'])

mol.to_csv('/data/Molecule_token_map.csv',index=False)

def mapWords(lst=None):
    if lst!=None:
        tlst = []
        for x in lst:
            tlst.append(molDict[x])
        return tlst
    else:
        return None



df['WordMap']=df.apply(lambda x: mapWords(x['WORDS']), axis=1)

df_texts = list(df['WordMap'])
for item in range(len(df_texts)):
    df_texts[item] = ' '.join(df_texts[item])





# Creating the Bag of Words model using CountVectorizer()
# This is equivalent to k-mer counting
# The n-gram size of 4 was previously determined by testing
cv = CountVectorizer(ngram_range=(8,8))
X = cv.fit_transform(df_texts)

#from sklearn.feature_extraction.text import CountVectorizer
#tf_transformer = TfidfTransformer(use_idf=False).fit(X)
#cv = TfidfTransformer(ngram_range=(1,1),use_idf=False)
#X = tf_transformer.transform(X)

count_vect_df = pd.DataFrame(X.todense(), columns=cv.get_feature_names())

#dff = pd.concat([df.reset_index(), count_vect_df], axis =1, ignore_index= False)

dff = pd.concat([dff, count_vect_df], axis =1, ignore_index= False)

print('number of drug selected: %d'%dff.shape[0])
print('number of features created: %d'%dff.shape[1])

dff[dff.columns[6:]].max().max()

dff[dff.columns[6:]]

#saving the dataFrame for future reference
dff.to_csv('/data/SMILES_FeatureEngineered_new.csv',index=False)

max_abs_scaler = StandardScaler()
d1_1 = max_abs_scaler.fit_transform(dff[dff.columns[6:]])
d1 = pd.DataFrame(d1_1,columns=dff.columns[6:].to_list())
dff1 = pd.concat([dff[dff.columns[:6]],d1],axis=1)
#dff1= dff.copy(deep=True)

dff1.to_csv('/data/SMILES_FeatureEngineered_new_scaled.csv',index=False)

dff1[dff.columns[6:]][:0].to_csv('/data/moleculeNames_v1.csv',index=False)

dff1[dff.columns[6:]][:0]

s = pd.Series(list(df_GDSC['TCGA_DESC'].unique()))

s.dropna(inplace=True)

s1=pd.get_dummies(s)

disease = pd.concat([s,s1],axis=1)

disease.rename(columns={0:'TCGA_DESC'}, inplace=True)

disease[disease.columns[1:]][:0].to_csv('/data/diseaseNames.csv',index=False)

#joining the GDSC Dataframe with Feature Engineered Drug Smiles
df1 = pd.merge(df_GDSC,disease, on=['TCGA_DESC'], how='left')
print(df1.shape)
df1 = pd.merge(df1,dff1, on=['DRUG_NAME'], how='left')
print(df_GDSC.shape)
print(df1.shape)

df1.drop(columns=['DATASET', 'NLME_RESULT_ID', 'NLME_CURVE_ID','PUTATIVE_TARGET', 'CELL_LINE_NAME', 'SANGER_MODEL_ID',
                         'PATHWAY_NAME', 'COMPANY_ID','WEBRELEASE', 'MIN_CONC', 'MAX_CONC','SMILES', 'WORDS', 'AUC', 'RMSE',
                         'Z_SCORE', 'index'],inplace=True)

df1.drop(index=df1[df1.isna().any(axis=1)].index,inplace=True)

"""### Saving the merged DataFrame of Drug Respnse and Feature Engineered drugs"""

df1.to_csv('/data/user/rsharma3/nbotw/Project-D2GNETs/data/GDSC_Drug_Feature_Engineered_new_scaled.csv',index=False)

dfx = df1[(df1['TCGA_DESC']=='GBM')&(df1['DRUG_NAME']=='Carmustine')]



GDSC = pd.merge(df_GDSC, df_GDSC_DrugInfo, on=[])

GE_df = pd.read_csv('/content/drive/MyDrive/COSMIC/GE_dimension_reduced_RANS_657.csv')
Drug_df = pd.read_csv('/content/drive/MyDrive/GDSCdata/GDSC_Drug_Feature_Engineered_with_RANs.csv')

censusGN = pd.read_csv('/content/drive/MyDrive/GDSCdata/Census_allSun May 22 17_29_44 2022.csv')

censusGN.head()

GE_df = pd.read_csv('/content/drive/MyDrive/COSMIC/AllGeneExpression.csv')
Drug_df = pd.read_csv('/content/drive/MyDrive/GDSCdata/GDSC_Drug_Feature_Engineered_new.csv')

GE_df.columns

X_cols = list(set(GE_df.columns).intersection(set(censusGN['Gene Symbol'].values)))

GE_df = GE_df[['SAMPLE_ID']+X_cols]

#Drug_df = pd.read_csv('/content/drive/MyDrive/GDSCdata/GDSC_Drug_Feature_Engineered_with_RANs.csv')

GE_df.drop(columns=['Unnamed: 0'],inplace=True)



#GE_df = pd.read_csv('/content/drive/MyDrive/COSMIC/AllGeneExpression.csv')
#Drug_df = pd.read_csv('/content/drive/MyDrive/GDSCdata/GDSC_Drug_Feature_Engineered.csv')


GE_df.dropna(inplace=True)

Drug_df.rename(columns={'COSMIC_ID':'SAMPLE_ID'},inplace=True)
Drug_df.dropna(inplace=True)

GBM_DRUG_LIST = ['Crizotinib','Docetaxel','Temozolomide', 'Camptothecin','Carmustine', 'Irinotecan','Nimotuzumab','Temozolomide']
TARGET = 'LN_IC50'
IRRELEVANT_FETURES = ['DATASET', 'NLME_RESULT_ID', 'NLME_CURVE_ID', 'COSMIC_ID',
       'CELL_LINE_NAME', 'SANGER_MODEL_ID', 'TCGA_DESC', 'DRUG_ID',
       'DRUG_NAME', 'PUTATIVE_TARGET', 'PATHWAY_NAME', 'COMPANY_ID',
       'WEBRELEASE', 'MIN_CONC', 'MAX_CONC', 'AUC', 'RMSE',
       'Z_SCORE', 'index', 'PUBCHEM_ID', 'SMILES', 'WORDS', 'SAMPLE_ID',]

Drug_df_GBM = Drug_df[Drug_df['TCGA_DESC']=='GBM'].copy(deep=True)

Drug_df.drop(Drug_df_GBM.index,inplace=True)

GBM_known_drugs_test_df =  Drug_df_GBM[(Drug_df_GBM['DRUG_NAME'].isin(GBM_DRUG_LIST))].copy(deep=True)

Drug_df_GBM.drop(GBM_known_drugs_test_df.index, inplace=True)

panCancer_train_df = pd.merge(Drug_df,GE_df, on=['SAMPLE_ID'], how='left')
GBM_train_df = pd.merge(Drug_df_GBM,GE_df, on=['SAMPLE_ID'], how='left')
GBM_test_df = pd.merge(GBM_known_drugs_test_df,GE_df, on=['SAMPLE_ID'], how='left')

Drug_df= None
GE_df = None
Drug_df_GBM = None
GBM_known_drugs_test_df = None

panCancer_train_df.columns[:50]

panCancer_train_df.dropna(inplace=True)

panCancer_train_df.to_csv('/content/drive/MyDrive/COSMIC/panCancer_train_657_RANS_df.csv',index=False)

panCancer_train_df=None

GBM_train_df.dropna(inplace=True)
GBM_train_df.to_csv('/content/drive/MyDrive/COSMIC/GBM_train_657_RANS_df.csv',index=False)
#GBM_train_df = None

GBM_test_df.dropna(inplace=True)
GBM_test_df.to_csv('/content/drive/MyDrive/COSMIC/GBM_test_657_RANS_df.csv',index=False)
#GBM_test_df = None

len(X_cols)

genes = np.asarray(data.columns[1:])

genes.shape

gene1 = np.reshape(genes,(-1,24))

gene1 = pd.DataFrame(gene1)

gene1.to_csv('/Supplementary_Material/genes16248.csv', index=False, header=False)

df_GDSC.columns

((df_GDSC[['COSMIC_ID','CELL_LINE_NAME','TCGA_DESC']].drop_duplicates()).dropna()).to_csv('//TCGA_classes_of_Cell_lines.csv',index=False)

