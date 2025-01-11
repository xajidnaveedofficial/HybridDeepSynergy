
import pandas as pd





def dataPrep(ExpName, GE_df,  Drug_df):
    GE_df.dropna(inplace=True)

    Drug_df.rename(columns={'COSMIC_ID':'SAMPLE_ID'},inplace=True)
    #Drug_df.dropna(inplace=True)
    #Drug_df.drop(columns=['DATASET', 'NLME_RESULT_ID', 'NLME_CURVE_ID','PUTATIVE_TARGET', 'CELL_LINE_NAME', 'SANGER_MODEL_ID',
     #                    'PATHWAY_NAME', 'COMPANY_ID','WEBRELEASE', 'MIN_CONC', 'MAX_CONC','SMILES', 'WORDS', 'AUC', 'RMSE',
      #                   'Z_SCORE', 'index'],inplace=True)
    GBM_DRUG_LIST = ['Crizotinib','Docetaxel','Temozolomide', 'Camptothecin','Carmustine', 'Irinotecan','Nimotuzumab','Temozolomide']
    TARGET = 'LN_IC50'
    IRRELEVANT_FETURES = ['DATASET', 'NLME_RESULT_ID', 'NLME_CURVE_ID', 'COSMIC_ID',
       'CELL_LINE_NAME', 'SANGER_MODEL_ID', 'TCGA_DESC', 'DRUG_ID',
       'DRUG_NAME', 'PUTATIVE_TARGET', 'PATHWAY_NAME', 'COMPANY_ID',
       'WEBRELEASE', 'MIN_CONC', 'MAX_CONC', 'AUC', 'RMSE',
       'Z_SCORE', 'index', 'PUBCHEM_ID', 'SMILES', 'WORDS', 'SAMPLE_ID',]

    #Drug_df.dropna(inplace=True)

    Drug_df_GBM = Drug_df[Drug_df['TCGA_DESC']=='GBM'].copy(deep=True)

    Drug_df.drop(Drug_df_GBM.index,inplace=True)

    #GBM_known_drugs_test_df =  Drug_df_GBM[(Drug_df_GBM['DRUG_NAME'].isin(GBM_DRUG_LIST))].copy(deep=True)

    #Drug_df_GBM.drop(GBM_known_drugs_test_df.index, inplace=True)

    panCancer_train_df = pd.merge(Drug_df,GE_df, on=['SAMPLE_ID'], how='left')
    GBM_train_df = pd.merge(Drug_df_GBM,GE_df, on=['SAMPLE_ID'], how='left')
    #GBM_test_df = pd.merge(GBM_known_drugs_test_df,GE_df, on=['SAMPLE_ID'], how='left')

    Drug_df= None
    GE_df = None
    Drug_df_GBM = None
    GBM_known_drugs_test_df = None

    panCancer_train_df.to_csv(f'/data/panCancer_train_{ExpName}_df.csv',index=False)
    panCancer_train_df=None

    GBM_train_df.dropna(inplace=True)
    GBM_train_df.to_csv(f'/data/GBM_train_{ExpName}_df.csv',index=False)
    GBM_train_df = None

    #GBM_test_df.dropna(inplace=True)
    #GBM_test_df.to_csv(f'//GBM_test_{ExpName}_df.csv',index=False)
    #GBM_test_df = None

"""# ML modeling Data preparation using entire gene set

### Data Praparation without Dimensions reduction
"""

GE_df = pd.read_csv('/data/AllGeneExpression.csv')
Drug_df = pd.read_csv('/data/GDSC_Drug_Feature_Engineered_new.csv')

dataPrep('All_GE_new_drug_features', GE_df, Drug_df)

"""### Data Preparation with Dimension reduced data (through RANs)"""

Drug_df = pd.read_csv('/data/GDSC_Drug_Feature_Engineered_with_RANs_new_drug_FE.csv')
GE_df = pd.read_csv('/data/GE_dimension_reduced_RANS.csv')

dataPrep('DRwRANS', GE_df, Drug_df)

"""# ML modeling Data Preparation using selected genes

### Selection based on cancer gene censue 657 genes
"""

GE_df = pd.read_csv('/data/AllGeneExpression.csv')
Drug_df = pd.read_csv('/data/GDSC_Drug_Feature_Engineered_new_scaled.csv')
censusGN = pd.read_csv('/data/Census_allFri_May_6_15_00_18_2022.csv')

GE_df.max().max()

X_cols = list(set(GE_df.columns).intersection(set(censusGN['Gene Symbol'].values)))
GE_df = GE_df[['SAMPLE_ID']+X_cols]

dataPrep('CGC_657_new_drug_features_v2', GE_df, Drug_df)

coldf = pd.DataFrame(columns=X_cols)
coldf.to_csv('/Project-D2GNETs/data/657_Gene_name.csv',index=False)

"""# ML modeling Data Preparation with CGC's 657 genes and Dimension Reduced drug features"""

GE_df = pd.read_csv('/data/AllGeneExpression.csv')
censusGN = pd.read_csv('/Census_allFri_May_6_15_00_18_2022.csv')
X_cols = list(set(GE_df.columns).intersection(set(censusGN['Gene Symbol'].values)))
GE_df = GE_df[['SAMPLE_ID']+X_cols]
Drug_df = pd.read_csv('/GDSC_Drug_Feature_Engineered_with_RANs_new_drug_FE.csv')

dataPrep('CGC_657_DR_Drug_features_new', GE_df, Drug_df)

panCancer_train_df = pd.read_csv('/data/panCancer_train_CGC_657_new_drug_features_v2_df.csv')

24+203-6

#panCancer_train_df.columns[24:221].to_list()
panCancer_train_df.columns[221:].to_list()

