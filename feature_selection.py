
# example of correlation feature selection for numerical data
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from matplotlib import pyplot
import pandas as pd

# feature selection
def select_features(X_train, y_train, X_test):
	# configure to select all features
	fs = SelectKBest(score_func=f_regression, k='all')
	# learn relationship from training data
	fs.fit(X_train, y_train)
	# transform train input data
	X_train_fs = fs.transform(X_train)
	# transform test input data
	X_test_fs = fs.transform(X_test)
	return X_train_fs, X_test_fs, fs

panCancer_train_df = pd.read_csv('/panCancer_train_CGC_657_new_drug_features_v2_df.csv')
GBM_train_df = pd.read_csv('/GBM_train_CGC_657_new_drug_features_v2_df.csv')
#GBM_test_df = pd.read_csv('/GBM_test_CGC_657_new_drug_features_v2_df.csv')

GBM_train_df[GBM_train_df.isna().any(axis=1)]

panCancer_train_df.columns[3500:]

Disease_Xcols = (pd.read_csv('/data/diseaseNames.csv')).columns.tolist()
GE_Xcols =  (pd.read_csv('/data/657_Gene_name.csv')).columns.tolist()+Disease_Xcols
Drug_Xcols =  (pd.read_csv('/data/moleculeNames_v1.csv')).columns.tolist()


panCancer_df = pd.concat([panCancer_train_df,GBM_train_df],axis=0)

panCancer_df = panCancer_df[Drug_Xcols+GE_Xcols+['TCGA_DESC','LN_IC50']]

panCancer_df.dropna(inplace=True)

#panCancer_df = panCancer_df[panCancer_df['TCGA_DESC']!='UNCLASSIFIED']

X = panCancer_df[Drug_Xcols+GE_Xcols]#.values
#X_GE = panCancer_df[GE_Xcols]#.values

y = panCancer_df['LN_IC50']

# load the dataset
#X, y = make_regression(n_samples=1000, n_features=100, n_informative=10, noise=0.1, random_state=1)
# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
# feature selection
X_train_fs, X_test_fs, fs = select_features(X_train, y_train, X_test)
# what are scores for the features
#for i in range(len(fs.scores_)):
	#print('Feature %d: %f' % (i, fs.scores_[i]))
# plot the scores
pyplot.bar([i for i in range(len(fs.scores_))], fs.scores_)
pyplot.show()

fs.scores_.max()

len(Drug_Xcols+GE_Xcols)

features_scores_df = pd.DataFrame()

features_scores_df['Name'] = Drug_Xcols+GE_Xcols

features_scores_df['Score'] = fs.scores_

features_scores_df

features_scores_df[:2985]

(features_scores_df)[features_scores_df['Score']>350]

# function to convert sequence strings into k-mer words, default size = 6 (hexamer words)
def scoreClass(score):
    if score!=None:
        if (score>=0.0 and score<10.0):
            return 'class1'
        elif score>=10.0 and score<20.0:
            return 'class2'
        elif score>=20.0 and score<30.0:
            return 'class3'
        elif score>=30.0 and score<40.0:
            return 'class4'
        elif score>=40.0 and score<50.0:
            return 'class5'
        elif score>=50.0 and score<60.0:
            return 'class6'
        elif score>=60.0 and score<80.0:
            return 'class7'
        elif score>=80.0 and score<100.0:
            return 'class8'
        elif score>=100.0 and score<150.0:
            return 'class9'
        elif score>=150.0 and score<300.0:
            return 'class10'
        elif score>=300.0 and score<500.0:
            return 'class11'
        elif score>=500.0 and score<1500.0:
            return 'class12'
        else:
            return 'class13'
    else:
        return None

features_scores_df['ScoreClass'] = features_scores_df.apply(lambda x: scoreClass(x['Score']), axis=1)

features_scores_df

features_scores_df.dtypes

(features_scores_df[:2985])['ScoreClass'].value_counts()

(features_scores_df[2985:])['ScoreClass'].value_counts()

features_scores_df.to_csv('/featureScore.csv',index=False)

