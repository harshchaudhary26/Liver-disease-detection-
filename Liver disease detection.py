import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbn

df = pd.read_csv('indian_liver_patient.csv')
df
df['Gender'].replace({"Female": 1 , "Male": 0},inplace = True)

df

df.isnull().sum()

df.shape

##Balanced or not

df['Dataset'].value_counts()

df['Dataset'].value_counts().plot.bar()

df.corr()

plt.figure(figsize=(20,10))
sbn.heatmap(df.corr(),annot=True , cmap = 'terrain')

df.describe()

missingValue = df.Albumin_and_Globulin_Ratio.mean()
y = df['Dataset']
Feature_selection
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
fit_features = SelectKBest(score_func = f_classif)
x = df.drop(['Dataset'],axis = 1)
x
fit_features.fit(x,y)
score_Col = pd.DataFrame(fit_features.scores_,columns = ['score value'])
score_Col
name_Col = pd.DataFrame(x.columns)
name_Col
top_features = pd.concat([name_Col,score_Col],axis = 1)
top_features
top_features.nlargest(7,'score value')

Feature importance
from sklearn.ensemble import ExtraTreesClassifier
model = ExtraTreesClassifier()
model.fit(x,y)
model.feature_importances_
top = pd.Series(model.feature_importances_,index = x.columns)
top
top.nlargest(3,keep='first').plot()
top.nlargest(10).plot(kind='bar')
top.nlargest(10).plot(kind = 'barh')
top.nlargest(10).plot(kind = 'hist')
top.nlargest(10).plot(kind = 'box')
top.nlargest(10).plot(kind = 'pie')

Scalling data
from sklearn.preprocessing import MinMaxScaler
MinMaxScaler = MinMaxScaler()
columns_to_scale = ['Age','Total_Bilirubin','Direct_Bilirubin','Alkaline_Phosphotase','Alamine_Aminotransferase','Aspartate_Aminotransferase','Total_Protiens','Albumin_and_Globulin_Ratio']
df[columns_to_scale] = MinMaxScaler.fit_transform(df[columns_to_scale])

df[columns_to_scale]

Training Data

import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set(style='whitegrid', color_codes=True)

from sklearn.model_selection import train_test_split, cross_val_score
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=20,stratify= y)

Sampling
from imblearn import under_sampling, over_sampling
from imblearn.over_sampling import SMOTE

resample = SMOTE()
x_train, y_train = resample.fit_resample(x_train, y_train)

pip install -U imbalanced-learn

pip install imblearn

!pip install imblearn

pip install -c glemaitre imbalanced-learn

pip install imblearn==0.0

from imblearn import under_sampling, over_sampling
from imblearn.over_sampling import SMOTE

resample = SMOTE()
x_train, y_train = resample.fit_resample(x_train, y_train)

!pip install scikit-learn==0.18.2

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

from imblearn import under_sampling, over_sampling
from imblearn.over_sampling import SMOTE

resample = SMOTE()
x_train, y_train = resample.fit_resample(x_train, y_train)

x_test

y_train

y_test

Total_values = y_train.value_counts()
print(Total_values)

CNN
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("ggplot")
plt.rcParams['figure.figsize'] = (12, 8)
import seaborn as sns
sns.set(style='whitegrid', color_codes=True)
import warnings
warnings.filterwarnings('ignore')
from sklearn.feature_selection import chi2,f_classif, mutual_info_classif, SelectKBest
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.layers import Conv1D, MaxPool1D
from tensorflow.keras.optimizers import Adam

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().fit(x_train)

x_train = scaler.transform(x_train)
x_train = pd.DataFrame(x_train)

x_test = scaler.transform(x_test)
x_test = pd.DataFrame(x_test)


# Setting Column Names from dataset
x_train.columns = x.columns
x_test.columns = x.columns


x_train = scaler.transform(x_train)
x_train = pd.DataFrame(x_train)

x_test = scaler.transform(x_test)
x_test = pd.DataFrame(x_test)


# Setting Column Names from dataset
x_train.columns = x.columns
x_test.columns = x.columns

x_train.shape

x_test.shape

x_train.ndim

## Converting pandas dataframe to numpy array

x_cnn_train = x_train.to_numpy()
x_cnn_test = x_test.to_numpy()
y_cnn_test = y_test.to_numpy()
y_cnn_train = y_train.to_numpy()
x_train.shape

## X_train is reshaped to 3-dimensional array because we have imported Conv1D which
## expects 3-D input. (455,30,1) means it has 455 rows (training) and 30 columns
## 1 means it is just one-channel data (Not an image).
x_cnn_train = x_cnn_train.reshape(624,10,1)
x_cnn_test = x_cnn_test.reshape(146,10,1)

x_cnn_train.ndim

epochs= 10

model = Sequential()

model.add(Conv1D(filters= 32, kernel_size= 2, activation = 'relu', input_shape = (10,1)))
model.add(BatchNormalization())
#model.add(MaxPool1D())  --- Pooling is bad for these kinds of data
model.add(Dropout(0.2))

model.add(Conv1D(filters= 64, kernel_size= 2 , activation= 'relu'))
model.add(BatchNormalization())
#model.add(MaxPool1D(1,1))
model.add(Dropout(0.5))

model.add(Conv1D(filters=128, kernel_size= 2 , activation= 'relu'))
model.add(BatchNormalization())
#model.add(MaxPool1D(1,1))
model.add(Dropout(0.5))

model.add(Conv1D(filters= 256, kernel_size= 2 , activation= 'relu'))
model.add(BatchNormalization())
#model.add(MaxPool1D(1,1))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(256,activation = 'relu'))
model.add(Dropout(0.5))

model.add(Dense(1,activation = 'sigmoid'))

model.summary()

DETECTION USING MLP
model.compile(optimizer = Adam(lr = 0.00005), loss = 'binary_crossentropy', metrics =['accuracy'])

history = model.fit(x_cnn_train, y_cnn_train, epochs= epochs, validation_data=(x_cnn_test, y_cnn_test), verbose=1)

from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(max_iter = 500, activation = 'relu',random_state=None)
mlp

mlp.fit(x_train, y_train)

prediction = mlp.predict(x_test)

prediction

mlp.score(x_test, y_test)

from sklearn.metrics import confusion_matrix,classification_report
print(confusion_matrix(y_test,prediction))

print(classification_report(y_test,prediction))


from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from matplotlib import pyplot
# generate 2 class dataset
x, y = make_classification(n_samples=1000, n_classes=2, random_state=1)
# split into train/test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=2)
# fit a model
model = LogisticRegression(solver='lbfgs')
model.fit(x_train, y_train)
# predict probabilities
lr_probs = model.predict_proba(x_test)
# keep probabilities for the positive outcome only
lr_probs = lr_probs[:, 1]
# predict class values
yhat = model.predict(x_test)
lr_precision, lr_recall, _ = precision_recall_curve(y_test, lr_probs)
lr_f1, lr_auc = f1_score(y_test, yhat), auc(lr_recall, lr_precision)
# summarize scores
print('Logistic: f1=%.3f auc=%.3f' % (lr_f1, lr_auc))
# plot the precision-recall curves
no_skill = len(y_test[y_test==1]) / len(y_test)
pyplot.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
pyplot.plot(lr_recall, lr_precision, marker='.', label='Logistic')
# axis labels
pyplot.xlabel('Recall')
pyplot.ylabel('Precision')
# show the legend
pyplot.legend()
# show the plot
pyplot.show()


from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from matplotlib import pyplot
# generate 2 class dataset
x, y = make_classification(n_samples=1000, n_classes=2, random_state=1)
# split into train/test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=2)
# fit a model
model = LogisticRegression(solver='lbfgs')
model.fit(x_train, y_train)
# predict probabilities
lr_probs = model.predict_proba(x_test)
# keep probabilities for the positive outcome only
lr_probs = lr_probs[:, 1]
# predict class values
yhat = model.predict(x_test)
lr_precision, lr_recall, _ = precision_recall_curve(y_test, lr_probs)
lr_f1, lr_auc = f1_score(y_test, yhat), auc(lr_recall, lr_precision)
# summarize scores
print('Logistic: f1=%.3f auc=%.3f' % (lr_f1, lr_auc))
# plot the precision-recall curves
no_skill = len(y_test[y_test==1]) / len(y_test)
pyplot.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
pyplot.plot(lr_recall, lr_precision, marker='.', label='Logistic')
# axis labels
pyplot.xlabel('Recall')
pyplot.ylabel('Precision')
# show the legend
pyplot.legend()
# show the plot
pyplot.show()

