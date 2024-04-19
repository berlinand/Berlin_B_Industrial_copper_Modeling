import streamlit as st
import seaborn as sns
import numpy as np
import pandas as pd
import math
from scipy.stats import skew
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score,precision_score,f1_score,recall_score,roc_auc_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

#This function is used to clean dataset by droping or filling null value and removing unwanted columns 
def datacleaning(df):

  df=df.dropna(subset=['status','selling_price','application','country','customer','id','item_date','thickness','delivery date'])
  df = df.copy()
  df['material_ref']=df['material_ref'].fillna(df['material_ref'].mode()[0]) 
  df.loc[df['material_ref'].str.startswith('000'),'material_ref']=np.nan
  df['material_ref']=df['material_ref'].fillna(df['material_ref'].mode()[0])
  df=df[df['status']!='Draft']
  df=df[df['status']!='To be approved']
  df=df[df['status']!='Not lost for AM']
  df=df[df['status']!='Revised']
  df['status'] = df['status'].replace('Wonderful', 'Won')
  df=df.drop(['id','item_date','delivery date'], axis=1)
  df['quantity tons'] = df['quantity tons'].astype(np.float64)
  return df
# This function clean outliers and return the dataframe without outliners with conditions
def clearoutliers(df,cns,cs2,ab):
    for cn in cns:
        q1=df[f'{cn}'].quantile(0.25)
        q3=df[f'{cn}'].quantile(0.75)

        iqr=q3-q1
        lb=q1-(1.5*iqr)
        ub=q3+(1.5*iqr)
        if cs2!=True:
            if ab=='AFTER':
              df=df[(df[f'{cn}']>=lb) & (df[f'{cn}']<=ub) ]
            elif ab=='BEFORE':
              df=df
        elif cs2 ==True:
           df=df.copy()
           df.loc[(df[f'{cn}']<lb,f'{cn}')]=lb
           df.loc[(df[f'{cn}']>ub,f'{cn}')]=ub
     

    return df

#This function adjust the dataset skewness to normal distribution  by log tranformation
def skewnes(af,aa):
   
  for col in aa:
    print(col)
    kw=skew(af[col])
    if kw>-0.5 and kw<0.5:
       print(skew(af[col]))

    else:
      print(np.log(skew(af[col])))
      af[f'{col} log']=np.log(af[col])
  return af

# This function change the catogerical column to numerical value in binary 
def lblencode(af):
  model= LabelEncoder()
  af['status']=model.fit_transform(af['status'])
  return af

# This function change the catogerical column  to numerical value not in binary
def onehotencode(af):
   af=pd.get_dummies(af,columns=['item type','material_ref'],dtype=int)
   return af
#This function display befor and after chart by user selection
def charts(df,aa,ab,chart,af):
   if ab=="BEFORE":
        if chart == 'BOXPLOT':
         for col in aa:
           st.write(f":red[column name]{'- '+col}")
           
           fig=plt.figure()
           sns.boxplot(df[col])
           st.pyplot(fig=fig)
        if chart == 'DISTPLOT':
         for col in aa:
           st.write(f":red[column name]{'- '+col}")
           st.write(f":red[skew]{skew(df[col])}")
           fig=plt.figure()
           sns.distplot(df[col])
           st.pyplot(fig=fig)
        if chart == 'HISTPLOT':
         for col in aa:
           st.write(f":red[column name]{'- '+col}")
           st.write(f":red[skew]{skew(df[col])}")
           fig=plt.figure()
           sns.histplot(df[col])
           st.pyplot(fig=fig)
          
        if chart == 'VIOLINPLOT':
         for col in aa:
           st.write(f":red[ncolumn name]{'- '+col}")
           fig=plt.figure()
           sns.violinplot(df[col])
           st.pyplot(fig=fig)
   if ab=="AFTER":
        if chart == 'BOXPLOT':
         for col in aa:
           st.write(f":red[column name]{'- '+col}")
           
           fig=plt.figure()
           sns.boxplot(af[col])
           st.pyplot(fig=fig)
        if chart == 'DISTPLOT':
         for col in aa:
           st.write(f":red[column name]{'- '+col}")
           kw=skew(af[col])
           if kw>-0.5 and kw<0.5:
             st.write(f":red[skew]{skew(af[col])}")
             fig=plt.figure()
             sns.distplot(af[col])
             st.pyplot(fig=fig)
           else:
             st.write(f":red[skew]{np.log(skew(af[col]))}")
             fig=plt.figure()
             sns.distplot(np.log(af[col]))
             st.pyplot(fig=fig)
        if chart == 'HISTPLOT':
         for col in aa:
           st.write(f":red[column name]{'- '+col}")
           kw=skew(af[col])
           if kw>-0.5 and kw<0.5:
             st.write(f":red[skew]{skew(af[col])}")
             fig=plt.figure()
             sns.histplot(af[col])
             st.pyplot(fig=fig)
           else:
             st.write(f":red[skew]{np.log(skew(af[col]))}")
             fig=plt.figure()
             sns.histplot(np.log(af[col]))
             st.pyplot(fig=fig)
          
        if chart == 'VIOLINPLOT':
         for col in aa:
           st.write(f":red[ncolumn name]{'- '+col}")
           fig=plt.figure()
           sns.violinplot(af[col])
           st.pyplot(fig=fig)
#This function will display heatmap of correlation of the columns
def heatmaps(af):
  afsf=[ 'application',
        'width', 'selling_price',
       'country log', 'customer log', 'product_ref log',
       'quantity tons log', 'thickness log']
  kf=af[afsf]
  co = kf.corr()
  fig=plt.figure()
  sns.heatmap(co, annot=True)
  st.pyplot(fig=fig)
 #This function display metrices value for the different classification models  in train and test data 
def mlclas(models,af,cd,ch3,ch4,ch5):
            if len(ch4)==0 or len(ch4)==1:
              X = af.drop(['status'], axis=1)
            else:
              X=af[ch4]
            y = af['status']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ch5)
            if models=='Logistic Regression':
             if cd!=None:
               model = LogisticRegression(**cd)
             else:
              model = LogisticRegression()

            if models=='K-NN':
             if cd!=None: 
              model = KNeighborsClassifier(**cd)   
             else:          
              model = KNeighborsClassifier(n_neighbors=3)
            if models=='svc':
             if cd!=None:
               model = SVC(**cd)
             else:
              model = SVC()
            if models=='Decision Tree Classifier':
             if cd!=None:
               model = DecisionTreeClassifier(**cd)
             else:
              model = DecisionTreeClassifier()            
            if models =='Random Forest Classifier':
             if cd!=None:
               model = RandomForestClassifier(**cd)
             else:
              model =RandomForestClassifier()                


            if models =='Gradient Boosting Classifier':
             if cd!=None:
               model = GradientBoostingClassifier(**cd)
             else:
              model =GradientBoostingClassifier()  
            model.fit(X_train, y_train)           
            train_predict = model.predict(X_train)
            test_predict = model.predict(X_test)
            st.write(f":violet[Train Accuracy]-{accuracy_score(y_train,train_predict)}")
            st.write(f":violet[Train Precision]-{precision_score(y_train,train_predict)}")
            st.write(f":violet[Train F1] -{f1_score(y_train,train_predict)}")
            st.write(f":violet[Train Recall] -{recall_score(y_train,train_predict)}")
            st.write(f":violet[Train Auc] -{roc_auc_score(y_train,train_predict)}")
            st.write(f":violet[Test Accuracy] -{accuracy_score(y_test,test_predict)}")
            st.write(f":violet[Test Precision]-{precision_score(y_test,test_predict)}")
            st.write(f":violet[Test F1 ]-{f1_score(y_test,test_predict)}")
            st.write(f":violet[Test Recall] -{recall_score(y_test,test_predict)}")
            st.write(f":violet[Test Auc] -{roc_auc_score(y_test,test_predict)}")

 #This function give good parameter value for the different classification models  in train and test data 
def cvse(models,af):
            if len(ch4)==0 or len(ch4)==1:
              X = af.drop(['status'], axis=1)
            else:
              X=af[ch4]
            y = af['status']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            if models=='Logistic Regression':
              model = LogisticRegression()
              params={
               'penalty':['l1', 'l2', 'elasticnet', None],
               'tol':[0.0001,0.0002,0.0003,0.0004,0.0003,0.0002]
               #'max_iter':[100,200,300,400]
               
            } 
            if models=='K-NN':
              model = KNeighborsClassifier()
              params={
              'n_neighbors':[3,4,5,6,7]
            }
            if models=='svc':
              model = SVC()
              params={
                 #'kernel':['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],
                 'gamma':['scale', 'auto'],
                 'tol':[0.0001,0.0002,0.0003,0.0004,0.0003,0.0002]
              }
            if models=='Decision Tree Classifier':
              model = DecisionTreeClassifier()   
              params={
                  'criterion':['gini', 'entropy', 'log_loss'],
                  'max_depth':[2,3,4,5,6,7],
                  'min_samples_split':[2,3,4,5,6,7,8,9]
              }
            if models =='Random Forest Classifier':
              model=RandomForestClassifier()
              params={
                  'criterion':['gini', 'entropy', 'log_loss'],
                  'max_depth':[2,3,4,5,6,7],
                  'min_samples_split':[2,3,4,5,6,7,8,9]
              }                
            if models =='Gradient Boosting Classifier':
              model =GradientBoostingClassifier()    
              params={
                
                 'loss':['log_loss', 'exponential'],
                  'learning_rate':[1.0,1.1,1.2,1.5,2.0,2.3,3.0],
                  'n_estimators':[25,50,100,150,200,250]

              }          
            
            cv=GridSearchCV(model,params)
            cv.fit(X_train, y_train)
            cds=cv.best_params_
            st.write(f":red[Best cv]-{cds}")
      
            return cds
 #This function display metrices value for the different regression models  in train and test data 
def mlreg(af,models,ch4,ch5):
            if len(ch4)==0 or len(ch4)==1:         
               X = af.drop(['selling_price'], axis=1)
            else:
              X=af[ch4]
            y = af['selling_price']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ch5)
            if models=='Linear Regression':
               model=LinearRegression()
            if models=='Lasso':
               model=Lasso()
            if models=='Ridge':
               model=Ridge()        
            if models=='Decision Tree Regression':
               model=DecisionTreeRegressor()
                  
            if models=='Random Forest Regression':
               model=RandomForestRegressor() 
            if models=='Gradient Boosting Regression':
               model=GradientBoostingRegressor()                    
            model.fit(X_train,y_train)
            train_pred=model.predict(X_train)
            test_pred=model.predict(X_test)
            st.write(F":violet[Train MSE]-{mean_squared_error(y_train,train_pred)} ")
            st.write(F":violet[Test MSE]-{mean_squared_error(y_test,test_pred)} ")
            st.write(F":violet[Train RMSE]-{math.sqrt(mean_squared_error(y_train,train_pred))} ")
            st.write(F":violet[Test RMSE]-{math.sqrt(mean_squared_error(y_test,test_pred))} ")
            st.write(F":violet[Train MAE]-{mean_absolute_error(y_train,train_pred)} ")
            st.write(F":violet[Test MAE]-{mean_absolute_error(y_test,test_pred)} ")


#This function to read the data file from local system and convert into dataframe
@st.cache_data          
def readdata():
 df=pd.read_excel(r"C:\Users\berli\Downloads\Copper_Set.xlsx",nrows=2000)
 return df
#This function will return random sample data from dataframe 
def sa_random(df):
  df=df.sample(n=rsa, random_state=1)
  return df




df=readdata()
st.title(":blue[Industrial] :orange[Copper] :blue[Modeling]")
ct1=st.checkbox(label='random sampling')
cs2=False
if ct1 ==True:
  c10,c11=st.columns(2)
  rsa=c10.number_input(label='No of random dataset',min_value=200)
  cs2=c11.checkbox(label='capping')
  df=sa_random(df)

df=datacleaning(df)
aa=list(df.columns)
aa.remove('material_ref')
aa.remove('status')
aa.remove('item type')
aa.sort()


  
co1,co2,co3=st.columns(3) 
ab=co1.selectbox(label="Before or After", options=["BEFORE","AFTER"])
ch1=co3.checkbox(label="HEATMAP")
af=clearoutliers(df,aa,cs2,ab)
af=skewnes(af,aa)
af=lblencode(af)
af=onehotencode(af)
chart=co2.selectbox(label="Select a Chart",options=["BOXPLOT" ,"VIOLINPLOT","DISTPLOT","HISTPLOT"])
charts(df,aa,ab,chart,af)
if ch1==True:
  heatmaps(af)
co4,co5,co6,co7,co8=st.columns(5)
recl=co4.selectbox(label="select a value",options=['Classification','Regression'])
if recl == 'Classification':
  models=co5.selectbox(label='select a model',options=['Logistic Regression','K-NN','svc','Decision Tree Classifier',
                                                       'Random Forest Classifier', 'Gradient Boosting Classifier'])
  ch3=co6.checkbox(label="Best CV")
  ch4=co7.multiselect(label="select columns",options=list(af.columns.values))
  ch5=co8.selectbox(label="test data ",options=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])
cds=None
if recl == 'Classification':
  mlclas(models,af,cds,ch3,ch4,ch5)
  if ch3==True:
    cds1=cvse(models,af,ch4)
    
    mlclas(models,af,cds1,ch3,ch4,ch5)
if recl =='Regression':
    models=co5.selectbox(label='select a model',options=['Linear Regression','Lasso','Ridge','Decision Tree Regression',
                                                          'Random Forest Regression', 'Gradient Boosting Regression'])
    
    ch4=co6.multiselect(label="select columns",options=list(af.columns.values))
    ch5=co7.selectbox(label="test data ",options=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])
    mlreg(af,models,ch4,ch5)

