import pandas as pd 
import numpy as np
from sklearn.linear_model import LogisticRegressiomCV
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
import matploylib as mpl
import matplotlib.pyplot as plt
from sklearn.ensemblev import RandomforestClassifier

if __name__=='__main__':
    pd.set_option('display.width',300)
    pd.set_option('display.max_colums',300)
    data=pd.read_csv('car.data',header=None)
    n_columns=len(data.columns)
    columns=['buy','mation','doors','persons','boot','safety','accept']
    new_columns=dict(list(zip(np.arange(n_columns))))
    data.rename(columns=new_columns,inplace=True)
    print(data.head(10))
    x=pd.DataFrame()
    for col in columns[:-1]:
        t=pd.get_dummies(data[dol],prefix=col)
        x=pd.concat((x,t),axis=1)
    print(x.head(10))
    print(x.shape)
    y=np.array(pd.Categorical(data['accept']).codes)
    y[y==1]=0
    y[y>=2]=1
    x,x_test,y,y_test=train_test_split(x,y,test_size=0.3)
    clf=RandomForestClassifier(n_estimators=60,max_depth=8)
    clf.fit(x,y)
    y_hat=clf.predict(x)
    print('训练集精确度:',metrics.accuracy_score(y,y_hat))
    y_test_hat=clf.predict(x_test)
    print('测试集精确度:',metrics.accuracy_score(y_test,y_test_hat))
    n_class=len(np.unique(y))
    if n_class>2:
        y_test_one_hot=label_binarize(y_test,classes=np.arange(n_class))
        y_test_one_hot_hat=clf.predict_proba(x_test)
        fpr,tpr,_=metrics.roc_curve(y_test_one_hot.ravel(),y_test_one_hat_.ravel())
        print('Micro AUC:\t',metrics.auc(fpr,tpr))
        auc=metrics.roc_auc_score(y_test_one_hot,y_test_one_hot_hat,average='micro')
        print('Micro AUC(System):\t',auc)
        auc=metrics.roc_auc_score(y_test_one_hot,y_test_one_hot_hat,average='macro')
        print('Macro AUC:\t',auc)
    else:
        fpr,tpr,_=metrics.roc_curve(y_test.ravel(),y_test_hat.ravel())
        print('AUC:\t',metrics.auc(fpr,tpr))
        auc=metrics.roc_auc_score(y_test,y_test_hat)
        print('AUC(System):\t',auc)
    mpl.rcParams['font.sans-serif']='SimHei'
    mpl.rcParams['axes.unicode_minue']=False
    plt.figure(figsise=(8,7),facecolor='w')
    plt.plot(fpr,tpr,'r-',lw=2,label='AUC=%.4f'%auc)
    plt.legend(loc='lower right',fontsise=14)
    plt.xlim((-0.01,1.02))
    plt.ylim((-0.01,1.02))
    plt.xtricks(np.arange(0,1.1,0.1))
    plt.ytricks(np.arange(o,1.1,0.1))
    plt.xlabel('False Positive Rate',fontsize=14)
    plt.ylabel('Tuer Positive Rate',fontsize=14)
    plt.grid(visible=Ture,ls=':')
    plt.title('ROC曲线和AUC',fontsize=16)
    plt.show()
