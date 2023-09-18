import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import torch
import argparse
import seaborn as sns
from abc import ABC,abstractmethod
from sklearn.metrics import roc_curve,auc
from  sklearn.model_selection import train_test_split
from typing import Union

def Cuda2Cpu(feature,label):
    feature=feature.detach().cpu().numpy()
    label=label.detach().cpu().numpy()
    return feature,label

#设计修饰器用于打印函数中出现的列表的内容
def print_list_content(func):
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        for arg in args:
            if isinstance(arg, list):
                print("the color you have choosed is:")
                for item in arg:
                    print(item)
        return result
    return wrapper

class ResultDrawer(ABC):
    def __init__(self,file_path,task_type,use_cuda,dpi=300,fig_size=(10,10)):
        self.__file_path=file_path
        self.color=list()
        if not isinstance(fig_size,tuple):
            raise ValueError("the figsize must be a tuple")
        self.figsize=fig_size
        self.features=None
        self.label=None
        if task_type not in ("train_mask","vaild_mask","test_mask","all"):
            raise ValueError("your input task type must be one of 'train_mask', 'test_mask', or 'valid_mask'")
        self.task_type=task_type
        self.use_cuda=use_cuda
        self.dpi=dpi
    def read_file(self):
        data_raw=np.load(self.__file_path)
        if self.task_type != "all":
            feature = data_raw['x'][data_raw[self.task_type]]
            labels = data_raw['label'].reshape(-1, 1)[data_raw[self.task_type]]
        else:
            feature=data_raw['x']
            labels=data_raw['label']

        if self.use_cuda:
            feature,labels=Cuda2Cpu(feature,labels)
        self.features=feature
        self.label=labels

    @print_list_content
    def setColorList(self,colorList):
        for color in colorList:
            self.color.append(color)
        return self.color
    @abstractmethod
    def drawer(self):
        pass
    @abstractmethod
    def process(self):
        pass

    def plotDrawing(self):
        self.drawer()
        plt.show()



class TsneDrawer(ResultDrawer):
    def __init__(self, file_path, task_type, use_cuda,color_list,dpi,fig_size=(10, 10)):
        super().__init__(file_path, task_type,use_cuda,dpi,fig_size)
        self.setColorList(color_list)
        self.process()

    def drawer(self):
        tsne = TSNE(n_components=2, random_state=42)
        embeded = tsne.fit_transform(self.features)
        plt.figure(figsize=(self.figsize[0],self.figsize[1]),dpi=self.dpi)
        colors = [self.color[0] if label == 0 else self.color[1] for label in self.label]
        plt.scatter(embeded[:, 0], embeded[:, 1], c=colors, s=10, alpha=0.7)
        plt.title('t-SNE Visualization of Node Features')
        plt.xlabel('t-SNE Dimension')

    def process(self):
        self.read_file()
        self.plotDrawing()


class KdeDrawer(ResultDrawer):
    def __init__(self,file_path, task_type, use_cuda,color_list,dpi,alpha,fig_size=(10, 10)):
        super().__init__(file_path, task_type,use_cuda,dpi,fig_size)
        self.alpha=alpha
        self.process()


    def drawer(self):
        tsne = TSNE(n_components=2, random_state=42)
        embeded = tsne.fit_transform(self.features)
        df = pd.DataFrame(embeded, columns=[f'feature_{i}' for i in range(2)])
        df['label'] = self.label
        plt.figure(figsize=(self.figsize[0],self.figsize[1]),dpi=self.dpi)
        sns.displot(data=df, x='feature_0', hue='label', kde=True, alpha=0.3, edgecolor='White', multiple='layer')
        plt.show()

    def process(self):
        self.read_file()
        self.plotDrawing()



"""
 传入的estimator参数为该类型
 estimator_list = [
        ('logistic_regression', {'C': 0.1, 'solver': 'liblinear'}),
        ('random_forest', {'n_estimators': 100, 'max_depth': 10}),
        ('svm', {'kernel': 'rbf', 'C': 1.0})
    ]
"""
class RocDrawer(ResultDrawer):
    def __init__(self,file_path,task_type, use_cuda,color_list,dpi,result_path,estimator_list,fig_size=(10, 10)):
        super().__init__(file_path, task_type,use_cuda,dpi,fig_size)
        self.rocfile_path=file_path
        self.__result_path=result_path
        self.estimators=[]
        for model_type, param_package in estimator_list:
            estimator = Estimator(model_type, param_package)
            estimator.create_estimator()
            self.estimators.append(estimator)
        self.setColorList(color_list)
        self.process()

    def read_result(self):
        result=np.load(self.__result_path)
        self.__result=result

    def read_data(self):
        data=np.load(self.rocfile_path)
        x = data['x'][:7000]
        y = data['label'][:7000]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=2)
        return x_train,x_test,y_train,y_test


    def process(self):
        self.read_file()
        self.read_result()
        self.plotDrawing()

    def drawer(self):
        self.DL_drawer()
        self.ML_drawer()

    def DL_drawer(self):
        y_pred_DL = self.__result["y_pred"]
        y_test_DL = self.__result["label"]
        fpr_DL, tpr_DL, thresholds_DL = roc_curve(y_test_DL, y_pred_DL[:, 1])
        roc_auc_DL = auc(fpr_DL, tpr_DL)
        plt.plot(fpr_DL, tpr_DL, self.color[0], label=u'Our Model AUC = %0.3f' % roc_auc_DL)

    def ML_drawer(self):
        x_train, x_test, y_train, y_test=self.read_data()
        i=1
        for estimator in self.estimators:
            y_score=estimator.process(x_train,y_train,x_test)[:,1]
            fpr,tpr,_=roc_curve(y_test,y_score)
            roc_auc=auc(fpr,tpr)
            plt.plot(fpr, tpr, self.color[i], label=u'{} AUC = %0.3f'.format(estimator.model_type) % roc_auc)
            i=i+1
    def plotDrawing(self):
        self.drawer()
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([-0.1, 1.1])
        plt.ylim([-0.1, 1.1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.grid(linestyle='-.')
        plt.grid(True)
        plt.show()


class Estimator():
    def __init__(self,model_type,params_package):
        self.model_type=model_type
        self.params_package=params_package
        self.model=None

    def create_estimator(self):
        if self.model_type == 'logistic_regression':
            model = LogisticRegression(**self.params_package)
        elif self.model_type == 'random_forest':
            model = RandomForestClassifier(**self.params_package)
        elif self.model_type == 'svm':
            model = SVC(**self.params_package)
        elif self.model_type=='adaboost':
            model=AdaBoostClassifier(**self.params_package)
        else:
            raise ValueError("Invalid model type")

        self.model=model

    def fit(self,x,y):
        model=self.model
        model.fit(x,y)
        return model

    def process(self,train_x,train_y,test_x):
        model=self.fit(train_x,train_y)
        prob_y=model.predict_proba(test_x)
        return prob_y


class HeatMapDrawer():




if __name__=='__main__':
    estimator_list = [
        ('random_forest', {'max_depth':8,'max_samples':128}),
        ('logistic_regression', {'C': 0.1, 'solver': 'liblinear'}),

    ]
    color_list=["#403990","#CF3D3E","#FBDD85","#80A6E2"]
    RocDrawer(file_path="data.npz",task_type="all",use_cuda=False,color_list=color_list,dpi=300,result_path="Data\dl_result.npz",estimator_list=estimator_list)





