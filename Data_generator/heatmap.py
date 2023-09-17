from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

#data=pd.read_csv("图卷积层调参结果.csv",encoding='gbk')
data=pd.read_csv("线性层调参结果.csv",encoding='gbk')


#custom_colors=["#d0dfe6","#c3d7df","#baccd9","#b0d5df","#8abcd1","#66a9c9","#619ac3","#2875b6"]
custom_colors=["#f0c2a2","#f5b087","#f29a76","#f9906f","#f29667","#f18f60","#f9723d","#ed6d46"]
#custom_colors=["#f9906f","#f29a76","#f5b087","#f0c2a2","#d0dfe6","#c3d7df","#baccd9","#b0d5df",]
#绘制热度图：
tick_=np.arange(-10,10,4).astype(float)
dict_={'orientation':'vertical',"label":"color  \
scale","drawedges":True,"ticklocation":"right","extend":"min", \
"filled":True,"alpha":0.8,"cmap":"black","ticks":tick_,"spaci,linewidths=0.5ng":'proportional'}
#绘制添加数值和线条的热度图：
cmap = sns.heatmap(data,linewidths=0.8,annot=True,fmt=".2f",yticklabels=[128,64,32,16],cmap=custom_colors)
plt.xlabel("dimensions of GCN layers",size=10)
plt.ylabel("dimensions of hidden layer",size=10,rotation=90)
#plt.title("heatmap",size=10)
#调整色带的标签：
cbar = cmap.collections[0].colorbar
cbar.ax.tick_params(labelsize=10,labelcolor="black")
#cbar.ax.set_ylabel(ylabel="color scale",size=10,color="red",loc="center")

plt.show()