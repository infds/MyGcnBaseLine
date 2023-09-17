import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import numpy as np
import plotly.io as pio


x_kafka=[75,125,200,300,400,450]
y_kafka=[0.37,0.40,0.29,8.52,13.9,13.95]

x_record=[200,400,600,800,1000]
y_raft=[0.34,0.46,0.50,2.9,10.01,11.43]
#y_addrecord=[0.347,0.371,0.359,0.351,0.363]
#y_getrecord=[0.411,0.439,0.432,0.468,0.422]

y_addrecord=[0.336,0.366,0.384,0.362,0.341]
y_getrecord=[0.399,0.421,0.457,0.483,0.431]
"""
plt.plot(x_kafka,y_kafka,label='kafka',color='blue')
plt.plot(x_kafka,y_raft,label='raft',color='orange')
"""

plt.plot(x_record,y_addrecord,label='add_record',color='blue')
plt.plot(x_record,y_getrecord,label='get_record',color='orange')

plt.xlabel('num of requests')
plt.ylabel('Average Latency (second)')

for i in range(len(x_record)):
    """
     plt.text(x_kafka[i],y_kafka[i],y_kafka[i],ha='center',va='center',color='blue')
    plt.text(x_kafka[i], y_raft[i], y_raft[i], ha='center', va='center', color='orange')
    """
    plt.text(x_record[i],y_addrecord[i],y_addrecord[i],ha='center',va='top',color='blue')
    plt.text(x_record[i], y_getrecord[i], y_getrecord[i], ha='center', va='bottom', color='orange')



plt.legend()
plt.savefig('figure.png')
