# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 22:22:55 2020

@author: 11717
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

data =pd.read_csv("dataset.csv",index_col=0)
data.dropna(axis=1,how='any')
colums_names = data.columns

pos = np.arange(0, 31, 2)
width = 0.5
fig, ax = plt.subplots(1)
fig.set_size_inches(14,10)
plt.grid(axis="y",linestyle='-.')
ax.bar(pos,data['Streptomycin '], width=width,label= 'Streptomycin ')
ax.bar(pos+width, data['Neomycin'], width=width, label="Neomycin")
ax.bar(pos+(2*width), data['Penicilin'], width=width, label="Penicilin")
ax.set_xticks(pos + width / 3)
ax.set_xticklabels(data['Bacteria '],rotation=40, ha="right")
ax.set_ylabel('Concentration of Bacteria')
ax.set_title('Effectiveness of three types of antibiotics')
ax.legend(loc='upper right')
ax.set_yscale('log')
plt.show()
fig.savefig("bar_chart.jpg",dpi=500,bbox_inches = 'tight')

#####################################################################

GramStainingList = data.iloc[:,4].tolist()
print(GramStainingList)
index_positive = [] 
index_nagative = []

for index,norp in enumerate(GramStainingList):
    if norp == 'negative':
        index_nagative.append(index)
    else:
        index_positive.append(index)
##################################################################
data_positive = data.iloc[index_positive,:]
data_nagative = data.iloc[index_nagative,:]

x = data_positive.iloc[:,0]
y1 = data_positive.iloc[:,1:4]


fig,(ax1,ax2)=plt.subplots(1,2)
plt.suptitle("Effectiveness of three types of antibiotics",fontsize=14)
fig.set_figheight(8)
fig.set_figwidth(10)

plt.sca(ax1)
ax1.plot(x,y1,marker='o')
ax1.legend(['Penicilin_Positive','Streptomycin_Positive','Neomycin_Positive'],loc='upper right')
ax1.set_xticklabels(data['Bacteria '],rotation=40, ha="right")
ax1.set_ylabel('Concentration of Bacteria(SymmetricalLogScale)')
ax1.set_yscale('symlog')

#=====================================================================================================
plt.sca(ax2)
x = data_nagative.iloc[:,0]
y1 = data_nagative.iloc[:,1:4]
ax2.plot(x,y1,marker="v")
ax2.legend(['Penicilin_Nagative','Streptomycin_Nagative','Neomycin_Nagative'],loc='upper right')
ax2.set_xticklabels(data['Bacteria '],rotation=40, ha="right")
ax2.get_yaxis().set_visible(False)
#ax2.set_yscale('log')
plt.show()
plt.suptitle("Effectiveness of three types of antibiotics")
fig.savefig("line_chart.jpg",dpi=500,bbox_inches = 'tight')

########################################################################

fig,ax = plt.subplots(figsize=(7,8))
mic_data = np.array(data.iloc[:,1:4])
mic_data = np.around(mic_data,decimals=1)
name_bac = data.iloc[:,0].tolist()
anti_bio_names = data.iloc[:,1:4].columns.tolist()

im = ax.imshow(mic_data,cmap=plt.cm.PiYG)
plt.colorbar(im)
ax.set_ylim(15.5,-0.5)
ax.set_xticks(np.arange(len(anti_bio_names)))
ax.set_yticks(np.arange(len(name_bac)))
ax.set_xticklabels(anti_bio_names)
ax.set_yticklabels(name_bac)

plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")


for i in range(len(name_bac)):
    for j in range(len(anti_bio_names)):
        text = ax.text(j, i, mic_data[i, j],
                       ha="center", va="center", color="w")


ax.set_title("Effectiveness of three types of antibiotics")
fig.tight_layout()
plt.show()
fig.savefig("heatmap.jpg", dpi=500,bbox_inches = 'tight')




