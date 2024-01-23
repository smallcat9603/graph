import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

def plot_dict_bar(xlabel="xlabel", ylabel="ylabel", legend=[], avg=False, percent=False, text=False, xlog=False, ylog=False, **dict): #dict={A:[], B:[]}
    fig, ax = plt.subplots()

    firstkey, firstvalue = list(dict.items())[0]
    ncol = len(firstvalue)
    items =[[] for i in range(ncol)]
    for key, value in dict.items():
        for i in range(ncol):
            items[i].append(value[i])
    x = np.arange(len(dict))
    keys = list(dict.keys())
    if avg == True:
        for i in range(ncol):
            items[i].append(np.mean(items[i]))
        x = np.arange(len(dict)+1) 
        keys.append("Avg")
    width = 0.8/ncol
    idx = -(ncol-1)/2
    textlist = []
    for i in range(ncol):
        ax.bar(x+width*idx, items[i], width)
        textlist += list(zip(x+width*idx, items[i]))
        idx += 1
    if text == True:
        for a, b in textlist:
            if percent == True:
                ax.text(a, b, '%.1f%%'%(b*100), ha="center", va="bottom", fontsize=12)
            else:
                ax.text(a, b, '%.0f'%(b), ha="center", va="bottom", fontsize=12)
    ax.set(xticks=x, xticklabels=keys)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if xlog == True:
        ax.set_xscale('log')
    if ylog == True:
        ax.set_yscale('log')
    if len(legend) == ncol:
        ax.legend(legend, loc='upper left', ncol=ncol//2+1)

    ax.spines['left'].set_color('black')
    ax.spines['left'].set_linewidth(1)
    ax.spines['right'].set_color('black')
    ax.spines['right'].set_linewidth(1)
    ax.spines['top'].set_color('black')
    ax.spines['top'].set_linewidth(1)
    ax.spines['bottom'].set_color('black')
    ax.spines['bottom'].set_linewidth(1)
    if percent == True:
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=1))

    st.pyplot(fig) 

st.title("Graph Construction Time")

st.header("DNP")

st.caption("Offline vs Semi-Online vs Online (nphrase=50)")
plot_dict_bar(xlabel="Graph Construction", 
              ylabel="Graph Construction Time (ms)", 
              legend=["Local", "Sandbox (w/o GCP NLP)"], 
              avg=False, 
              percent=False, 
              text=True, 
              xlog=False, 
              ylog=True, 
              Offline=[298.8, 1117.3], 
              Semi_Online=[26747.1, 295170.4], 
              Online=[120191.6, 1000000], # gcp nlp not available in sandbox
              )

st.header("Wiki")

st.caption("P100 vs P1000 vs P10000 (Online, nphrase=50)")
plot_dict_bar(xlabel="Dataset", 
              ylabel="Number", 
              legend=["# nodes", "# edges"], 
              avg=False, 
              percent=False, 
              text=True, 
              xlog=False, 
              ylog=True, 
              P100=[3099, 7292], 
              P1000=[23081, 276711], 
              P10000=[51193, 1172705],
              )    
