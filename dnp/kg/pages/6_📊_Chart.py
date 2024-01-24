import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

def plot_dict_bar(xlabel="xlabel", ylabel="ylabel", legend=[], avg=False, percent=False, text=False, xlog=False, ylog=False, f=0, **dict): #dict={A:[], B:[]}
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
                ax.text(a, b, f'%.{f}f'%(b), ha="center", va="bottom", fontsize=12)
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
              f=0,
              Offline=[298.8, 1117.3], 
              Semi_Online=[26747.1, 295170.4], 
              Online=[120191.6, 1000000], # gcp nlp not available in sandbox
              )

st.caption("nphrase = 10 vs 20 vs 30 vs 40 vs 50 vs 60 vs 70 vs 80 vs 90 vs 100 (Semi-Online, txt, C-1, top-20, jaccard/cosine)")
plot_dict_bar(xlabel="Number of Phrases Extracted", 
              ylabel="Graph Construction Time (ms)", 
              legend=["Local"], 
              avg=False, 
              percent=False, 
              text=True, 
              xlog=False, 
              ylog=True, 
              f=0,
              n10=[46550.4], 
              n20=[71326.1], 
              n30=[77871.4], 
              n40=[52145.8], 
              n50=[82088.5], 
              n60=[84264.5], 
              n70=[65402.7], 
              n80=[104143.2], 
              n90=[112897.6], 
              n100=[140090.0], 
              )

st.divider()

st.title("Graph Size")

st.header("DNP")

st.caption("nphrase = 10 vs 20 vs 30 vs 40 vs 50 vs 60 vs 70 vs 80 vs 90 vs 100 (Semi-Online, txt, C-1, top-20, jaccard/cosine)")
plot_dict_bar(xlabel="Number of Phrases Extracted", 
              ylabel="Number", 
              legend=["# nodes", "# edges"], 
              avg=False, 
              percent=False, 
              text=True, 
              xlog=False, 
              ylog=True, 
              f=0,
              n10=[855, 4928], 
              n20=[1492, 6569], 
              n30=[2029, 8140], 
              n40=[2484, 9521], 
              n50=[2968, 10643], 
              n60=[3389, 11671], 
              n70=[3812, 12666], 
              n80=[4206, 13528], 
              n90=[4593, 14444], 
              n100=[4982, 15265], 
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
              f=0,
              P100=[3099, 7292], 
              P1000=[23081, 276711], 
              P10000=[51193, 1172705],
              )    

st.divider()

st.title("nDCG")

st.header("DNP")

st.caption("nphrase = 20 vs 30 vs 40 vs 50 vs 60 vs 70 vs 80 vs 90 vs 100 (C-1, p=20, jaccard, linear)")
plot_dict_bar(xlabel="Number of Phrases Extracted", 
              ylabel="nDCG", 
            #   legend=["Linear"], 
              avg=False, 
              percent=False, 
              text=True, 
              xlog=False, 
              ylog=False, 
              f=3,
              n20=[0.8426041137752035], 
              n30=[0.9016290795217933], 
              n40=[0.9097269734827902], 
              n50=[0.9295534087685132], 
              n60=[0.9719214072407957], 
              n70=[0.9940449558041299], 
              n80=[0.9975939072599207], 
              n90=[0.9993570847175404], 
              n100=[1.0], 
              )

st.caption("nphrase = 20 vs 30 vs 40 vs 50 vs 60 vs 70 vs 80 vs 90 vs 100 (C-1, p=20, jaccard, exponential)")
plot_dict_bar(xlabel="Number of Phrases Extracted", 
              ylabel="nDCG", 
            #   legend=["Exponential"], 
              avg=False, 
              percent=False, 
              text=True, 
              xlog=False, 
              ylog=False, 
              f=3,
              n20=[0.840209026139336], 
              n30=[0.900006042256782], 
              n40=[0.9081490997008801], 
              n50=[0.9283279320428501], 
              n60=[0.9715995355521987], 
              n70=[0.9939891623894176], 
              n80=[0.9975657202676101], 
              n90=[0.9993502245573821], 
              n100=[1.0], 
              )

st.caption("nphrase = 20 vs 30 vs 40 vs 50 vs 60 vs 70 vs 80 vs 90 vs 100 (C-1, p=20, cosine, linear)")
plot_dict_bar(xlabel="Number of Phrases Extracted", 
              ylabel="nDCG", 
            #   legend=["Linear"], 
              avg=False, 
              percent=False, 
              text=True, 
              xlog=False, 
              ylog=False, 
              f=3,
              n20=[0.8772030685410965], 
              n30=[0.9239751607324898], 
              n40=[0.9443252006288475], 
              n50=[0.9451233392774365], 
              n60=[0.9569083564384423], 
              n70=[0.9889435578574491], 
              n80=[0.9961796569569021], 
              n90=[0.9998489460748377], 
              n100=[1.0], 
              )

st.caption("nphrase = 20 vs 30 vs 40 vs 50 vs 60 vs 70 vs 80 vs 90 vs 100 (C-1, p=20, cosine, exponential)")
plot_dict_bar(xlabel="Number of Phrases Extracted", 
              ylabel="nDCG", 
            #   legend=["Exponential"], 
              avg=False, 
              percent=False, 
              text=True, 
              xlog=False, 
              ylog=False, 
              f=3,
              n20=[0.8733013551208197], 
              n30=[0.9215000529388362], 
              n40=[0.9425099875658931], 
              n50=[0.9433186157074812], 
              n60=[0.9553039668095741], 
              n70=[0.9885486940368893], 
              n80=[0.9961049321363387], 
              n90=[0.9998445599850421], 
              n100=[1.0], 
              )

st.caption("p = 5 vs 10 vs 15 vs 20 (C-1, nphrase=50, cosine, exponential)")
plot_dict_bar(xlabel="Top-k Rank", 
              ylabel="nDCG", 
            #   legend=["n50"], 
              avg=False, 
              percent=False, 
              text=True, 
              xlog=False, 
              ylog=False, 
              f=3,
              p5=[0.9533662650719359],
              p10=[0.9309694198312977],
              p15=[0.9363462634037488],
              p20=[0.9433186157074812], 
              )