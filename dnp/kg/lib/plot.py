import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

# log: key, value
def plot_log_line(log_name, xlabel="xlabel", ylabel="ylabel", baseline=0.0, baseline_label="baseline", xbegin=0, xend=0):
    data = pd.read_csv(log_name, sep="\s+", names=["key", "value"])
    if baseline > 0.0:
        plt.axhline(y=baseline, xmin=0, xmax=1, linestyle='-', linewidth=1, color='red', label=baseline_label)   
    ax = plt.gca()
    ax.spines['left'].set_color('black')
    ax.spines['left'].set_linewidth(1)
    ax.spines['right'].set_color('black')
    ax.spines['right'].set_linewidth(1)
    ax.spines['top'].set_color('black')
    ax.spines['top'].set_linewidth(1)
    ax.spines['bottom'].set_color('black')
    ax.spines['bottom'].set_linewidth(1)
    if xbegin > 0 and xend > 0:
        plt.xlim((xbegin, xend))
    plt.plot(data["key"], data["value"], '^-', label='tuned', color='black', linewidth=1, markersize=8)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tick_params(top=True, bottom=True, left=True, right=True, direction='in')
    plt.tick_params(which='major', length=4)
    plt.grid(False)
    plt.legend(frameon=False, fontsize=12, loc='upper left', ncol=1)
    # plt.savefig(log_name+'.eps', bbox_inches='tight', transparent=True)
    plt.show()

def plot_dict_bar(xlabel="xlabel", ylabel="ylabel", legend=[], avg=False, percent=False, text=False, xlog=False, ylog=False, **dict): #dict={A:[], B:[]}
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
        plt.bar(x+width*idx, items[i], width)
        textlist += list(zip(x+width*idx, items[i]))
        idx += 1
    if text == True:
        for a, b in textlist:
            if percent == True:
                plt.text(a, b, '%.1f%%'%(b*100), ha="center", va="bottom", fontsize=12)
            else:
                plt.text(a, b, '%.4f'%(b), ha="center", va="bottom", fontsize=12)
    plt.xticks(x, keys)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if xlog == True:
        plt.yscale('log')
    if ylog == True:
        plt.yscale('log')
    if len(legend) == ncol:
        plt.legend(legend, loc='upper left', ncol=ncol//2+1)
    ax = plt.gca()
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
    # plt.savefig('abc.eps', bbox_inches='tight', transparent=True)
    plt.show()        

def get_dict(lower, upper, total, out):
    out_ratio = [(i/total)*100 for i in out]
    in_ratio = [(1-i/total)*100 for i in out]
    dict = {
        lower: np.array(in_ratio),
        upper: np.array(out_ratio),
    }
    return dict

def plot_dict_ratio_bar(dataset, species, dict):
    width = 0.6  # the width of the bars: can also be len(x) sequence

    fig, ax = plt.subplots()
    bottom = np.zeros(len(species))

    for in_out, ratio in dict.items():
        p = ax.bar(species, ratio, width, label=in_out, bottom=bottom)
        bottom += ratio
        ax.bar_label(p, label_type='center', fmt="%.1f%%")

    ax.set_title(dataset)
    ax.legend()
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0*100, decimals=0))

    xlabel = "Number of Servers"
    ylabel = "Ratio"
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    plt.show()
