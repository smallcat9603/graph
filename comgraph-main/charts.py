import subprocess
from tkinter.tix import X_REGION
import numpy as np
import matplotlib.pyplot as plt
import statistics
import math
from typing import List, Dict
import seaborn as sns
import sys
import pandas as pd
WIDTH = 33 / 6


def color_normalize(r, g, b):
    return (r/256, g/256, b/256)


RED = color_normalize(219, 68, 55)
GREEN = color_normalize(15, 157, 88)
BLUE = color_normalize(66, 133, 244)
YELLOW = color_normalize(244, 160, 0)
BLACK = color_normalize(59, 59, 59)
PURPLE = color_normalize(67, 2, 151)
ORANGE = color_normalize(243, 112, 33)
COLORS = [BLUE, RED, GREEN, YELLOW, BLACK, PURPLE, ORANGE]

SOLID = "solid"
DASHED = "dashed"
DASHDOT = "dashdot"
DOTTED = "dotted"
LINESTYLES = [SOLID, DASHED, DASHDOT, DOTTED]


def draw_boxplot(
    data,
    title="",
    labels=None,
    x_axis_title=None,
    y_axis_title=None,
    yscale="linear",
    top=None,
    bottom=None,
    rotation=None,
    filename="tmp/tmp.png",
    figsize=(math.sqrt(2) * WIDTH, WIDTH),
    show_graph=False,
    showfliers=False,
    show_median=False,
    print_filename=True,
):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(title)
    for i, d in enumerate(data):
        if labels and show_median:
            print(labels[i], " average: ", sum(d) / len(d),
                  ", median: ", statistics.median(d), sep="")
    ax.boxplot(data, labels=labels, showfliers=showfliers)
    ax.set_xlabel(x_axis_title)
    ax.set_ylabel(y_axis_title)
    ax.set_ylim(top=top, bottom=bottom)
    ax.set_yscale(yscale)
    plt.xticks(rotation=rotation)
    fig.tight_layout()
    if print_filename:
        print(filename)
    plt.savefig(filename, dpi=300)
    plt.close()
    if show_graph:
        try:
            subprocess.run(["open", filename])
        except FileNotFoundError:
            pass


def boxplot_multi(
    data: List[List[List[float]]],
    x_axes: List[str],
    labels: List[str],
    title="",
    label_type="graphs",
    x_axis_title=None,
    y_axis_title=None,
    top=None,
    bottom=None,
    rotation=None,
    filename="tmp/tmp.png",
    show_graph=False,
    showfliers=False,
):
    sns.set()
    sns.set_style('whitegrid')
    sns.set_palette('Set3')

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title(title)
    dfs = []
    for i, label in enumerate(labels):
        dfTa = pd.DataFrame(
            {x_axis: data[i][j] for j, x_axis in enumerate(x_axes)},
            index=[0]
        )
        dfTa_melt = pd.melt(dfTa)
        dfTa_melt[label_type] = label
        dfs.append(dfTa_melt)

    df = pd.concat(dfs, axis=0)
    sns.boxplot(
        x='variable',
        y='value',
        data=df,
        hue=label_type,
        showfliers=showfliers,
        palette='Set3',
        ax=ax
    )
    ax.set_xlabel(x_axis_title)
    ax.set_ylabel(y_axis_title)
    ax.set_ylim(top=top, bottom=bottom)
    plt.xticks(rotation=rotation)
    print(filename)
    plt.savefig(filename, dpi=300)
    if show_graph:
        try:
            subprocess.run(["open", filename])
        except FileNotFoundError:
            pass


def draw_chart(
    x_axes,
    y_axes,
    list_errorbars=None,
    title=None,
    labels=None,
    legend_title=None,
    x_axis_title="",
    y_axis_title="",
    xscale="linear",
    yscale="linear",
    marker='o',
    capthick=1,
    capsize=5,
    lw=1,
    right=None,
    left=None,
    top=None,
    bottom=None,
    colors=None,
    linestyles=None,
    loc="lower right",
    filename="tmp/yeah.png",
    figsize=(math.sqrt(2) * WIDTH * 2.1 / 3, WIDTH * 2.1 / 3),
    show_graph=False,
    transparent=True,
    print_filename=True,
):
    fig, ax = plt.subplots(figsize=figsize)
    for i, y_axis in enumerate(y_axes):
        x_axis = x_axes[i]
        if colors:
            color = colors[i]
        else:
            color = COLORS[i % len(COLORS)]
        if linestyles:
            linestyle = linestyles[i]
        else:
            linestyle = None
        if labels:
            label = labels[i]
        else:
            label = None
        if list_errorbars:
            errorbars = list_errorbars[i]
            plt.errorbar(
                x_axis,
                y_axis,
                yerr=errorbars,
                c=color,
                marker=marker,
                capthick=capthick,
                capsize=capsize,
                lw=lw,
                linestyle=linestyle,
                label=label
            )
        else:
            plt.plot(
                x_axis,
                y_axis,
                c=color,
                marker=marker,
                linestyle=linestyle,
                label=label
            )
    plt.grid()
    ax.set_xlim(left=left, right=right)
    ax.set_ylim(bottom=bottom, top=top)
    ax.set_xlabel(x_axis_title)
    ax.set_ylabel(y_axis_title)
    ax.set_title(label=title)
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    if labels:
        plt.legend(loc=loc, fontsize=8, title=legend_title)
    if print_filename:
        print(filename)
    fig.savefig(filename, dpi=300, format='pdf', transparent=transparent)
    plt.close()
    if show_graph:
        try:
            subprocess.run(["open", filename])
        except FileNotFoundError:
            pass


def draw_scatter(
    x_axes,
    y_axes,
    title=None,
    labels=None,
    x_axis_title="",
    y_axis_title="",
    xscale="linear",
    yscale="linear",
    right=None,
    left=None,
    top=None,
    bottom=None,
    s=100,
    colors=None,
    alpha=1,
    filename="tmp/yeah.png",
    figsize=(math.sqrt(2) * WIDTH, WIDTH),
    show_graph=False,
):
    fig, ax = plt.subplots(figsize=figsize)
    for i, y_axis in enumerate(y_axes):
        if colors:
            color = colors[i]
        else:
            color = COLORS[i % len(COLORS)]
        if labels:
            label = labels[i]
        else:
            label = None
        plt.scatter(
            x_axes[i],
            y_axis,
            color=color,
            alpha=alpha,
            s=s,
            label=label
        )
    plt.grid()
    ax.set_xlim(left=left, right=right)
    ax.set_ylim(bottom=bottom, top=top)
    ax.set_xlabel(x_axis_title)
    ax.set_ylabel(y_axis_title)
    ax.set_title(label=title)
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    if labels:
        plt.legend(loc="lower right", fontsize=8)
    print(filename)
    fig.savefig(filename, dpi=300)
    plt.close()
    if show_graph:
        try:
            subprocess.run(["open", filename])
        except FileNotFoundError:
            pass


def draw_heatmap(
    df: pd.DataFrame,
    title=None,
    x_axis_title: str = None,
    y_axis_title: str = None,
    cbar_title: str = None,
    square=True,
    annot=True,
    filename="tmp/yeah.png",
    figsize=(math.sqrt(2) * WIDTH, WIDTH),
    show_graph=False,
    print_filename=True,
):
    fig, ax = plt.subplots(figsize=figsize)
    ax = sns.heatmap(
        df,
        linewidth=0.5,
        cbar_kws={"label": cbar_title},
        cmap="rocket",
        vmax=1,
        vmin=0,
        square=square,
        annot=annot,
    )

    ax.set_title(title)
    ax.set_xlabel(x_axis_title)
    ax.set_ylabel(y_axis_title)
    fig.tight_layout()
    if print_filename:
        print(filename)
    fig.savefig(filename, dpi=300)
    plt.close()
    if show_graph:
        try:
            subprocess.run(["open", filename])
        except FileNotFoundError:
            pass


def draw_pie_chart(
    data: List[float],
    labels: List[str],
    title=None,
    figsize=(WIDTH, WIDTH),
    print_filename=True,
    colors='pastel',
    autopct='%1.1f%%',
    startangle=90,
    counterclock=False,
    filename="tmp/yeah.png",
):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(title)
    colors = sns.color_palette(colors)
    plt.pie(
        data,
        labels=labels,
        colors=colors,
        autopct=autopct,
        startangle=startangle,
        counterclock=counterclock,
    )
    if print_filename:
        print(filename)
    plt.savefig(filename, dpi=300)
    plt.close()


def draw_band_chart(
    df: pd.DataFrame,
    label_name: str,
    title: str = "",
    figsize=(math.sqrt(2) * WIDTH, WIDTH),
    # y_axis_title
    loc=None,
    horizontal=False,
    print_filename=True,
    show_text=True,
    rot=0,
    filename="tmp/yeah.png",
):
    df.plot(
        x=label_name,
        kind='barh' if horizontal else 'bar',
        stacked=True,
        title=title,
        figsize=figsize,
        rot=rot,
        # mark_right=True,
    )
    plt.legend(loc=loc)
    if show_text:
        df_total = df.iloc[:, 1:len(df.columns)].sum(axis=1)
        df_rel = df[df.columns[1:]].div(df_total, 0)*100
        for n in df_rel:
            for i, (cs, ab, pc) in enumerate(zip(df.iloc[:, 1:].cumsum(1)[n],
                                                 df[n], df_rel[n])):
                plt.text(i, cs - ab / 2, str(np.round(pc, 1)) + '%',
                         va='center', ha='center')
    plt.savefig(filename, dpi=300)
    if print_filename:
        print(filename)
    plt.close()


def draw_posneg_bar_chart(
    dfa,
    dfb,
    label_name,
    figsize=(math.sqrt(2) * WIDTH,  WIDTH),
    title="",
    show_text=True,
    filename="tmp/tmp.png",
    print_filename=True,
):
    fig, ax = plt.subplots(figsize=figsize)
    dfb *= -1
    df = pd.concat([dfa, dfb.iloc[:, 1:]],
                   axis=1)
    # dict to df
    result_df = df.iloc[:, 1:]
    labels = result_df.columns
    plt.bar(result_df.index, result_df[labels[0]], color='#337AE3')
    plt.bar(result_df.index, result_df[labels[1]],
            bottom=result_df[labels[0]], color='#5E96E9')
    plt.bar(result_df.index, result_df[labels[2]], color='#DB4444')
    plt.bar(result_df.index, result_df[labels[3]],
            bottom=result_df[labels[2]], color='#E17979')
    # x and y limits
    # plt.xlim(-0.6, 10.5)
    plt.ylim(-1, 1)
    # remove spines
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    # grid
    ax.set_axisbelow(True)
    ax.yaxis.grid(color='gray', linestyle='dashed', alpha=0.7)
    # x ticks
    xticks_labels = df[label_name]
    plt.xticks(result_df.index, labels=xticks_labels)
    plt.yticks([i / 100 for i in range(-100, 101, 25)], [abs(i / 100)
               for i in range(-100, 101, 25)])
    # title and legend
    legend_label = labels
    plt.legend(
        legend_label,
        ncol=4,
        loc='upper left',
        bbox_to_anchor=(-0.05, -0.05),
        frameon=False
    )
    plt.title(title)

    if show_text:
        dfa_total = dfa.iloc[:, 1:len(dfa.columns)].sum(axis=1)
        dfa_rel = dfa[dfa.columns[1:]].div(dfa_total, 0)*100
        for n in dfa_rel:
            for i, (cs, ab, pc) in enumerate(zip(dfa.iloc[:, 1:].cumsum(1)[n],
                                                 dfa[n], dfa_rel[n])):
                plt.text(i, cs - ab / 2, str(np.round(pc, 1)) + '%',
                         va='center', ha='center')
        dfb_total = dfb.iloc[:, 1:len(dfb.columns)].sum(axis=1)
        dfb_rel = dfb[dfb.columns[1:]].div(dfb_total, 0)*100
        for n in dfb_rel:
            for i, (cs, ab, pc) in enumerate(zip(dfb.iloc[:, 1:].cumsum(1)[n],
                                                 dfb[n], dfb_rel[n])):
                plt.text(i, cs - ab / 2, str(np.round(pc, 1)) + '%',
                         va='center', ha='center')

    plt.savefig(filename, dpi=300)
    if print_filename:
        print(filename)
    plt.show()
    pass


def draw_percentile():
    pass
