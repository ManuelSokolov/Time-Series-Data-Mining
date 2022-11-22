
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

NR_COLUMNS: int = 3
HEIGHT: int = 4
WIDTH_PER_VARIABLE: int = 0.5

#Pallete de cores para os grafs
my_palette = {'yellow': '#ECD474', 'pale orange': '#E9AE4E', 'salmon': '#E2A36B', 'orange': '#F79522', 'dark orange': '#D7725E',
              'pale acqua': '#92C4AF', 'acqua': '#64B29E', 'marine': '#3D9EA9', 'green': '#10A48A', 'olive': '#99C244',
              'pale blue': '#BDDDE0', 'blue2': '#199ED5', 'blue3': '#1DAFE5', 'dark blue': '#0C70B2',
              'pale pink': '#D077AC', 'pink': '#EA4799', 'lavender': '#E09FD5', 'lilac': '#B081B9', 'purple': '#923E97',
              'white': '#FFFFFF', 'light grey': '#D2D3D4', 'grey': '#939598', 'black': '#000000'}
LINE_COLOR = my_palette['dark blue']
FILL_COLOR = my_palette['pale blue']
DOT_COLOR = my_palette['blue3']
ACTIVE_COLORS = [my_palette['dark blue'], my_palette['yellow'], my_palette['pale orange'],
                 my_palette['acqua'], my_palette['pale pink'], my_palette['lavender']]



def set_elements(ax: plt.Axes = None, title: str = '', xlabel: str = '', ylabel: str = '',
                  percentage: bool = False):
    if ax is None:
        ax = plt.gca()
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if percentage:
        ax.set_ylim(0.0, 1.0)
    return ax


def multiple_bar_chart(xvalues: list, yvalues: dict, ax: plt.Axes = None, title: str = '', xlabel: str = '',
                        ylabel: str = '',
                        percentage: bool = False):
    FONT_TEXT = FontProperties(size=6)
    TEXT_MARGIN = 0.05
    ax = set_elements(ax=ax, title=title, xlabel=xlabel, ylabel=ylabel, percentage=percentage)
    ngroups = len(xvalues)
    nseries = len(yvalues)
    pos_group = np.arange(ngroups)
    width = 0.8 / nseries
    pos_center = pos_group + (nseries - 1) * width / 2
    ax.set_xticks(pos_center)
    ax.set_xticklabels(xvalues)
    i = 0
    legend = []
    for metric in yvalues:
        ax.bar(pos_group, yvalues[metric], width=width, edgecolor=LINE_COLOR, color=ACTIVE_COLORS[i])
        values = yvalues[metric]
        legend.append(metric)
        for k in range(len(values)):
            ax.text(pos_group[k], values[k] + TEXT_MARGIN, f'{values[k]:.2f}', ha='center',
                    fontproperties=FONT_TEXT)
        pos_group = pos_group + width
        i += 1
    ax.legend(legend, fontsize='x-small', title_fontsize='small')