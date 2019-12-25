import matplotlib.pyplot as plt
from style import tableau, make_colors, create_axes_if_needed


def plot_bars_with_percentage(dataframe, xcol, ycol, 
                              xlabel='Number of Respondents', 
                              ylabel='',
                              orient='horizontal',
                              ax=None, colors=None, **fig_kwargs):
    
    assert orient in ('horizontal', 'vertical'), f'invalid orient value: {orient}'
    f, ax = create_axes_if_needed(ax, **fig_kwargs)
    colors = colors if colors is not None else make_colors(tableau, len(dataframe))

    def generate_bars(df, ax=None):
        plot_fn = dict(
            horizontal=dataframe.plot.barh,
            vertical=dataframe.plot.bar)[orient.lower()]
        ax = dataframe.plot.barh(x=xcol, y=ycol, ax=ax, color=colors)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_axisbelow(True)
        ax.grid(True)
        ax.get_legend().remove()
        return ax
    
    def generate_percentage_annotations(df, ax):
        total = df[ycol].sum()
        for i, count in enumerate(df[ycol]):
            ax.text(
                count + 200, i, f'{count/total:2.2%}', fontsize=12, 
                verticalalignment='center', 
                horizontalalignment='left',
                bbox=dict(facecolor='white', edgecolor='black'))
        return ax
    
    def adjust_limits(ax):
        x_min, x_max = ax.get_xlim()
        x_max *= 1.1
        ax.set_xlim(x_min, x_max)
        return ax
    
    ax = generate_bars(dataframe, ax=ax)
    ax = generate_percentage_annotations(dataframe, ax=ax)
    ax = adjust_limits(ax)
    return ax