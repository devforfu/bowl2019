import matplotlib.pyplot as plt


palette1 = '173f5f 20639b 3caea3 f6d55c ed553b'
palette2 = '264653 2a9d8f e9c46a f4a261 e76f51'
tableau = '4e79a7 f28e2b e15759 76b7b2 59a14f edc948 b07aa1 ff9da7 9c755f bab0ac'


def show_all(df):
    with pd.option_context('display.max_columns', None, 'display.max_rows', None):
        display(df)


def make_cycler(colors):
    from cycler import cycler
    colors_str = ', '.join([f"'{c}'" for c in colors.split()])
    return f"cycler('color', [{colors_str}])"


def hex2rgba(hex_value: str) -> tuple:
    n = len(hex_value)
    if n == 6:
        r, g, b = hex_value[:2], hex_value[2:4], hex_value[4:]
        a = 'ff'
    elif n == 8:
        r, g, b, a = [hex_value[i:i+2] for i in (0, 2, 4, 6)]
    else:
        raise ValueError(f'wrong hex string: {hex_value}')
    rgba = tuple(int(value, 16)/255. for value in (r, g, b, a))
    return rgba


def make_colors(base, size):
    from itertools import islice, cycle
    colors = list(islice(cycle([hex2rgba(x) for x in base.split()]), None, size))
    return colors


def create_axes_if_needed(ax, **fig_kwargs):
    if ax is None:
        f = plt.figure(**fig_kwargs)
        ax = f.add_subplot(111)
    else:
        f = ax.figure
    return f, ax

        
class VisualStyle:
    """Convenience wrapper on top of matplotlib config."""

    def __init__(self, config, default=None):
        if default is None:
            default = plt.rcParams
        self.default = default.copy()
        self.config = config

    def replace(self):
        plt.rcParams = self.config

    def override(self, extra=None):
        plt.rcParams.update(self.config)
        if extra is not None:
            plt.rcParams.update(extra)

    def restore(self):
        plt.rcParams = self.default

    def __enter__(self):
        self.override()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.restore()

    
class NotebookStyle(VisualStyle):
    def __init__(self):
        super().__init__({
            'figure.figsize': (8, 6),
            'figure.titlesize': 20,
            'font.family': 'monospace',
            'font.monospace': 'Liberation Mono',
            'axes.titlesize': 18,
            'axes.labelsize': 16,
            'axes.spines.right': False,
            'axes.spines.top': False,
            'xtick.labelsize': 14,
            'ytick.labelsize': 14,
            'font.size': 14,
            'axes.prop_cycle': make_cycler(tableau)
        })
