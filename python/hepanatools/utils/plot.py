import matplotlib.pyplot as plt
import matplotlib
import sys

# global style settings
matplotlib.rcParams.update({"font.size": 16})


def split_subplots(*args, **kwargs):
    return plt.subplots(
        *args, gridspec_kw={"height_ratios": [2, 1], "hspace": 0}, sharex=True, **kwargs
    )


def savefig(name, **kwargs):
    print(f"Writing {name}")
    sys.stdout.flush()
    plt.savefig(name, **kwargs)
    plt.close()
