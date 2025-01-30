import matplotlib.pyplot as plt
import numpy as np

def fmt(x):
    s = f"{x:.1f}"
    if s.endswith("0"):
        s = f"{x:.0f}"
        return rf"{s} \%" if plt.rcParams["text.usetex"] else f"{s}"

def fmt_personalized(cslevels, x):
    fmt = {}
    for l, s in zip(cslevels, x):
        fmt[l] = s
    return fmt


def adjustFigAspect(fig,aspect=1):
    '''
    Adjust the subplot parameters so that the figure has the correct
    aspect ratio.
    '''
    xsize,ysize = fig.get_size_inches()
    minsize = min(xsize,ysize)
    xlim = .4*minsize/xsize
    ylim = .4*minsize/ysize
    if aspect < 1:
        xlim *= aspect
    else:
        ylim /= aspect
    fig.subplots_adjust(left=.5-xlim,
                        right=.5+xlim,
                        bottom=.5-ylim,
                        top=.5+ylim)


def manual_labeling(ax, cs, i=1):
    # get limits if they're automatic
    xmin,xmax,ymin,ymax = ax.axis()
    # work with logarithms for loglog scale
    # middle of the figure:
    logmid = (np.log10(xmin)+np.log10(xmax))/2, (np.log10(ymin)+np.log10(ymax))/2

    label_pos = []
    for line in cs.collections:
        for path in line.get_paths():
            logvert = np.log10(path.vertices)
            # find closest point
            logdist = np.linalg.norm(logvert-logmid, ord=2, axis=1)
            min_ind = np.argmin(logdist)
            label_pos.append(10**logvert[min_ind,:])
    return label_pos


def get_xy_coordinates(cs):
    x_arr = []  # x points
    y_arr = []  # y points
    for item in cs.collections:
        for j in item.get_paths():
            v = j.vertices
            x = v[:, 0]
            y = v[:, 1]
            x_arr.append(x)
            y_arr.append(y)
    x_arr = x_arr[0]
    y_arr = y_arr[0]
    return x_arr, y_arr

def replace_negative(v, to_replace=0):
    for i in range(len(v)):
        if v[i] < 0:
            v[i] = to_replace
    return v


def replace_positive(v, to_replace=0):
    for i in range(len(v)):
        if v[i] > 0:
            v[i] = to_replace
    return v
