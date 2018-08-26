import matplotlib.pyplot as plt
from matplotlib import cm

def plot_matrices_1(sequence, filename, step):

    ncols = 1
    nrows = 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 20, nrows * 4))

    plt.rcParams["axes.grid"] = False

    axes.imshow(sequence, interpolation='nearest', cmap=cm.Purples, aspect='auto')
    axes.set_title("Synergy latens step {}".format(step))

    fig.savefig(filename)  # save the figure to file
    plt.close(fig)

