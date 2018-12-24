from sklearn.metrics import roc_auc_score
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib


def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Arguments:
        data       : A 2D numpy array of shape (N,M)
        row_labels : A list or array of length N with the labels
                     for the rows
        col_labels : A list or array of length M with the labels
                     for the columns
    Optional arguments:
        ax         : A matplotlib.axes.Axes instance to which the heatmap
                     is plotted. If not provided, use current axes or
                     create a new one.
        cbar_kw    : A dictionary with arguments to
                     :meth:`matplotlib.Figure.colorbar`.
        cbarlabel  : The label for the colorbar
    All other arguments are directly passed on to the imshow call.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=["black", "white"],
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Arguments:
        im         : The AxesImage to be labeled.
    Optional arguments:
        data       : Data used to annotate. If None, the image's data is used.
        valfmt     : The format of the annotations inside the heatmap.
                     This should either use the string format method, e.g.
                     "$ {x:.2f}", or be a :class:`matplotlib.ticker.Formatter`.
        textcolors : A list or array of two color specifications. The first is
                     used for values below a threshold, the second for those
                     above.
        threshold  : Value in data units according to which the colors from
                     textcolors are applied. If None (the default) uses the
                     middle of the colormap as separation.

    Further arguments are passed on to the created text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[im.norm(data[i, j]) > threshold])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


def displayDetails(conf, loss, Loss, acc, num_layer, filename):
    # Summarize history of average halting scores
    plt.title('Average Confidence Scores vs Epoch')
    legend = ['h'+str(i) for i in xrange(num_layer)]
    for y_axis in conf:
        plt.plot(y_axis)
    plt.ylabel('avg confidence scores')
    plt.xlabel('epoch')
    plt.legend(legend, loc='upper right')
    plt.savefig(filename+'/confidence.png')
    plt.clf()

    # Summarize history of average halting scores
    plt.title('Early Exit Losses (ln) vs Epoch')
    legend = ['l'+str(i) for i in xrange(num_layer+1)]
    for y_axis in loss:
        plt.plot(y_axis)
    plt.ylabel('Losses')
    plt.xlabel('epoch')
    plt.legend(legend, loc='upper right')
    plt.savefig(filename+'/losses.png')
    plt.clf()
    
    with open(filename+'/losses.csv', 'w') as f:
        for l in xrange(num_layer+1):
            f.write('        l%d'%l)
        for l in xrange(num_layer+1):
            f.write('        h%d'%l)
        for l in xrange(num_layer+1):
            f.write('        L%d'%l)
        f.write('\n')        

        epoch = len(loss[0])
        for e in xrange(epoch):
            for l in xrange(num_layer+1):
                f.write('%10.4f'%loss[l][e])
            for l in xrange(num_layer+1):
                f.write('%10.4f'%conf[l][e])
            for l in xrange(num_layer+1):
                f.write('%10.4f'%Loss[l][e])
            f.write('\n')


    # Summarize history of overall test accuracy
    plt.title('Overall Test Accuracy')
    plt.plot(acc)
    plt.ylabel('test accuracy')
    plt.xlabel('epoch')
    plt.savefig(filename+'/test_acc.png')
    plt.clf()


def displaySurface(X, Y, test_acc, test_cost, acc_over_cost, filename):
    fig, ax = plt.subplots()
    ax.set_xlabel('Loss penalty constant, c')
    ax.set_ylabel('Exit threshold, T')
    im, cbar = heatmap(test_acc, Y, X, ax=ax, cmap="YlGn", cbarlabel="Test accuracy, A")
    texts = annotate_heatmap(im, valfmt="{x:.3f}")
    fig.tight_layout()
    plt.savefig(filename+"/heatmap_acc.png")
    plt.clf()

    fig, ax = plt.subplots()
    ax.set_xlabel('Loss penalty constant, c')
    ax.set_ylabel('Exit threshold, T')
    im, cbar = heatmap(test_cost, Y, X, ax=ax, cmap="YlGn", cbarlabel="Test cost, C")
    texts = annotate_heatmap(im, valfmt="{x:.3f}")
    fig.tight_layout()
    plt.savefig(filename+"/heatmap_cost.png")
    plt.clf()

    fig, ax = plt.subplots()
    ax.set_xlabel('Loss penalty constant, c')
    ax.set_ylabel('Exit threshold, T')
    im, cbar = heatmap(acc_over_cost, Y, X, ax=ax, cmap="YlGn", cbarlabel="Test accuracy/cost, A/C")
    texts = annotate_heatmap(im, valfmt="{x:.3f}")
    fig.tight_layout()
    plt.savefig(filename+"/heatmap_aoverc.png")
    plt.clf()

    X, Y = np.meshgrid(X, Y)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    # Plot the surface
    surf = ax.plot_surface(X, Y, test_acc, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    ax.set_zlim(-1.01, 1.01)
    ax.set_xlabel('Loss penalty constant, c')
    ax.set_ylabel('Exit threshold, T')
    ax.set_zlabel('Test accuracy, A')
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.title('Test Accuracy Surface')
    plt.savefig(filename+'/surface_acc.png')
    plt.clf()

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    # Plot the surface
    surf = ax.plot_surface(X, Y, test_cost, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    ax.set_zlim(-1.01, 1.01)
    ax.set_xlabel('Loss penalty constant, c')
    ax.set_ylabel('Exit threshold, T')
    ax.set_zlabel('Test cost, C')
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.title('Test Cost Surface')
    plt.savefig(filename+'/surface_cost.png')
    plt.clf()

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    # Plot the surface
    surf = ax.plot_surface(X, Y, acc_over_cost, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    ax.set_zlim(-1.01, 1.01)
    ax.set_xlabel('Loss penalty constant, c')
    ax.set_ylabel('Exit threshold, T')
    ax.set_zlabel('Test accuracy over cost, A/C')
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.title('Test Accuracy Over Cost Surface')
    plt.savefig(filename+'/surface_aoverc.png')
    plt.clf()


def display(history, num_layer, filename):
    # Configuration for layer names
    outs = ['final']
    confs = []
    for i in xrange(num_layer):
        outs.append('out'+str(i))
        confs.append('conf'+str(i))

    # Summarize history for gate accuracy
    plt.title('Model Accuracy')
    y_axis = [s+'_acc' for s in outs] + ['val_'+s+'_acc' for s in outs]
    legend = ['train_'+s for s in outs] + ['test_'+s for s in outs]
    for y in y_axis:
        if y in history.history:
            plt.plot(history.history[y])
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(legend, loc='upper right')
    plt.savefig(filename+'/accuracy.png')
    plt.clf()

    # Summarize history for gate and total losses
    plt.title('Model Loss')
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    y_axis = [s+'_loss' for s in outs] + ['val_'+s+'_loss' for s in outs]
    legend = ['train','test'] + ['train_'+s for s in outs] + ['test_'+s for s in outs]
    for y in y_axis:
        if y in history.history:
            plt.plot(history.history[y])
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(legend, loc='upper right')
    plt.savefig(filename+'/loss.png')
    plt.clf()

