import sys
import os
import re
import numpy as np
import matplotlib.pyplot as plt

def readData(filename):
    with open(filename) as f:
        data = f.read()

    data = np.array([list(x.values()) for x in [eval(line) for line in data.split('\n')[:-1]]])

    return data

    x = data[:,0].astype(np.int16)
    counts = data[:,2:4].astype(np.int16)
    scores = data[:,4:].astype(np.float16)

    return x, counts, scores

def plotSummaryData(data):
    x = np.unique(data[:,0]).astype(np.int16)
    training_y = data[data[:,1] == 'training'][:,2:].astype(np.float32)
    test_y = data[data[:,1] == 'test'][:,2:].astype(np.float32)
    labels = ['naive count', 'count difference', 'pixel difference']

    fig, ax = plt.subplots(ncols=2)
    ax[0].plot(x, training_y[:,1], 'o-b', label=f'train')
    ax[0].plot(x, test_y[:,1], 'o-r', label=f'test')
    ax[1].plot(x, training_y[:,2], 'o-b', label=f'train')
    ax[1].plot(x, test_y[:,2], 'o-r', label=f'test')

    ax[0].set_xticks(x)
    ax[1].set_xticks(x)

    ax[0].set_ylabel('count difference')
    ax[1].set_ylabel('pixel difference (%)')
    ax[1].yaxis.tick_right()
    ax[1].yaxis.set_label_position('right')
    ax[0].set_xlabel('epoch')
    ax[1].set_xlabel('epoch')

    ax[0].legend()
    ax[1].legend()
    fig.set_size_inches((7,4))
    fig.set_tight_layout(True)
    plt.savefig(f'epoch_summary_to_epoch{x.max():02d}.png')
    plt.close()

def summaryTable():
    match = re.compile('^epoch([0-9]*)_results_(test|training)$')
    datafiles = np.array([(f, *match.match(os.path.basename(f)).groups()) for f in sys.argv[1:]])
    datafiles = np.concatenate((datafiles[(datafiles == 'training').any(axis=1)],
                                datafiles[(datafiles == 'test').any(axis=1)]))

    output = []
    table = []
    table.append(f'epoch |  dataset | naive count | better count | naive difference')
    table.append( '------|----------|-------------|--------------|-----------------')
    for datafile in datafiles:
        filename, epoch, dataset = datafile
        if os.path.exists(filename):
            data = readData(filename)
            x = data[:,0].astype(np.int16)
            counts = data[:,2:4].astype(np.int16)
            scores = data[:,4:].astype(np.float16)

            nc = scores[:,0].mean()
            bc = scores[:,1].mean()
            nd = scores[:,2].mean()

            output.append((epoch, dataset, nc, bc, nd))
            table.append(f' {epoch:^4s} | {dataset:^8s} | {nc:11.2f} | {bc:12.2f} | {nd:16.2f}')
        else:
            print(f'file not found: {filename}')

    return np.array(output), table

def plotPerSample(data, epoch, dataset):
    # Plot each sample
    x = data[:,0].astype(np.int16)
    y = data[:,4:].astype(np.float32)
    labels = ['naive count','count difference','pixel difference']

    plot(x, y, labels, epoch, dataset, "sample")

def plotPerImage(data, epoch, dataset):
    # Plot average metrics per image
    scores_arr = data
    image_data = {image: np.delete(
                            scores_arr[(scores_arr == image).any(axis=1)], 1, axis=1
                      ).astype(np.float16) for image in np.unique(scores_arr[:,1])}
    image_data = np.stack([np.hstack((image_data[image][0,0],
                                      image_data[image][:,3:].mean(axis=0)))
                            for image in image_data.keys()]).astype(np.float16)
    image_data = image_data[image_data[:,0].argsort()]

    x = image_data[:,0].astype(np.int16)
    if type == 'training':
        x //= 10
    y = image_data[:,1:]
    labels = ['naive count','count difference','pixel difference']

    plot(x, y, labels, epoch, dataset, "image")

def plot(x, y, labels, epoch, dataset, type):
    fig, ax = plt.subplots()
    ax.plot(x, y[:,1], label=labels[1], marker=',')

    ax2 = ax.twinx()
    ax2.plot(x, y[:,2], label=labels[2], marker=',', color='red')

    ax.set_ylabel('count difference')
    ax2.set_ylabel('pixel difference (%)')
    ax.set_xlabel('iteration')

    if dataset == 'test':
        ax.set_xticks(np.arange(0, x.max() + 1, 2))

    fig.legend()
    ax.set_title(f'Epoch {int(epoch)} from {dataset} set')

    fig.set_tight_layout(True)
    fig.savefig(f'epoch{epoch}_by-{type}_{dataset}.png')
    plt.close()

def plotEpochResults():
    for filename in sys.argv[1:]:
        match = re.match('^epoch([0-9]*)_results_(test|training)$', os.path.basename(filename))
        if match and os.path.exists(filename):
            epoch = match.group(1)
            dataset = match.group(2)

            data = readData(filename)

            plotPerSample(data, epoch, dataset)
            plotPerImage(data, epoch, dataset)
        else:
            print(f'file not found: {filename}')

def main():
    summaryData, table = summaryTable()
    print('\n'.join(table))
    plotSummaryData(summaryData)

    plotEpochResults()

if __name__ == '__main__':
    main()
