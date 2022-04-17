import sys
import os
import re
import numpy as np
import matplotlib.pyplot as plt

def readData(filename):
    with open(filename) as f:
        data = f.read()

    scores = [eval(score) for score in data.split('\n')[:-1]]

    return scores

def plotSummaryData(data):
    x = np.unique(data[:,0]).astype(np.int16)
    training_y = data[data[:,1] == 'training'][:,2:].astype(np.float32)
    test_y = data[data[:,1] == 'test'][:,2:].astype(np.float32)
    labels = ['naive count', 'better count', 'naive difference']

    fig, ax = plt.subplots()
    ax.plot(x, training_y[:,1], 'o-b', label=f'train: {labels[1]}')
    ax.plot(x, test_y[:,1], 'o--r', label=f'test: {labels[1]}')
    ax2 = ax.twinx()
    ax2.plot(x, training_y[:,2], 'o-g', label=f'train: {labels[2]}')
    ax2.plot(x, test_y[:,2], 'o--m', label=f'test: {labels[2]}')

    ax.set_xticks(x)

    ax.set_ylabel('count difference')
    ax2.set_ylabel('pixel difference (%)')
    ax.set_xlabel('epoch')

    fig.legend()
    fig.set_tight_layout(True)
    plt.show()
    plt.close()

def summaryTable():
    match = re.compile('^epoch([0-9]*)_results_(test|training)$')
    datafiles = np.array([(f, *match.match(os.path.basename(f)).groups()) for f in sys.argv[1:]])
    datafiles = np.concatenate((datafiles[(datafiles == 'training').any(axis=1)],
                                datafiles[(datafiles == 'test').any(axis=1)]))

    data = []
    table = []
    table.append(f'epoch |  dataset | naive count | better count | naive difference')
    table.append( '------|----------|-------------|--------------|-----------------')
    for datafile in datafiles:
        filename, epoch, dataset = datafile
        if os.path.exists(filename):
            scores = readData(filename)
            scores = np.array([list(s.values()) for s in scores])[:,2:].astype(np.int16)
            if dataset == 'training':
                num_pixels = 200*200
            elif dataset == 'test':
                num_pixels = 500*500

            nc = scores[:,0].mean()
            bc = scores[:,1].mean()
            nd = scores[:,2].mean() / num_pixels * 100

            data.append((epoch, dataset, nc, bc, nd))
            table.append(f' {epoch:^4s} | {dataset:^8s} | {nc:11.0f} | {bc:12.0f} | {nd:16.2f}')
        else:
            print(f'file not found: {filename}')

    return np.array(data), table

def plotPerSample(scores, epoch, dataset):
    # Plot each sample
    x = np.array([score['sample_num'] for score in scores])
    y = np.array([[score['naive_count'],
                   score['better_count'],
                   score['naive_difference']]
                  for score in scores])
    labels = ['naive count','better count','pixel difference']

    plot(x, y, labels, epoch, dataset, "sample")

def plotPerImage(scores, epoch, dataset):
    # Plot average metrics per image
    scores_arr = np.array([list(s.values()) for s in scores])
    image_data = {image: np.delete(
                            scores_arr[(scores_arr == image).any(axis=1)], 1, axis=1
                      ).astype(np.int16) for image in np.unique(scores_arr[:,1])}
    image_data = np.stack([np.hstack((image_data[image][0,0],
                                      image_data[image][:,1:].mean(axis=0)))
                            for image in image_data.keys()]).astype(int)
    image_data = image_data[image_data[:,0].argsort()]

    x = image_data[:,0]
    if type == 'training':
        x //= 10
    y = image_data[:,1:]
    labels = ['naive count','better count','pixel difference']

    plot(x, y, labels, epoch, dataset, "image")

def plot(x, y, labels, epoch, dataset, type):
    fig, ax = plt.subplots()
    ax.plot(x, y[:,0], label=labels[0], marker=',')
    ax.plot(x, y[:,1], label=labels[1], marker=',')

    if dataset == 'training':
        num_pixels = 200*200
    elif dataset == 'test':
        num_pixels = 500*500
    ax2 = ax.twinx()
    ax2.plot(x, y[:,2] / num_pixels * 100, label=labels[2], marker=',', color='red')

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

            scores = readData(filename)

            plotPerSample(scores, epoch, dataset)
            plotPerImage(scores, epoch, dataset)
        else:
            print(f'file not found: {filename}')

def main():
    summaryData, table = summaryTable()
    print('\n'.join(table))
    plotSummaryData(summaryData)

    plotEpochResults()

if __name__ == '__main__':
    main()
