#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import sys, os
import caffe
import matplotlib.mlab as mlab
from tqdm import *
import argparse
import glob
import shutil

from GraphConf import *

""" 
Usage:

./graph_weights file.prototxt file.caffemodel


"""

SPLIT_LEVEL = 2

def weight_distribution(x, label):
    if not os.path.exists("graphs/weight_dist"):
        os.makedirs("graphs/weight_dist")

    std = np.std(x)
    mean = np.mean(x)

    bins = 1000

    # Hist
    q, bins, patches = plt.hist(x, bins=bins)

    # Best fit line
    y = mlab.normpdf(bins, mean, std) * sum(q * np.diff(bins))
    plt.plot(bins, y, '--')

    plt.title("Weight Distribution for layer "+str(label) )
    plt.ylabel('Amount')
    plt.xlabel('Value')
    plt.savefig("graphs/weight_dist/"+label.replace("/","__")+"_wdist.jpg")
    plt.clf()


def split_on_popularity(x, label, n):

    if not os.path.exists("graphs/pop_splitting"):
        os.makedirs("graphs/pop_splitting")

    x = np.sort(x)
    x = np.array_split(x, n)
    # print("#---------Splitting on "+str(n)+"--------#")
    # for arr in x:
    #     print(arr[0], "to", arr[-1])
    #
    # print("#--------------------------------------------#")
    plt.hist(x, bins=500, lw=0)
    plt.title("Splitting layer "+str(label) +" on " + str(n) + " items based on popularity")
    plt.ylabel('Amount')
    plt.xlabel('Value')
    plt.savefig("graphs/pop_splitting/"+label.replace("/","__")+"_thresh_on_pop_"+str(n)+".svg", format="svg")
    plt.clf()


def plot_dims(x, label):
    if not os.path.exists("graphs/dims"):
        os.makedirs("graphs/dims")

    fig, ax = plt.subplots(nrows=len(x.shape) - 1, ncols=1)

    if len(x.shape) < 3:
        return
    Y = x.shape[2]
    X = x.shape[3]
    if X == 1 and Y == 1:
        return

    print(x.shape)
    quit()

    for xx,row in enumerate(ax):    # In the case 64*3*7*7
        if xx == 0:
            row.plot(x[0,0,:,:])    # 7*7
        elif xx == 1:
            row.plot(x[0,:,:,0])    # 3*7
        elif xx == 2:
            row.plot(x[:,:,0,0])    # 64*3
                                    # Where the y axis is normalized.
#    plt.plot(x)
#    plt.title("Range of layer "+str(label) )
    plt.ylabel('Value')
    plt.xlabel('Axis Index')
    plt.savefig("graphs/dims/"+label.replace("/","__")+"_dims.jpg")
    plt.close()


def plot_dims2(x, label):
    if not os.path.exists("graphs/dims2"):
        os.makedirs("graphs/dims2")

    fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(18,9))

    # DISCARD SECTION
    if len(x.shape) < 3:
        return        
    Y = x.shape[2]
    X = x.shape[3]
    if X == 1 and Y == 1:
        return
    # DISCARD SECTION

    plt.ylabel('Value')
    plt.xlabel('Filter Height Index')
    plt.legend(loc='upper left',  prop={'size': 6})

    
    charts = [
            x[0,0,:,:],     # Fh*Fw
            x[:,:,0,0],     # ic*oC
    ]
    
    for ix, data in enumerate(charts):
        flat_data = data.flatten()

        top_box = ax[0]
        if ix == 0:
            top_box.set_title("Single Fw*Fh Channel Data Scatter")
        else:        
            top_box.set_title("Single iC*oC Filter Data Scatter")
    
        top_box.plot(flat_data, len(flat_data)*[1] , "x")
        
        box2 = ax[1]
        box2.set_title("Histogram & Quant16")
        box2.hist(flat_data, bins=1000, alpha=0.8)
        
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        inset_axes = inset_axes(ax[1],
                        width="30%", # width = 30% of parent_bbox
                        height=1., # height : 1 inch
                        loc=2)
        inset_axes.hist(flat_data, bins=16)
        
        
        box3 = ax[2]
        box3.set_title("All Data")
        new_x = x.reshape((x.shape[0]*x.shape[1], x.shape[2]*x.shape[3]))
        new_x = np.transpose(new_x)
        for ondx,su in enumerate(new_x):
            if ondx >3:
                break
            box3.hist(su, bins=500, alpha=0.6)
        
        
        plt.savefig("graphs/dims2/"+str(ix)+"/"+label.replace("/","__")+"_dims.jpg")
        plt.cla()
    plt.close()
        
    
def range_graph(x, label):
    if not os.path.exists("graphs/range_graphs"):
        os.makedirs("graphs/range_graphs")
    x = np.unique(x)
    plt.plot(x)
    plt.title("Range of layer "+str(label) )
    plt.ylabel('Value')
    plt.xlabel('Sorted Item #')
    plt.savefig("graphs/range_graphs/"+label.replace("/","__")+"_range.jpg")
    plt.clf()

def abs_range_graph(x, label):
    if not os.path.exists("graphs/abs_range_graphs"):
        os.makedirs("graphs/abs_range_graphs")
    x = np.unique(x)
    x = np.absolute(x)
    plt.plot(x)
    plt.title("Range where negative values are flipped of layer "+str(label) )
    plt.ylabel('Value')
    plt.xlabel('Sorted Item #')
    plt.savefig("graphs/abs_range_graphs/"+label.replace("/","__")+"_absrange.jpg")
    plt.clf()

    

# Main Section


parser = argparse.ArgumentParser(description='Generate charts of Neural Network Properties')
parser.add_argument('prototxt', metavar='Prototxt', type=str, nargs='?',
                    help='Description of the network')
parser.add_argument('caffemodel', metavar='Caffemodel', type=str, nargs='?',
                    help='Trained Binary of the network')

parser.add_argument('--layer', metavar='layer', type=str, nargs='?',
                    help='Override single layer selection')
parser.add_argument('--buckets', metavar='buckets', type=int, nargs='?', default=4,
                    help='Choose Number of Buckets')

parser.add_argument('--one-only', dest='one_only', action='store_const',
                    default=False, required=False, const=True,
                    help='only run a single layer of the network (the first)')
parser.add_argument('--preclear', dest='preclear', action='store_const',
                    default=False, required=False, const=True,
                    help='Clean a directory before writing new graphs')

parser.add_argument('--max-only', dest='max_only', action='store_const',
                    default=False, required=False, const=True,
                    help='Only run for the maximal graph')


args = parser.parse_args()

# Argument Checking
if args.prototxt is None or args.caffemodel is None:
    print("Prototxt/Caffemodel Needed")
    quit()

for idx, c in enumerate(Chart):
    print(idx+1, ": ", c)

num = ""
while True:

    num = input("Choose a Chart Type:\n")

    try:
        num = int(num)
    except(ValueError):
        pass

    if type(num) is int and num <= max([e.value for e in Chart]) and num > 0:
        break
    else:
        print("That is an invalid Choice\n")


choice = num    
net = caffe.Net(args.prototxt, args.caffemodel, caffe.TEST)   # Load Network

"""
im = Image.open("/home/ian/Personal/Masters/Datasets/ImageNetValidation/ILSVRC2012_img_val/n01440764/ILSVRC2012_val_00000293.JPEG")
print((net.blobs['data'].data.shape[2:]))
im = im.resize((net.blobs['data'].data.shape[2:]))

im = np.array(im)
im.reshape((1,im.shape))

#net.blobs['data'].reshape(*im.shape)
#im.reshape(*net.blobs['data'])


print(net.blobs['data'].data.shape)
print(im.shape)

net.blobs['data'].data[...] = im


net.forward()

"""

for label in tqdm(net.params):

    # User may have requested an overwrite of the label to test
    if args.layer is not None:
        label = args.layer


    data = net.params[label][0].data
    
    gen_chart = chart_fns[Chart(choice)]
    folder = directories[Chart(choice)]
        
    # Check if we were directed to empty the folder beforehand.
    if args.preclear:        
        if os.path.exists(folder):
            shutil.rmtree(folder)

    # If the folder doesn't exist, Make it
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    gen_chart(data, label, folder, args)
    
    # User may only want one inference
    if args.one_only or args.layer is not None:
        break

# Old Code Below

#for label in tqdm(net.params):
#    data = net.params[label][0].data
#    flattened_data = data.flatten()
#    plot_dims2(data, label)
#    heatmap_weights(data, label)
#    weight_distribution(flattened_data, label)
#    range_graph(flattened_data, label)
#    abs_range_graph(flattened_data, label)
#    for n in tqdm(range(SPLIT_LEVEL)):         # Values in 4bit
#        split_on_popularity(flattened_data, label, n+1)
#        split_on_importance(flattened_data, label, n+1)
#        split_on_regularity(flattened_data, label, n+1)
