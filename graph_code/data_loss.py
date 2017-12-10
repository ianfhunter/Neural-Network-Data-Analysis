import numpy as np
import matplotlib.pyplot as plt
from graph_code.split import *
from tqdm import *


def data_loss_item(x, label, n, split):
    x = np.sort(x)
    maxX = np.amax(x)
    minX = np.amin(x)


    vals = split_fns[split](x, label, n, maxX, minX)

    data = []

    for idx in range(n):  # Bucket
        data.append([])

    x_counter = 0

    for num in x:  # Number in data
        if x_counter >= n-1:
            # All final values in the last slot
            data[-1].append(num)

        elif num >= vals[x_counter]:
            # Values in the next bucket, increment
            data[x_counter+1].append(num)
            x_counter += 1
        else:
            # Values in this bucket
            data[x_counter].append(num)


    i = 0
    bucket_loss = 0    
    for x in data:  # For each bucket
        i += 1
        npdata = np.array(x)
#        print("\t Items in Bucket", i, ": ", len(x))
        bucket_av = np.sum(npdata) / len(data)
        running_total = 0
        for s in x: # Get the difference between every element and the average
            loss = abs(s - bucket_av)
            running_total += loss
        bucket_loss += running_total
#    print("Individual Bucket Loss", bucket_loss)
            
    return bucket_loss

    

def data_loss(x, label, directory, args):

    orig_x = x
    x = x.flatten()

    SPLIT_LEVEL = args.buckets

    
    for s in SplitType:
        hist_data = []
        for n in tqdm(range(SPLIT_LEVEL)):     
            bucket_loss = data_loss_item(x, label, n+1, s)
            hist_data.append(bucket_loss)

        print("Historical Data: ", hist_data)
        plt.plot(hist_data)

        plt.title("Data Loss when using " + str(n+1) + " buckets based on "+dirs[s])
        plt.ylabel('Amount')
        plt.xlabel('Value')

        plt.savefig(directory+"/"+label.replace("/","__")+"_bucket_loss_"+dirs[s]+"_"+str(n+1)+".svg", format="svg")
        plt.close()

