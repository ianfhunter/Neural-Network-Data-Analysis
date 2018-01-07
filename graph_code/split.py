import os
import numpy as np
import matplotlib.pyplot as plt


def popular_split(x, label, n, maxX, minX):
    raise NotImplementedError

def regular_split(x, label, n, maxX, minX):
    seperation = (maxX + abs(minX)) / (n)  # Should this be plus

    vals = []
    for a in range(n-1):
        vals.append(minX+seperation)
        seperation += seperation

    return vals


def importance_split(x, label, n, maxX, minX):
    vals = []

    for a in range(n-1):
        half = (maxX + minX) / 2
        vals.append(half)
        minX = half

    return vals

def minimizeloss_split(x, label, n, maxX, minX):
    print(x.shape)
    vals = []
    quit()


# Split Enums
from enum import Enum
class SplitType(Enum):
    Regular = 1         # Regular Intervals
    Importance = 2      # Prioritize Extreme Positives
    MinimizeLoss = 3   # Prioritize minimal loss figure
    

split_fns = {
    SplitType.Regular: regular_split,
    SplitType.Importance: importance_split,
    SplitType.MinimizeLoss: minimizeloss_split,
}

dirs = {
    SplitType.Regular: "regular",
    SplitType.Importance: "importance",
    SplitType.MinimizeLoss: "minimizeLoss",
}

def split(x, label, directory, args):

    orig_x = x
    x = x.flatten()

    for s in SplitType:
        if not os.path.exists(directory + "/"+dirs[s]):
            os.makedirs(directory + "/"+dirs[s])

    SPLIT_LEVEL = args.buckets

    # Shared Code
    x = np.sort(x)
    maxX = np.amax(x)
    minX = np.amin(x)    

    for n in range(SPLIT_LEVEL):
        m = n+1
        for s in SplitType:
        
            # Get all Split Values        
            vals = split_fns[s](x, label, m, maxX, minX)
    
            # Assign Data
            data = []
            #inc
            for idx in range(m):  # Bucket
                data.append([])

            x_counter = 0
            for num in x:                           # Number in data
            #inc
                if x_counter >= m-1:                # All final values in the last slot
                    data[-1].append(num)
                elif num >= vals[x_counter]:        # Values in the next bucket, increment
                    data[x_counter+1].append(num)
                    x_counter += 1
                else:                               # Values in this bucket
                    data[x_counter].append(num)


            plt.hist(data, bins=500, lw=0)
            plt.title("Splitting layer "+str(label) +" on " + str(m) + " items.")
            plt.ylabel('Amount')
            plt.xlabel('Value')
            plt.savefig(directory + "/"+dirs[s]+"/"+label.replace("/","__")+"_thresh_on_imp_"+str(m)+".svg", format="svg")
            plt.clf()

