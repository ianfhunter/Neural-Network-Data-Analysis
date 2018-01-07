from graph_code.heatmap import *
from graph_code.data_loss import *
from graph_code.split import *


from enum import Enum
class Chart(Enum):
    Heatmap = 1
    Abs_range = 2
    Weight_Dist = 3
    DataLoss = 4
    Split = 5
    
chart_fns = {
    Chart.Heatmap: heatmap_weights,
    Chart.DataLoss: data_loss,
    Chart.Split: split,
}

directories = {
    Chart.Heatmap: "graphs/heatmap_weights",
    Chart.DataLoss: "graphs/dataloss",
    Chart.Split: "graphs/split",
}

