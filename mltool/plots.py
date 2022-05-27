"""
perceptron and multilayer perceptron implementation
Lucas Pires Camargo, 2022
Tópicos Especiais em Sistemas Eletrônicos IV - Aprendizado de Máquina
Programa de Pós-Graduação em Engenharia de Sistemas Eletrônicos – PPGESE
"""


# deps
import numpy as np
import pickle
from matplotlib import pyplot as plt



##################################################################################################
# DATA COLLECTION

# ... todo graphs and stuff here

class TrainingGraph:

    def __init__(self):
        """Params are the variables of the training"""
        self.configs = {}

    def add_datapoints(self, params_id:str, epoch: int, **kwargs):
        coll = self.configs.get(params_id, {})
        for k,v in kwargs.items():
            lst = coll.get(k, [])
            lst.append(v)
            coll[k] = lst
        self.configs[params_id] = coll

    def get_series(self, params_id:str, datapoint):
        return self.configs[params_id][datapoint]


def show_plots(tg: TrainingGraph, saveloc:str = None):
    fig, axis = plt.subplots(2,1)#(3,1)
    plt.subplots_adjust(hspace = 0.5)
    axis[0].set_title("Average cost", fontsize=10)
    axis[1].set_title("Average validation cost", fontsize=10)
    #axis[2].set_title("Misclassification count", fontsize=10)
    maxes = [0.0,0.0,0.0]
    for cfgname, vals in tg.configs.items():
        dp_avg_cost = vals['cost']
        dp_v_avg_cost = vals['v_cost']
        dp_misclassify_count = vals['sum_misses']
        axis[0].plot(list(range(len(dp_avg_cost))), dp_avg_cost, label=cfgname)
        maxes[0] = max(maxes[0], np.max(dp_avg_cost))
        axis[1].plot(list(range(len(dp_v_avg_cost))), dp_v_avg_cost, label=cfgname)
        maxes[1] = max(maxes[1], np.max(dp_v_avg_cost))
        #axis[2].plot(list(range(len(dp_misclassify_count))), dp_misclassify_count, label=cfgname)
        #maxes[2] = max(maxes[2], np.max(dp_misclassify_count))
    axis[0].legend(fontsize='x-small')
    for i in range(2):#(3):
        axis[i].set_ylim(0, maxes[i])
    if saveloc:
        plt.savefig(saveloc)
        with open(saveloc+'.bin', 'wb') as f:
            pickle.dump(tg.configs, f)
    else:
        plt.show()
    
