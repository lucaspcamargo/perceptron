import numpy as np

IMG_W = 128
STEPS = 1024
ETTA_INITIAL = 1.0


# teste simples com problema linearmente separavel
# base de dados inicial: iris dataset: https://archive.ics.uci.edu/ml/datasets/iris

# deve ser multiclasse

w = np.zeros(IMG_W*IMG_W)
def etta(i:int):
    return ETTA_INITIAL

def training():

    for i in range(STEPS):

        y[i] = sgn(w[i].transposed() * x[i])

        # adaptacao de pesos:
        w[i+1] = w[i] + etta(i) * (desired[i] - y[i]) * x[i]

        # TODO: embaralhar os dados em épocas
        
        
def classify():
    pass
