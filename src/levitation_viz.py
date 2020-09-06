import numpy as np
from src.config import TotalSteps

#this should hopefully eventually have a more complete levitaation viz, for now just thisS

def DataFill(DataInputMatrix, ColumnIndex):
    Data = np.zeros((TotalSteps+1, 1))
    for i in range(TotalSteps):
        Data[i] = DataInputMatrix[i][ColumnIndex]
    return Data

#Use for "Particles"
def DataFill2(DataInputMatrix, LayerIndex, ColumnIndex):
    Data = np.zeros((TotalSteps+1, 1))
    for i in range(TotalSteps):
        Data[i] = DataInputMatrix[LayerIndex][i][ColumnIndex]
    return Data
