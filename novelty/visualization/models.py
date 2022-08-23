from matplotlib import pyplot as plt
import torch

def plot_accs(accs):
    accs = torch.stack(accs,dim=0)
    plt.plot(accs, label=list(range(10)))
    plt.ylim(0.4,1)
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend()
    plt.show()