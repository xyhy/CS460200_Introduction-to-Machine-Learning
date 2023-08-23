
import os
import numpy as np
import matplotlib.pyplot as plt

def draw(records, filename):

    if records is None:
        print("The records input is None!!!")
    if filename is None:
        print("The filename is None!!!")
    else:
        records = np.asarray(records)
        count = records.ndim

        plt.figure(figsize=(8, 5))
        # plt.ylim((0, 500))

        if count == 1:
            plt.plot(records, linewidth = 1)
        else:
            for idx in range(count):
                plt.plot(records[idx], linewidth = 1)

        if not os.path.exists("records"):
            os.makedirs("records")

        plt.savefig("records/{}.png".format(filename))
