import matplotlib.pyplot as plt
from glob import glob
import numpy as np


def draw(save_path, legend):
    files = glob(save_path + "*.npy")
    # cumulative rewards
    data = np.array([np.load(f) for f in files])
    # optimal
    opt = np.array([i * 0.1 for i in range(10001)])
    data = opt - data
    Max = np.max(data, axis=0)
    Min = np.min(data, axis=0)
    Mean = np.mean(data, axis=0)
    x = np.arange(len(Mean))
    plt.plot(x, Mean, label=legend)
    plt.fill_between(x, Min, Max, alpha=0.5)

if __name__ == '__main__':
    save_path1 = "/home/isi/karino/master/lecture/ThompsonSamplingData/"
    save_path2 = "/home/isi/karino/master/lecture/UCBData/"
    plt.style.use("mystyle2")
    plt.figure()
    plt.xscale('log')
    draw(save_path1, "ThompsonSampling")
    draw(save_path2, "UCB")

    plt.xlabel("# of trials: T")
    # plt.ylabel("cumulative rewards")
    plt.ylabel("Regret")

    plt.title("10 arms Bandit")
    plt.legend()
    # plt.show()
    plt.savefig("results.pdf")

