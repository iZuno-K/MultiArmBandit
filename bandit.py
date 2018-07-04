import numpy as np


class Thompson(object):
    def __init__(self):
        self.a = np.ones(10, np.uint16)
        self.b = np.ones(10, np.uint16)

    def sample(self, size=None):
        sample = np.random.beta(self.a, self.b, size=size)
        return np.argmax(sample)

    def update(self, idx, x):
        if x == 1:
            self.a[idx] += 1
        else:
            self.b[idx] += 1


class UCB(object):
    def __init__(self):
        self.arm_num = 10
        self.one_counts = np.zeros(10, dtype=np.int16)
        self.trial_counts = np.zeros(10, dtype=np.int16)
        self.UCB_scores = np.zeros(10)
        self.t = np.zeros(10, dtype=np.int16)

    def update(self, idx, x):
        self.t += 1
        self.trial_counts[idx] += 1
        if x == 1:
            self.one_counts[idx] += 1
        self.UCB_scores = self.one_counts / self.trial_counts + np.sqrt(np.log(self.t) / 2. /self.trial_counts)

    def sample(self):
        return np.argmax(self.UCB_scores)


class BanditProblem(object):
    def __init__(self):
        self.arm_num = 10
        self.arm_prob = [0.1, 0.05, 0.05, 0.05, 0.02, 0.02, 0.02, 0.01, 0.01, 0.01]

    def pull(self, idx):
        prob = self.arm_prob[idx]
        return np.random.binomial(1, prob)


if __name__ == '__main__':
    # save_path = "/home/isi/karino/master/lecture/ThompsonSamplingData/"
    save_path = "/home/isi/karino/master/lecture/UCBData/"
    for s in range(100):
        np.random.seed(seed=s)
        BP = BanditProblem()
        # alg = Thompson()
        alg = UCB()
        results = [0]
        for i in range(10000):
            arm_idx = alg.sample()
            result = BP.pull(idx=arm_idx)
            alg.update(idx=arm_idx, x=result)
            results.append(result + results[-1])
        np.save(file=save_path + "seed{}".format(s), arr=np.asarray(results))

