import numpy as np
import matplotlib.pyplot as plt

class LearnNN:
    """Neural Network class for deterministic learning with RBFNN."""
    def __init__(self):
        """Initialize RBFNN with 5D state space and grid parameters."""
        dime = 5  # Number of state dimensions
        minmax = np.array([[19, -19], [19, -19], [154.964, -2.642], 
                           [185.147, 55.125], [294.607, 210.952]])
        length = 12  # Grid size per dimension
        aa = np.linspace(minmax[:, 0], minmax[:, 1], num=length, endpoint=True, dtype=np.float32)
        fcent = aa[:, 0]
        for i in range(1, dime):
            cent1 = np.repeat(aa[:, i], length ** i)
            cent2 = np.tile(fcent, length ** 1)
            fcent = np.vstack((cent1, cent2))
        self.fcent = fcent[::-1, :]
        self.eta = aa[1, :] - aa[0, :]

    def state2NNS(self, x):
        """Convert state to neural network output using Gaussian RBF."""
        error = -(self.fcent.T - x)
        S = np.exp(-1 * np.sum((error / self.eta) ** 2, axis=1))
        return S

def dataplot(w_all, xh_all, e_all, posxall, posyall, poszall):
    """Plot dynamic comparisons between estimated and actual states."""
    w_all = np.array(w_all)
    xh_all = np.array(xh_all)
    e_all = np.array(e_all)

    plt.figure(1)
    plt.title('Dynamic comparison for x position', fontsize=25)
    plt.plot(xh_all[7:, 2], label='State driven from learning process')
    plt.plot(posxall[7:], label='Actual system output')
    plt.plot(e_all[7:, 2], label='Corresponding error')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=5, fontsize=20)
    plt.savefig("3rd_state_comparison.png")

    plt.figure(2)
    plt.title('Dynamic comparison for z position', fontsize=25)
    plt.plot(xh_all[7:, 3], label='State driven from learning process')
    plt.plot(poszall[7:], label='Actual system output')
    plt.plot(e_all[7:, 3], label='Corresponding error')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=5, fontsize=20)
    plt.savefig("2nd_state_comparison.png")

    plt.figure(3)
    plt.title('Dynamic comparison for y position', fontsize=25)
    plt.plot(xh_all[7:, 4], label='State driven from learning process')
    plt.plot(posyall[7:], label='Actual system output')
    plt.plot(e_all[7:, 4], label='Corresponding error')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=5, fontsize=20)
    plt.savefig("4th_state_comparison.png")

    plt.figure(4)
    plt.title('Neural Network Weights', fontsize=25)
    plt.plot(w_all[:, 0, :])
    plt.xlabel('Steps')
    plt.show()

def main():
    """Train RBFNN using deterministic learning and evaluate results."""
    # Load data
    u1all = np.load("u1all_Org.npy")
    u2all = np.load("u2all_Org.npy")
    posxall = np.load("xpos_Org.npy")
    poszall = np.load("zpos_Org.npy")
    posyall = np.load("ypos_Org.npy")

    # Parameters
    num_state = 5
    dim = 12
    nndim = dim ** num_state
    ts = 1
    p = 5
    P = np.zeros((num_state, num_state), int)
    np.fill_diagonal(P, p)
    a = 0.5
    alpha = 1.5
    lam = 5

    # Initialize neural network and variables
    s = LearnNN()
    it = 0
    xh_all = []
    w_all = []
    e_all = []
    s0 = np.zeros(nndim)
    xh0 = np.zeros(num_state)
    xh1 = np.zeros(num_state)
    w = np.zeros((num_state, nndim))

    # Training loop
    while True:
        x0 = np.array((u1all[it - 1], u2all[it - 1], posxall[it - 1], poszall[it - 1], posyall[it - 1]))
        s0 = s.state2NNS(x0)
        x1 = np.array((u1all[it], u2all[it], posxall[it], poszall[it], posyall[it]))
        s1 = s.state2NNS(x1)
        e0 = xh0 - x0
        e1 = xh1 - x1
        alphap = alpha * P
        ae = a * e0
        ed = e1 - ae
        es = np.outer(ed, s0)
        num = np.matmul(alphap, es)
        den = 1 + lam * np.dot(s0.T, s0)
        div = num / den
        wk = w - div
        rbfnn = np.dot(wk, s1)
        trbfnn = ts * rbfnn
        axhx = a * e1
        xhk = x1 + axhx + trbfnn

        xh_all.append(xhk)
        w_all.append(w)
        e_all.append(e1)

        w = np.copy(wk)
        xh0 = np.copy(xh1)
        xh1 = np.copy(xhk)
        it += 1
        if it > (len(u1all) - 1):
            break

    dataplot(w_all, xh_all, e_all, posxall, posyall, poszall)

if __name__ == "__main__":
    main()