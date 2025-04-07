import numpy as np
import matplotlib.pyplot as plt

class LearnNN:
    """Neural Network class for state estimation using RBFNN."""
    def __init__(self):
        """Initialize RBFNN with 5D state space and grid parameters."""
        dime = 5  # Number of state dimensions
        minmax = np.array([[19, -19], [19, -19], [104.743, -47.663], 
                           [100.893, -41.957], [267.404, 184.525]])
        length = 16  # Grid size per dimension
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

def main():
    """Perform state estimation and fault detection with error analysis."""
    # Initialize neural network and parameters
    s = LearnNN()
    num_state = 5
    dim = 16
    ts = 1
    b = -0.25
    B = np.zeros((num_state, num_state), float)
    np.fill_diagonal(B, b)
    w = np.load("w_motor3.npy")  # Pre-trained weights

    # Load fault data
    u1all = np.load("u1all_Fault3_motor4.npy")
    u2all = np.load("u2all_Fault3_motor4.npy")
    posxall = np.load("xpos_Fault3_motor4.npy")
    poszall = np.load("zpos_Fault3_motor4.npy")
    posyall = np.load("ypos_Fault3_motor4.npy")

    # State estimation
    xbar = np.zeros((num_state), float)
    xbar_all = []
    ebar_all = []
    for i in range(posxall.shape[0]):
        x = np.array((u1all[i], u2all[i], posxall[i], poszall[i], posyall[i]))
        s1 = s.state2NNS(x)
        ws = np.dot(w, s1)
        ebar = xbar - x
        be = np.matmul(B, ebar)
        xbar1 = xbar + ts * be + ts * ws
        xbar_all.append(xbar1)
        ebar_all.append(ebar)
        xbar = np.copy(xbar1)

    xbar_all = np.array(xbar_all)
    ebar_all = np.array(ebar_all)

    # Error analysis with moving average
    ebar_all = np.array(ebar_all[20:, :]).T  # Exclude first 20 steps
    y_error = np.abs(ebar_all)
    weight = 120
    y_error = np.hstack((np.zeros((ebar_all.shape[0], weight-1)), y_error))
    norm_error = np.zeros((ebar_all.shape[1], ebar_all.shape[0]))
    for i in range(ebar_all.shape[0]):
        norm_error[:, i] = np.convolve(y_error[i, :], np.ones(weight), 'valid') / weight

    # Plot results
    plt.figure(9)
    plt.title('Moving Average vs Error for x position')
    plt.plot(ebar_all[2, :], label='Error')
    plt.plot(norm_error.T[2, :], label='Moving Average')
    plt.legend()

    plt.figure(10)
    plt.title('Moving Average vs Error for z position')
    plt.plot(ebar_all[3, :], label='Error')
    plt.plot(norm_error.T[3, :], label='Moving Average')
    plt.legend()

    plt.figure(11)
    plt.title('Moving Average vs Error for y position')
    plt.plot(ebar_all[4, :], label='Error')
    plt.plot(norm_error.T[4, :], label='Moving Average')
    plt.legend()

    plt.show()

if __name__ == "__main__":
    main()