import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf

class TransferFunctions:
    def __init__(self):
        pass

    # V-Shaped Transfer Functions
    @staticmethod
    def vstf_01(x):
        """VSTF 01: Uses the error function."""
        return np.abs(erf((np.sqrt(np.pi) / 2) * x))

    @staticmethod
    def vstf_02(x):
        """VSTF 02: Utilizes the hyperbolic tangent function."""
        return np.abs(np.tanh(x))

    @staticmethod
    def vstf_03(x):
        """VSTF 03: Normalized linear output relative to squared terms."""
        return np.abs(x / np.sqrt(1 + np.square(x)))

    @staticmethod
    def vstf_04(x):
        """VSTF 04: Implemented using the arc tangent function."""
        return np.abs((2 / np.pi) * np.arctan((np.pi / 2) * x))

    # S-Shaped Transfer Functions
    @staticmethod
    def sstf_01(x):
        """SSTF 01: Steeper logistic function."""
        return 1 / (1 + np.exp(-2 * x))

    @staticmethod
    def sstf_02(x):
        """SSTF 02: Standard logistic curve."""
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sstf_03(x):
        """SSTF 03: A gentler logistic curve with increased scale."""
        return 1 / (1 + np.exp(-x / 3))

    @staticmethod
    def sstf_04(x):
        """SSTF 04: Another variation of the logistic function."""
        return 1 / (1 + np.exp(-x / 2))

    @staticmethod
    def plot_functions(funcs, title):
        """Plot a list of functions."""
        test_values = np.linspace(-10, 10, num=100)
        for func in funcs:
            outputs = func(test_values)
            plt.plot(test_values, outputs, label=func.__name__)
        plt.title(title)
        plt.xlabel('Input Values')
        plt.ylabel('Output Values')
        plt.legend()
        plt.grid()
        plt.show()

if __name__ == "__main__":
    transfer_funcs = TransferFunctions()

    # Test V-shaped functions
    transfer_funcs.plot_functions(
        [TransferFunctions.vstf_01, TransferFunctions.vstf_02,
         TransferFunctions.vstf_03, TransferFunctions.vstf_04],
        'V-Shaped Transfer Functions'
    )

    # Test S-shaped functions
    transfer_funcs.plot_functions(
        [TransferFunctions.sstf_01, TransferFunctions.sstf_02,
         TransferFunctions.sstf_03, TransferFunctions.sstf_04],
        'S-Shaped Transfer Functions'
    )