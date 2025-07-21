from numpy import *

def run():
    points = genfromtxt("data.csv", delimiter=",")
    x = points[:, 0]
    y = points[:, 1]

    plt.scatter(x, y)
    plt.show()


if __name__ == "__main__":
    exit(run())
