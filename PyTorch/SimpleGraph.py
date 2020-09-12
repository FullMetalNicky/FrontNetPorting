import numpy as np
import matplotlib.pyplot as plt




def main():
    x_r2 = [0.77, 0.79, 0.84]
    y_r2 = [0.78, 0.79, 0.82]
    z_r2 = [0.6, 0.59, 0.66]
    phi_r2 = [0.3, 0.39, 0.45]
    axis = range(3)

    # plt.plot(axis, x_r2, color='black')
    # plt.plot(axis, y_r2, color='blue')
    # plt.plot(axis, z_r2, color='red')
    plt.plot(axis, phi_r2, color='green', label="phi")
    plt.xticks([0, 1, 2],
               ["80x48", "108x60", "160x96"])
    plt.xlim(-0.5, 2.5)
    plt. title('R2 Score vs. the input size')
    plt.xlabel("Input size")
    plt.ylabel("R2 Score of Phi")

    #plt.legend()
    plt.savefig("phivsinputsize.png")
    plt.show()




if __name__ == '__main__':
    main()