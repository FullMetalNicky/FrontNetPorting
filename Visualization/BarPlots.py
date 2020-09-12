import numpy as np
import matplotlib.pyplot as plt


def pixelinfo():
    x_r2 = [0.8445, 0.8716, 0.8140, 0.6943]
    y_r2 = [0.824, 0.763, 0.7857, 0.7199]
    z_r2 = [0.6652, 0.6745, 0.6298, 0.466]
    phi_r2 = [0.4508, 0.4659, 0.3184, 0.1667]

    fig, axs = plt.subplots(2, 2, figsize=(8, 8))
    labels = ['160x96', '80x48', '40x48', '20x12']
    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    axs[0, 0].bar(x, x_r2, width, color='orangered')
    axs[0, 0].set_title('Output variable: x', fontsize=18)
    axs[0, 0].set_ylabel('R2', fontsize=18)
    axs[0, 0].set_xticks(x)
    axs[0, 0].set_xticklabels(labels)
    axs[0, 0].tick_params(axis='x', labelsize=14)

    axs[1, 0].bar(x, z_r2, width, color='orangered')
    axs[1, 0].set_title('Output variable: z', fontsize=18)
    axs[1, 0].set_ylabel('R2', fontsize=18)
    axs[1, 0].set_xticks(x)
    axs[1, 0].set_xticklabels(labels)
    axs[1, 0].tick_params(axis='x', labelsize=14)

    axs[0, 1].bar(x, y_r2, width, color='orangered')
    axs[0, 1].set_title('Output variable: y', fontsize=18)
    axs[0, 1].set_ylabel('R2', fontsize=18)
    axs[0, 1].set_xticks(x)
    axs[0, 1].set_xticklabels(labels)
    axs[0, 1].tick_params(axis='x', labelsize=14)

    axs[1, 1].bar(x, phi_r2, width, color='orangered')
    axs[1, 1].set_title('Output variable: phi', fontsize=18)
    axs[1, 1].set_ylabel('R2', fontsize=18)
    axs[1, 1].set_xticks(x)
    axs[1, 1].set_xticklabels(labels)
    axs[1, 1].tick_params(axis='x', labelsize=14)

    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    plt.suptitle('R2 Score vs. Pixel Information', fontsize=22)
    plt.savefig("barplotpixinfo.png")
    plt.show()


def strategy():
    x_r2 = [0.4977, 0.778, 0.788, 0.6167]
    y_r2 = [0.3127, 0.8083, 0.7604, 0.7098]
    z_r2 = [0.275, 0.5619, 0.5859, 0.4117]
    phi_r2 = [0.1569, 0.2637, 0.2421, 0.1502]

    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    labels = ['Cropping', 'NN', 'Bilinear', 'Foveation']
    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    axs[0, 0].bar(x, x_r2, width, color='orangered')
    axs[0, 0].set_title('Output variable: x', fontsize=18)
    axs[0, 0].set_ylabel('R2', fontsize=18)
    axs[0, 0].set_xticks(x)
    axs[0, 0].set_xticklabels(labels)
    axs[0, 0].tick_params(axis='x', labelsize=14)

    axs[1, 0].bar(x, z_r2, width, color='orangered')
    axs[1, 0].set_title('Output variable: z', fontsize=18)
    axs[1, 0].set_ylabel('R2', fontsize=18)
    axs[1, 0].set_xticks(x)
    axs[1, 0].set_xticklabels(labels)
    axs[1, 0].tick_params(axis='x', labelsize=14)

    axs[0, 1].bar(x, y_r2, width, color='orangered')
    axs[0, 1].set_title('Output variable: y', fontsize=18)
    axs[0, 1].set_ylabel('R2', fontsize=18)
    axs[0, 1].set_xticks(x)
    axs[0, 1].set_xticklabels(labels)
    axs[0, 1].tick_params(axis='x', labelsize=14)

    axs[1, 1].bar(x, phi_r2, width, color='orangered')
    axs[1, 1].set_title('Output variable: phi', fontsize=18)
    axs[1, 1].set_ylabel('R2', fontsize=18)
    axs[1, 1].set_xticks(x)
    axs[1, 1].set_xticklabels(labels)
    axs[1, 1].tick_params(axis='x', labelsize=14)

    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    plt.suptitle('R2 Score vs. Resize Strategy', fontsize=22)
    plt.savefig("barplotstrategy.png")
    plt.show()


def main():
    pixelinfo()
    #strategy()


if __name__ == '__main__':
    main()
