import pandas as pd
import numpy as np
import sklearn.metrics
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def PlotGTVsEstimation(x_c, y_c, z_c, phi_c, x_py, y_py, z_py, phi_py):
    fig = plt.figure(666, figsize=(16, 6))



    gs = gridspec.GridSpec(1, 3)
    ax = plt.subplot(gs[0, 0])
    ax.set_title('Output variable: x', fontsize=18)
    ax.set_xlabel('PyTorch model prediction', fontsize=18)
    ax.set_ylabel('Quantized model prediction', fontsize=18)
    #ax.set_xmargin(0.2)

    plt.scatter(x_py, x_c, color='green', marker='o')
    plt.plot(x_py, x_py, color='black')


    ax = plt.subplot(gs[0, 1])
    ax.set_title('Output variable: y', fontsize=18)
    ax.set_xlabel('PyTorch model prediction', fontsize=18)
    ax.set_ylabel('Quantized model prediction', fontsize=18)
    #ax.set_xmargin(0.2)

    plt.scatter(y_py, y_c, color='blue', marker='*')
    plt.plot(y_py, y_py, color='black')


    # ax = plt.subplot(gs[1, 0])
    # ax.set_title('z')
    # ax.set_xmargin(0.2)
    #
    # plt.scatter(z_py, z_c, color='r', marker='^')
    # plt.plot(z_py, z_py, color='black', linestyle='--')
    #
    # plt.legend()

    ax = plt.subplot(gs[0,2])
    ax.set_title('Output variable: phi', fontsize=18)
    ax.set_xlabel('PyTorch model prediction', fontsize=18)
    ax.set_ylabel('Quantized model prediction', fontsize=18)
    #ax.set_xmargin(0.2)

    plt.scatter(phi_py, phi_c, color='orangered', marker='D')
    plt.plot(phi_py, phi_py, color='black')


    #plt.subplots_adjust(hspace=0.3)
    plt.suptitle('Full Precision PyTorch vs Nemo/Dory', fontsize=22)
    gs.tight_layout(fig, rect=[0, 0.03, 1, 0.95])
    plt.savefig('PyvsNemoDory.png')

def PlotGTVsError(x_err, y_err, z_err, phi_err, x_py, y_py, z_py, phi_py):
    plt.figure(666, figsize=(16, 20))



    gs = gridspec.GridSpec(3, 1)
    ax = plt.subplot(gs[0, 0])
    ax.set_title('x')
    ax.set_xmargin(0.2)
    plt.scatter(x_py, x_err, color='green', marker='o')
    plt.legend()

    ax = plt.subplot(gs[1, 0])
    ax.set_title('y')
    ax.set_xmargin(0.2)
    plt.scatter(y_py, y_err, color='blue', marker='*')
    plt.legend()

    # ax = plt.subplot(gs[1, 0])
    # ax.set_title('z')
    # ax.set_xmargin(0.2)
    #
    # plt.scatter(z_py, z_c, color='r', marker='^')
    # plt.plot(z_py, z_py, color='black', linestyle='--')
    #
    # plt.legend()

    ax = plt.subplot(gs[2,0])
    ax.set_title('phi')
    ax.set_xmargin(0.2)
    plt.scatter(phi_py, phi_err, color='m', marker='D')
    plt.legend()

    plt.subplots_adjust(hspace=0.3)
    plt.suptitle('Full Precision PyTorch vs Residual Error')
    plt.savefig('ErrorNemoDory.png')

def main():


    df = pd.read_csv('results.csv')
    df.head()
    x_c = df['x_pr_c'].values
    y_c = df['y_pr_c'].values
    z_c = df['z_pr_c'].values
    phi_c = df['phi_pr_c'].values
    x_py = df['x_pr_py'].values
    y_py = df['y_pr_py'].values
    z_py = df['z_pr_py'].values
    phi_py = df['phi_pr_py'].values

    MAE = [np.mean(abs(x_c - x_py)), np.mean(abs(y_c - y_py)), np.mean(abs(z_c - z_py)), np.mean(abs(phi_c - phi_py))]
    print("MAE: {}".format(MAE))
    R2_score = [sklearn.metrics.r2_score(x_c, x_py), sklearn.metrics.r2_score(y_c, y_py),
                sklearn.metrics.r2_score(z_c, z_py), sklearn.metrics.r2_score(phi_c, phi_py)]
    print("R2_score: {}".format(R2_score))
    PlotGTVsEstimation(x_c, y_c, z_c, phi_c, x_py, y_py, z_py, phi_py)
  #  PlotGTVsError(x_c - x_py, y_c-y_py, z_c - z_py, phi_c - phi_py, x_py, y_py, z_py, phi_py)


if __name__ == '__main__':
    main()