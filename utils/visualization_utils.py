import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import jax.numpy as jnp


class PendulumVis:

    @staticmethod
    def init_plot():
        PLOT_TIKZ = True
        matplotlib.rcParams["font.family"] = "Times New Roman"
        matplotlib.rcParams["figure.figsize"] = [15, 10]
        matplotlib.rcParams["legend.fontsize"] = 16
        matplotlib.rcParams["axes.titlesize"] = 22
        matplotlib.rcParams["figure.titlesize"] = 22
        matplotlib.rcParams["axes.labelsize"] = 22

    @staticmethod
    def plot_trajectory(x1, x2, true_x1, true_x2, u, T, save_path=None):
        f, a = plt.subplots(3, 2)
        t = range(T)

        ymax = jnp.pi/8

        a[0, 0].set_title('Predicted/optimized trajectories')
        a[0, 0].set_ylabel("$\\theta$")
        a[1, 0].set_ylabel("$\dot{\\theta}$")
        a[2, 0].set_ylabel("$Nm$")
        a[2, 0].set_xlabel("$t$")
        a[0, 0].plot(t, x1, "b+-")
        a[1, 0].plot(t, x2, "b+-")
        a[2, 0].plot(t, u, "b+-")
        a[0, 0].hlines(y=0, xmin=0, xmax=T, linewidth=2, color='r', linestyle='dashed')
        a[1, 0].hlines(y=0, xmin=0, xmax=T, linewidth=2, color='r',linestyle='dashed')
        a[2, 0].hlines(y=0., xmin=0, xmax=T, linewidth=2, color='r',linestyle='dashed')
        a[0, 0].set_ylim(-ymax, ymax)
        a[1, 0].set_ylim(-ymax, ymax)
        a[2, 0].set_ylim(-3, 3)

        a[0, 1].set_title('Trajectory under true dynamics')
        a[0, 1].set_ylabel("$\\theta$")
        a[1, 1].set_ylabel("$\dot{\\theta}$")
        a[2, 1].set_ylabel("$Nm$")
        a[2, 1].set_xlabel("$t$")
        a[0, 1].plot(t, true_x1, "b+-")
        a[1, 1].plot(t, true_x2, "b+-")
        a[2, 1].plot(t, u, "b+-")
        a[0, 1].hlines(y=0, xmin=0, xmax=T, linewidth=2, color='r', linestyle='dashed')
        a[1, 1].hlines(y=0, xmin=0, xmax=T, linewidth=2, color='r', linestyle='dashed')
        a[2, 1].hlines(y=0., xmin=0, xmax=T, linewidth=2, color='r', linestyle='dashed')

        a[0, 1].set_ylim(-ymax, ymax)
        a[1, 1].set_ylim(-ymax, ymax)
        a[2, 1].set_ylim(-3, 3)

        f.suptitle('Pendulum Horizon = %d' % (T-1))

        if save_path is not None:
            plt.savefig(save_path)
        else:
            plt.show()


class LQRVisWatson:

    @staticmethod
    def init_plot():
        PLOT_TIKZ = True
        matplotlib.rcParams["font.family"] = "Times New Roman"
        matplotlib.rcParams["figure.figsize"] = [10, 10]
        matplotlib.rcParams["legend.fontsize"] = 16
        matplotlib.rcParams["axes.titlesize"] = 22
        matplotlib.rcParams["axes.labelsize"] = 22

    @staticmethod
    def plot_trajectory(x1, x2, u, T, save_path=None):
        f, a = plt.subplots(3, 1)

        t = range(T)
        a[0].set_title("State Trajectory")
        a[0].set_ylabel("$x_1$")
        a[1].set_ylabel("$x_2$")
        a[2].set_ylabel("$u$")
        a[2].set_xlabel("$t$")

        a[0].plot(t, x1, "k+-")
        a[1].plot(t, x2, "k+-")
        a[2].plot(t, u, "k+-")

        a[0].hlines(y=10, xmin=0, xmax=T, linewidth=2, color='r', linestyle='dashed')
        a[1].hlines(y=10, xmin=0, xmax=T, linewidth=2, color='r', linestyle='dashed')
        a[2].hlines(y=0., xmin=0, xmax=T, linewidth=2, color='r', linestyle='dashed')
        a[1].set_ylim(-3, 10.5)

        a[0].legend()
        if save_path is not None:
            # 'assets/LQR_watson20.png'
            plt.savefig(save_path)
        else:
            plt.show()


class LQRVisGTSAM:

    @staticmethod
    def init_plot():
        PLOT_TIKZ = True
        matplotlib.rcParams["font.family"] = "Times New Roman"
        matplotlib.rcParams["figure.figsize"] = [10, 10]
        matplotlib.rcParams["legend.fontsize"] = 16
        matplotlib.rcParams["axes.titlesize"] = 22
        matplotlib.rcParams["axes.labelsize"] = 22

    @staticmethod
    def plot_trajectory(x1, true_x, u, T, save_path=None):
        f, a = plt.subplots(2, 1)

        t = range(T)
        a[0].set_title("State Trajectory")
        a[0].set_ylabel("$x$")
        a[1].set_ylabel("$u$")
        a[1].set_xlabel("$t$")

        a[0].plot(t, x1, "k+-", label='Predicted trajectory')
        a[0].plot(t[::2], true_x[::2], "b-", label='True trajectory')
        a[1].plot(t, u, "k+-", label='Planned Action')

        a[0].hlines(y=0, xmin=0, xmax=T, linewidth=2, color='r', linestyle='dashed')
        a[1].hlines(y=0, xmin=0, xmax=T, linewidth=2, color='r', linestyle='dashed')
        # a[1].set_ylim(-3, 10.5)
        plt.legend()

        a[0].legend()
        if save_path is not None:
            # 'assets/LQR_watson20.png'
            plt.savefig(save_path)
        else:
            plt.show()
