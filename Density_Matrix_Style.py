import numpy as np
import matplotlib.pyplot as plt


def Plot_Density_Matrix(Stokes):
    S1, S2, S3 = Stokes
    rho = 0.5 * np.array([
        [1 + S3, S1 - 1j * S2],
        [S1 + 1j * S2, 1 - S3]
    ], dtype=complex)

    re_rho = np.real(rho)
    im_rho = np.imag(rho)

    def plot_matrix_3d(ax, M, title, zlim=None):
        n = M.shape[0]
        x = np.arange(n)
        y = np.arange(n)
        xx, yy = np.meshgrid(x, y)
        xx = xx.ravel()
        yy = yy.ravel()
        zz = np.zeros_like(xx, dtype=float)

        dx = 0.6 * np.ones_like(xx, dtype=float)
        dy = 0.6 * np.ones_like(yy, dtype=float)
        dz = M.ravel()

        colors = ["tab:blue" if v >= 0 else "tab:red" for v in dz]
        ax.bar3d(xx, yy, zz, dx, dy, dz, color=colors, alpha=0.9, shade=True)

        ax.set_xticks(x + 0.3)
        ax.set_yticks(y + 0.3)
        ax.set_xticklabels(["|0>", "|1>"])
        ax.set_yticklabels(["|0>", "|1>"])
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")
        ax.set_zlabel("Value")
        ax.set_title(title)

        if zlim is not None:
            ax.set_zlim(zlim[0], zlim[1])

        for i in range(n):
            for j in range(n):
                v = M[i, j]
                ax.text(j + 0.3, i + 0.3, v + (0.03 if v >= 0 else -0.06), f"{v:.3f}",
                        ha="center", va="center", fontsize=9)

    max_abs = max(np.max(np.abs(re_rho)), np.max(np.abs(im_rho)), 1e-9)
    zlim = (-1.05 * max_abs, 1.05 * max_abs)

    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")

    plot_matrix_3d(ax1, re_rho, "Re(rho)", zlim=zlim)
    plot_matrix_3d(ax2, im_rho, "Im(rho)", zlim=zlim)

    plt.tight_layout()
    plt.show()