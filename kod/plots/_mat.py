import os
import matplotlib.pyplot as plt


def set_theme_and_params():
    plt.style.use(os.environ.get("MAT_THEME", "dark_background"))
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["image.cmap"] = "jet"
    plt.rcParams["font.size"] = 12
    plt.rcParams["legend.fontsize"] = 10
    plt.rcParams["figure.titlesize"] = "medium"
    # plt.rcParams["figure.dpi"] = 200
    # plt.rcParams["savefig.dpi"] = 200
