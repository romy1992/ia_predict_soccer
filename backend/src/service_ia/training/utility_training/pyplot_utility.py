import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

class PyplotUtility:

    @staticmethod
    def plot_data(x, y, title="Data Plot", xlabel="X-axis", ylabel="Y-axis"):
        """
        Crea un grafico a linee con i dati forniti.
        :param x:
        :param y:
        :param title:
        :param xlabel:
        :param ylabel:
        :return: None
        """
        plt.figure(figsize=(15, 15))
        plt.scatter(x, y, alpha=0.3)
        plt.plot(x, y, marker='o')
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True)
        plt.show()


    @staticmethod
    def plot_heatmap(data: pd.DataFrame, title="Heatmap", xlabel="X-axis", ylabel="Y-axis"):
        plt.figure(figsize=(15, 15))
        sns.heatmap(data.corr(), annot=True, fmt=".2f", cmap="YlGnBu")
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()