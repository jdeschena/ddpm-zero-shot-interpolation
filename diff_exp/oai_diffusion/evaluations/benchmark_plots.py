import matplotlib.pyplot as plt
import wandb
import pandas as pd
import numpy as np


def remove_cols_with_word(df, word):
    df = df.iloc[:,~df.columns.str.contains(word, case=False)] 
    return df


def calculate_ema(data, alpha):
    ema = [data[0]]  # Initialize EMA with the first value

    for x in data:
        ema_value = alpha * ema[-1] + (1 - alpha) * x
        ema.append(ema_value)

    return np.array(ema[1:])


def plot_with_ema(x, signal, legend, alpha, color="r", ema=0.9):
    sig_ema = calculate_ema(signal, ema)
    plt.plot(x, signal, c=color, alpha=alpha)
    plt.plot(x, sig_ema, c=color, label=legend)

    



data = pd.read_csv("params_norm.csv")
data = remove_cols_with_word(data, "MIN")
data = remove_cols_with_word(data, "MAX")

start = 50
x = data.iloc[start:, 0]
y1 = data.iloc[start:, 1].to_numpy()
y2 = data.iloc[start:, 2].to_numpy()
y3 = data.iloc[start:, 3].to_numpy()

ema = 0.95
alpha = 0.1
plot_with_ema(x, y1, "BF16", alpha, color="r", ema=ema)
plot_with_ema(x, y2, "FP16", alpha, color="b", ema=ema)
plot_with_ema(x, y3, "FP32", alpha, color="g", ema=ema)

plt.legend()
plt.title(f"Parameters norm (ema {ema})")

plt.tight_layout()
plt.savefig("params_norm.png")
