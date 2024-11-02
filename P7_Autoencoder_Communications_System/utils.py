# Scientific and vector computation for python
# work with math and erfc
import math

# Plotting library
import matplotlib.pyplot as plt
import numpy as np

# Machine Learning Library
import torch
from pylab import *

# Optimization module in scipy
from scipy.special import erfc
from sklearn.manifold import TSNE


# Função Q(x)
def func_q(x):
    return (1 / 2) * erfc((1 / math.sqrt(2)) * x)


# Detector ótimo para os esquemas tradicionais de modulação
def genareteBER(M, EbNodB_range):
    # gera o vetor de BER para cada uma das modulações tradicionais: M-ASK, M-PSK e M-QAM para
    # uma dada lista de SNRs
    m_ask = [None] * len(EbNodB_range)
    m_psk = [None] * len(EbNodB_range)
    m_qam = [None] * len(EbNodB_range)
    for n in range(0, len(EbNodB_range)):
        EbNo = 10.0 ** (EbNodB_range[n] / 10.0)
        m_ask[n] = (
            2
            * (1 - (1 / M))
            * func_q((math.sqrt((6 * math.log2(M)) / (M**2 - 1) * EbNo)))
        )
        m_psk[n] = 2 * func_q(
            math.sin(math.pi / M) * math.sqrt(2 * math.log2(M) * EbNo)
        )
        m_qam[n] = (
            4
            * (1 - (1 / math.sqrt(M)))
            * func_q(math.sqrt((3 * math.log2(M) / (M - 1)) * EbNo))
        )
    return m_ask, m_psk, m_qam


# Retorna um vetor adicionado de ruído Gaussiano
def add_noise(scatter_plot, M, n_channel, n_net=2, EbNo=7, N_noise=1):
    x_emb = scatter_plot
    EbNo_nuvem = 10.0 ** (EbNo / 10.0)
    noise_std = np.sqrt(1 / (2 * (M / n_channel) * EbNo_nuvem)).item()
    x_parse = np.array([N_noise, n_net])
    for j in range(M):
        for i in range(N_noise):
            # noise = noise_std * np.random.randn(1,n_net)
            noise = torch.normal(0.0, noise_std, [1, n_net]).numpy()
            x_parse = x_emb[j, :] + noise
            if j == 0 and i == 0:
                X_embedded_noise = [x_parse]
            else:
                X_embedded_noise = np.concatenate((X_embedded_noise, [x_parse]), axis=0)
    return X_embedded_noise.reshape(X_embedded_noise.shape[0], 2)


# Função que usa a técnica t-SNE para redução das constelações para duas dimensões
def reduceMatrix(x):
    # t-SNE
    X_embedded = TSNE(
        learning_rate=700, n_components=2, n_iter=35000, random_state=0, perplexity=60
    ).fit_transform(x)
    # normalize
    X_embedded = (X_embedded - X_embedded.mean()) / (
        X_embedded.max() - X_embedded.min()
    )
    # return a matrix of type [M,2]
    return X_embedded


# Plot de uma constelação com ou sem ruído
def plotConstellation(X_embedded_noise, X_embedded, flag_noise=False):
    # ploting constellation diagram

    if flag_noise == True:
        plt.plot(X_embedded_noise[:, 0], X_embedded_noise[:, 1], "bo")
        plt.title("Constallation with cloud of Noise")
    else:
        plt.title("Constallation Diagram")
    plt.plot(X_embedded[:, 0], X_embedded[:, 1], "ro")
    # plt.axis((-2.5,2.5,-2.5,2.5))
    plt.grid()
    plt.xlabel("I Axis")
    plt.ylabel("Q Axis")
    plt.show()


# Separação da parte real e imaginária de um símbolo no espaço vetorial
def split_symb(symb):
    return symb[:, ::2], symb[:, 1::2]


# Commented out IPython magic to ensure Python compatibility.
# Plot das contelações e gráficos de indicadores
def plot_graphs(x, y, number_of_subplots=1, n_symb=0, sizefig=20):
    # titulo_centro = ''
    #   %matplotlib inline
    # x e y é uma matriz do tipo Mxn_channel
    # e number_of_subplots = n_channel

    list_color = ["ro", "bo", "go", "ko"]

    subplots_adjust(hspace=0.000)
    if number_of_subplots % 2 == 1:
        n_rows = int(number_of_subplots / 2) + number_of_subplots % 2
    else:
        n_rows = number_of_subplots / 2
    n_rows = int(n_rows)
    f, ax1 = plt.subplots(n_rows, 2, figsize=(sizefig + 5, sizefig))
    # f, ax1 = plt.subplots(n_rows,2,constrained_layout=True)
    for i in range(number_of_subplots):
        v = i + 1
        ax1 = subplot(n_rows, 2, v)
        for j in range(4):
            ax1.plot(x[j, i], y[j, i], list_color[j])
        ax1.set_xlabel("Real Axis")
        ax1.set_ylabel("Imaginary Axis")
        ax1.grid()
        channel = str(i)
        title = "Channel Use: " + channel
        ax1.set_title(title)
    f.tight_layout()  # Or equivalently,  "plt.tight_layout()"
    plt.show()
