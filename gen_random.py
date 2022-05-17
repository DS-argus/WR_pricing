import matplotlib.pyplot as plt
import numpy as np

def gen_BM(U, M, I, corr, fixed_seed=False):
    """
    :param U: int, 기초자산의 개수
    :param M: int, 구간의 개수
    :param I: int, 시뮬레이션의 개수
    :param corr: np.array(U x U), 상관계수행렬(없으면 단위행렬인 np.identity(n= I)로 지정)
    :param fixed_seed: 난수 고정 여부
    :return: np.array (U, M+1, I)
    """
    if fixed_seed:
        np.random.seed(1000)

    dt = 1/M

    random_matrix = np.random.standard_normal(size=(U, M, I))                   # U x M x I

    corr_matrix = corr                                                          # U x U
    cholesky = np.linalg.cholesky(corr_matrix)                                  # U x U

    for i in range(I):
        random_matrix_i = random_matrix[:,:,i].reshape((U, M))                  # U x M
        corr_random_i = np.matmul(cholesky, random_matrix_i)                    # (U x U) x (U x M) = U x M
        corr_random_i = corr_random_i.reshape((U, M, 1))                        # U x M x 1
        if i == 0:
            corr_random = corr_random_i
        else:
            corr_random = np.concatenate((corr_random, corr_random_i), axis=2)

    dB = np.sqrt(dt) * corr_random                                              # len(dB) == M (axis=1기준)
    B0 = np.zeros(shape=(U, 1, I))
    B = np.concatenate((B0, np.cumsum(dB,axis=1)), axis=1)                      # U x M+1 x I    len(B) == M+1 (axis=1기준)

    return B                                               # 0부터 시작하는 brownian motion (구간이 M이면 M+1개의 값 return)

def quadratic_variation(B):
    """ gen_brownian_motion으로 얻은 결과가 정확한지 확인. y=x 꼴과 유사하면 OK """

    return np.cumsum(np.power(np.diff(B, axis=1, prepend=0), 2), axis=1)

def gen_GBM(U, M, I, S, u, sigma, corr, fixed_seed=False):
    """
    시뮬레이션 횟수가 많아질수록 너무 느려짐 --> 개선필요
    :param U: int, 기초자산의 개수
    :param M: int, 구간의 개수
    :param I: int, 시뮬레이션의 개수
    :param S: list(1 x U), 각 기초자산의 S0
    :param u: list(1 x U), 각 기초자산의 평균수익률(구간별로 환산)
    :param sigma: list(1 x U), 각 기초자산의 변동성(구간별로 환산)
    :param corr: list,np.array(U x U), 상관계수행렬(없으면 단위행렬인 np.identity(n= I)로 지정)
    :param fixed_seed: 난수 고정 여부
    :return: np.array (U, M+1, I)
    """
    if fixed_seed:
        np.random.seed(1000)

    dt = 1/M

    random_matrix = np.random.standard_normal(size=(U, M, I))                   # U x M x I

    corr_matrix = corr                                                          # U x U

    # Cholesky 분해를 통한 상관관계를 갖는 난수 생성
    cholesky = np.linalg.cholesky(corr_matrix)                                  # U x U

    for i in range(I):
        random_matrix_i = random_matrix[:,:,i].reshape((U, M))                  # U x M
        corr_random_i = np.matmul(cholesky, random_matrix_i)                    # (U x U) x (U x M) = U x M
        corr_random_i = corr_random_i.reshape((U, M, 1))                        # U x M x 1
        if i == 0:
            corr_random = corr_random_i
        else:
            corr_random = np.concatenate((corr_random, corr_random_i), axis=2)

    W_matrix = corr_random                                                      # U x M x I

    H0 = np.ones(shape=(U, 1, I))                                                # Broadcasting용 array 생성
    S0 = np.array(S).reshape((U, 1, 1))
    S0_matrix = S0 * H0                                                          # U x 1 x I matrix 생성

    H = np.ones(shape=(U, M, I))                                                # Broadcasting용 array 생성
    U_matrix = np.array(u).reshape((U, 1 ,1)) * H                               # U x M x I
    sigma_matrix = np.array(sigma).reshape((U, 1, 1)) * H

    K = [i for i in range(1, M+1)]
    K = np.array(K).reshape(1, M, 1)
    dt_matrix = dt * K * H                                                      # U x M x I
    W_matrix =  np.cumsum((np.sqrt(dt) * W_matrix), axis=1)

    exp_matrix = (U_matrix - 0.5 * sigma_matrix**2) * dt_matrix + sigma_matrix * W_matrix
    exp_matrix = np.exp(exp_matrix)                                             # U x M x I

    St_matrix = S0_matrix * exp_matrix                                          # U x M x I

    S_matrix = np.concatenate((S0_matrix, St_matrix), axis=1)

    return S_matrix

if __name__ == "__main__":

    corr_I = np.identity(n=3)

    corr_1 = [[1]]

    corr_2 = [[1, 0.7],
            [0.7, 1]]

    corr_3 = [[1, 0.279, 0.2895],
            [0.279, 1, 0.5256],
            [0.2895, 0.5256, 1]]



    BM = gen_BM(1, 1000, 1, corr_1)
    QD = quadratic_variation(BM)

    # GBM_1 = gen_GBM(1, 252, 10000, [100], [0.05], [0.15], corr_1)
    # GBM_2 = gen_GBM(2, 1000, 10000, [200,200], [0.05, 0.05], [0.2, 0.2], corr_2)
    # GBM_3 = gen_GBM(3, 1000, 10000, [100,200,300], [0.05, 0.07, 0.08], [0.2, 0.25, 0.3], corr_3)

    x = np.linspace(0, 1, 1001)

    plt.plot(x, BM[0], linewidth= 0.3)
    plt.show()

