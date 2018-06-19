import numpy as np
from matplotlib import pyplot as plt

st_num = 3  # 素線数
m = 100  # 分割数
It = 300  # 通電電流
l = 1  # 素線長さ
dx = l / m  # 分割幅

r = np.ones((1, st_num))
g = np.ones((1, st_num))

# ========行列をいじる============
# r[0, 0] = 0.5  # 素線1の抵抗を1に設定
# r[0, 1] = 3

rr = np.zeros((m, 1, st_num))  # 抵抗行列の初期化
gg = np.zeros((m, st_num, st_num))  # コンダクタンス行列の初期化

for i in range(m):
    rr[i] = r
    gg[i] = g
# ======抵抗行列を操作===========================
rr[int(m / 2), 0, 0] = 10  # m/2番目の素線1の抵抗を10倍に設定


R = np.zeros((m, st_num, st_num))  # 抵抗セルの初期化
G = np.zeros((m, st_num, st_num))  # コンダクタンスセルの初期化
M = np.zeros((m, st_num, st_num))  # M = G * R

# ========行列R、Gの作成=======================
for i in range(m):
    for j in range(st_num):
        for k in range(st_num):
            if j == k:
                R[i, j, k] = rr[i, :, k]
                G[i, j, k] = -1 * (sum(gg[i, j, :])-gg[i, j, k])
            else:
                G[i, j, k] = gg[i, j, k]

    M[i] = np.matrix(G[i]) * np.matrix(R[i])
# =======初期条件======================
y1_0 = np.matrix(np.ones((st_num, 1))) * It / st_num
y2_0 = np.matrix(np.zeros((st_num, 1)))

# =======計算用行列の初期化====================
y1 = np.zeros((m, st_num, 1))
y2 = np.zeros((m, st_num, 1))


for i in range(m):
    y1[i] = np.matrix(np.zeros((st_num, 1)))
    y2[i] = np.matrix(np.zeros((st_num, 1)))

y1[0] = np.matrix(y1_0)
y2[0] = np.matrix(y2_0)

x = np.zeros((1, m, 1))

YY = np.zeros((m, st_num))
YY2 = np.zeros((m, st_num))

# =======ルンゲ・クッタ法の適用=================
for i in range(1, m):
    k1_y1 = np.matrix(y2[i-1])
    k1_y2 = np.matrix(M[i-1]) * np.matrix(y1[i-1])
    k2_y1 = np.matrix(y2[i-1]) + dx * np.matrix(k1_y2) / 2
    k2_y2 = np.matrix(M[i-1]) * np.matrix(y1[i-1] + dx * k1_y1 / 2)
    k3_y1 = np.matrix(y2[i-1]) + dx * np.matrix(k2_y2) / 2
    k3_y2 = np.matrix(M[i-1]) * np.matrix(y1[i-1] + dx * k2_y1 / 2)
    k4_y1 = np.matrix(y2[i-1]) + dx * np.matrix(k3_y2)
    k4_y2 = np.matrix(M[i-1]) * np.matrix(y1[i-1] + dx * k3_y1)

    k_y1 = (dx / 6) * np.matrix(k1_y1 + 2 * k2_y1 + 2 * k3_y1 + k4_y1)
    k_y2 = (dx / 6) * np.matrix(k1_y2 + 2 * k2_y2 + 2 * k3_y2 + k4_y2)

    y1[i] = np.matrix(y1[i-1]) + np.matrix(k_y1)
    y2[i] = np.matrix(y2[i-1]) + np.matrix(k_y2)
    x[0, i, 0] = x[0, i-1, 0] + dx

for i in range(m):
    for j in range(st_num):
        YY[i, j] = y1[i, j, 0]
        YY2[i, j] = y2[i, j, 0]

data = np.hstack((x[0], YY))  # 行列の連結

# ======データのプロット=============
plt.plot(data[:, 0], data[:, 1:st_num])
plt.show()

# ========テキストファイルに保存=============
np.savetxt('data.txt', data)









