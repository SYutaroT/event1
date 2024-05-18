# ライブラリのインポート
import torch  # 機械学習用のライブラリ
import torch.nn as nn  # ニューラルネットワーク構築
import torch.optim as optim  # 最適化
from torch.optim.lr_scheduler import StepLR  # 学習率調整
from torch.optim.lr_scheduler import CosineAnnealingLR  # 学習率調整
import numpy as np  # 計算用のライブラリ
import matplotlib.pyplot as plt  # グラフのプロット
import copy  # コピー
from random import randrange  # ランダム
from scipy.stats import beta  # ベータ分布
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import LambdaLR
import os
import parameter
import time
import winsound
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
# ディレクトリのパス

start_time = time.time()
# ?----------------------------------------------------------------保存
name = parameter.Folder
File = parameter.Dt
path = File + name+"R"
directory = os.path.dirname(path)

# ディレクトリが存在しない場合、作成する
if not os.path.exists(directory):
    os.makedirs(directory)
# ?----------------------------------------------------------------乱数固定


def seedinit(seed):
    torch.manual_seed(int(seed))
    np.random.seed(seed)


# ?----------------------------------------------------------------ニューラルネットワークの構築


class DisturbanceDataset(Dataset):  # ミニバッチ学習用の学習データ作成プログラム
    def __init__(self, num_samples, N, T, minw, maxw, alpha, beta2):
        self.disturbances = [torch.tensor(beta.rvs(alpha, beta2, size=(
            N, T - 1), loc=minw, scale=maxw - minw)) for _ in range(num_samples)]

    def __len__(self):
        return len(self.disturbances)

    def __getitem__(self, idx):
        return self.disturbances[idx]


class CSTR(nn.Module):  # 深層展開のメインクラス
    def __init__(self, T, N, epoch, device, path):  # パラメータの定義
        super(CSTR, self).__init__()
        self.device = device
        self.x0 = torch.tensor([1], dtype=torch.double)
        self.xf = torch.tensor([0.0], dtype=torch.double)
        self.T = T
        self.N = N
# ?----------------------------------------------------------------最適化パラメータ
        self.K = nn.Parameter(torch.zeros(
            1, self.N, dtype=torch.double, requires_grad=True, device=device))
        self.L = nn.Parameter(torch.zeros(
            2*N, 2*N, dtype=torch.double, requires_grad=True, device=device))
        self.M = nn.Parameter(torch.zeros(
            1, 2*N, dtype=torch.double, requires_grad=True, device=device))
        self.Mo = nn.Parameter(torch.zeros(
            1, 1, dtype=torch.double, requires_grad=True, device=device))


# ?----------------------------------------------------------------外乱設定
        self.alpha = 1
        self.beta = 1
        self.minw = parameter.wmin
        self.maxw = parameter.wmax
# ?----------------------------------------------------------------コスト関数
        self.Q = torch.tensor([[1, 0], [0, 1]],
                              dtype=torch.double, device=device)
        self.Qf = 10*torch.eye(N, dtype=torch.double, device=device)
        self.R = torch.eye(1, dtype=torch.double, device=device)
        self.lam = 1
# ?----------------------------------------------------------------その他
        self.batchSize = 2
        self.epoch = epoch
        self.lr = 0.5
        self.delta = torch.zeros(1, self.T-1, dtype=torch.double)
        self.beta_val = 3.0
        self.h = 0.01
        self.sig = nn.Sigmoid()
        self.x = torch.tensor([[1.0]], dtype=torch.double, device=device)
        self.best = 0
        self.path = path
        self.datasize = 32
# ?----------------------------------------------------------------関数

    def f(self, x, u):  # 4次のルンゲクッタ法による離散化
        h = self.h
        Rx = torch.zeros(2, 1, dtype=torch.double).to(self.device)
        x1 = x[0]
        x2 = x[1]

        k1_x1 = parameter.x1_dot(x1, x2, u)
        k1_x2 = parameter.x2_dot(x1, x2, u)

        k2_x1 = parameter.x1_dot(x1, x2 + h / 2, u)
        k2_x2 = parameter.x2_dot(x1, x2 + h / 2, u)

        k3_x1 = parameter.x1_dot(x1, x2 + h / 2, u)
        k3_x2 = parameter.x2_dot(x1, x2 + h / 2, u)

        k4_x1 = parameter.x1_dot(x1, x2 + h, u)
        k4_x2 = parameter.x2_dot(x1, x2 + h, u)

        x1_dot_avg = (k1_x1 + 2 * k2_x1 + 2 * k3_x1 + k4_x1) / 6
        x2_dot_avg = (k1_x2 + 2 * k2_x2 + 2 * k3_x2 + k4_x2) / 6

        Rx[0] = x1 + h * x1_dot_avg
        Rx[1] = x2 + h * x2_dot_avg
        return Rx

    def phi(self, x, Ix):  # !柔軟なトリガ
        Rx = torch.zeros(2*self.N, 1, dtype=torch.double).to(self.device)
        for i in range(0, self.N):
            Rx[i] = x[i]
        for i in range(0, self.N):
            Rx[i+self.N] = Ix[i]
        return Rx.T@self.L@Rx+self.M@Rx+self.Mo

    def J(self, T, w):  # !コストを算出
        cost = torch.zeros(1, dtype=torch.double).to(self.device)
        x, delta = self.makex(T, w)
        for i in range(0, T-1):
            cost = cost+self.StageCost(x[:, [i]],
                                       self.K@x[:, [i]], delta[:, [i]])
        cost = cost+self.FinalCost(x[:, [T]])
        return cost

    def BatchJ(self, T, w):  # !バッチ処理用
        cost = torch.zeros(1, self.batchSize,
                           dtype=torch.double).to(self.device)
        for i in range(self.batchSize):
            # print("w", w)
            cost[0, i] = self.J(T, w[i])
        return cost
# ?----------------------------------------------------------------x,x_hatの生成

    def makex(self, T, w):  # !制御プラントから次のxを生成
        x = torch.zeros(self.N, self.T, dtype=torch.double).to(self.device)
        x_hat = torch.zeros(self.N, 1, dtype=torch.double).to(self.device)
        delta = torch.zeros(1, self.T-1, dtype=torch.double).to(self.device)
        u = torch.zeros(1, self.T-1, dtype=torch.double).to(self.device)
        for i in range(0, self.N):
            x[i, 0] = self.x0[i]
            x_hat[i, 0] = self.x0[i]
        for i in range(0, T):
            if i == 0:
                delta[:, [i]] = 1
                u[:, [i]] = self.K@x_hat
            else:
                delta[:, [i]] = self.sig(
                    self.phi(x[:, [i]], x_hat))
                x_hat = delta[:, [i]]*x[:, [i]] + \
                    (1-delta[:, [i]])*x_hat
            u[:, [i]] = self.K@x_hat
            x[:, [i+1]] = self.f(x[:, [i]], u[:, [i]])+w[:, [i]]
        self.delta = delta
        self.x = x
        return x, delta
# ?----------------------------------------------------------------コスト計算

    def StageCost(self, x, u, delta):  # !ステージコスト
        return (x-self.xf).T @ self.Q @ (x-self.xf) + u.T @ self.R @ u+self.lam*delta

    def FinalCost(self, x):  # !終端コスト
        return (x-self.xf).T @ self.Qf @ (x-self.xf)


# ?----------------------------------------------------------------メイン

    def utrain(self):  # !最重要、ここで訓練
        torch.autograd.set_detect_anomaly(True)
        optimizer = optim.AdamW(
            self.parameters(), lr=self.lr)  # 最適化手法、今回はAdamW
        target = torch.zeros(
            1, self.batchSize, dtype=torch.double).to(self.device)
        cosine_scheduler = CosineAnnealingWarmRestarts(
            optimizer, T_0=int(self.T/5), T_mult=4, eta_min=1e-7)  # 学習率スケジューラ、今回は全ステップの1/5分学習率を減少、その後再度減少
        liner_scheduler = LambdaLR(
            optimizer, lr_lambda=lambda epoch: 10 ** epoch)  # LR Range Test用の学習率スケジューラ、これはエポックごとにLrを10倍
        self.train()
        loss_fn = nn.L1Loss()
        loss_history = []
        lr_history = []
        for e in range(self.epoch):  # epoch分繰り返す：これが1サイクル
            print("epoch", e)
            for i in range(1, self.T):  # インクリメンタル学習、T_maxまで層を増やしながら学習
                dataset = DisturbanceDataset(
                    num_samples=self.datasize, N=self.N, T=i+1, minw=self.minw, maxw=self.maxw, alpha=self.alpha, beta2=self.beta)

                dataloader = DataLoader(
                    dataset, batch_size=self.batchSize, shuffle=True)

                if i % 10 == 0:
                    print("Training", i)
                for data in dataloader:  # ミニバッチ学習、インテレーション数分学習
                    optimizer.zero_grad()
                    costJ = self.BatchJ(i, data.to(self.device))
                    loss = loss_fn(costJ.squeeze(), target)
                    loss.backward()
                    optimizer.step()
                # TODO 学習率スケジューラを付けたいなら下のプログラムをコメントアウトから解放
                # current_lr = optimizer.param_groups[0]['lr']
                # loss_history.append(loss.item())
                # lr_history.append(current_lr)
                # cosine_scheduler.step()
            # TODO LR Range Testがしたいなら下のプログラムをコメントアウトから解放
            liner_scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            loss_history.append(loss.item())
            lr_history.append(current_lr)
        # TODO 学習率の推移、評価関数と学習率関係をプロット
        plt.figure(figsize=(10, 6))
        plt.plot(lr_history, loss_history, marker='o', linestyle='-')
        plt.title('Loss vs. Learning Rate')
        plt.xscale('log')
        plt.xlabel('Learning Rate')
        plt.ylabel('Loss')
        plt.grid(True)
        path = self.path
        plt.savefig(path + "lrVScost" + str(self.epoch)+"lr" +
                    str(self.lr)+"T"+str(self.T) + ".svg")
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(lr_history) + 1),
                 lr_history, marker='o', linestyle='-')
        plt.title('Learning Rate over Epochs')
        plt.xlabel('Epoch')
        plt.yscale('log')
        plt.ylabel('Learning Rate')
        plt.grid(True)
        path = self.path
        plt.savefig(path + "学習率se" + str(self.epoch)+"lr" +
                    str(self.lr)+"T"+str(self.T) + ".svg")

# ?----------------------------------------------------------------外乱生成

    def w_for_training(self):  # !訓練用の外乱
        return torch.tensor(beta.rvs(self.alpha, self.beta, size=(self.N, self.T - 1), loc=self.minw, scale=self.maxw - self.minw)).to(self.device)

    def w_for_evaluation(self, T):  # !評価用の外乱
        return torch.tensor(beta.rvs(self.alpha, self.beta, size=(self.N, T), loc=self.minw, scale=self.maxw - self.minw)).to(self.device)


# ?----------------------------------------------------------------ここからスタート
seedinit(1)
device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
# !----------------------------------------------------------------数値設定(必要に応じて変更)
T = parameter.T
N = parameter.N
epoch = parameter.epoch
beta_val = parameter.beta_val
h = parameter.h
lr = parameter.lr
lam = parameter.lam
Batch = parameter.Batch
datasize = parameter.datasize
x0 = parameter.x0.to(device)
xf = torch.tensor([[0.0], [0.0]], dtype=torch.double, device=device)
# TODO グラフ保存
path2 = "e" + str(epoch) + "B" + str(Batch) + "\\w"+str(parameter.wmin)+str(parameter.wmax) + \
    "\\lr" + str(lr) + "\\T" + str(T) + "-h" + str(h) + "\\lam" + str(lam)+"\\"
imgname = path+path2
data = imgname+"L2data.txt"
directory = os.path.dirname(imgname)
if not os.path.exists(directory):
    os.makedirs(directory)
cstr = CSTR(T, N, epoch, device, imgname)


# ?----------------------------------------------------------------関数


def f(x, u, h):
    Rx = torch.zeros(2, 1, dtype=torch.double).to(device)
    x1 = x[0]
    x2 = x[1]

    k1_x1 = parameter.x1_dot(x1, x2, u)
    k1_x2 = parameter.x2_dot(x1, x2, u)

    k2_x1 = parameter.x1_dot(x1, x2 + h / 2, u)
    k2_x2 = parameter.x2_dot(x1, x2 + h / 2, u)
    k3_x1 = parameter.x1_dot(x1, x2 + h / 2, u)
    k3_x2 = parameter.x2_dot(x1, x2 + h / 2, u)
    k4_x1 = parameter.x1_dot(x1, x2 + h, u)
    k4_x2 = parameter.x2_dot(x1, x2 + h, u)

    x1_dot_avg = (k1_x1 + 2 * k2_x1 + 2 * k3_x1 + k4_x1) / 6
    x2_dot_avg = (k1_x2 + 2 * k2_x2 + 2 * k3_x2 + k4_x2) / 6
    Rx[0] = x1 + h * x1_dot_avg
    Rx[1] = x2 + h * x2_dot_avg
    return Rx


def J(x, u, Q, R):
    return x.T @ Q @ x + u.T @ R @ u


def calculate_cost(w, T, N, x0, Q, R, Qf, lam, K, h, Name, No):
    x = torch.zeros(2, T, dtype=torch.double).to(device)
    x_hat = torch.zeros((N, 1), dtype=torch.double).to(device)
    u = torch.zeros(1, T-1, dtype=torch.double).to(device)
    mark = torch.zeros(1, T-1, dtype=torch.double).to(device)
    cost = torch.zeros(1, dtype=torch.double).to(device)
    cost2 = torch.zeros(1, dtype=torch.double).to(device)
    mark2 = torch.zeros(1, T-1, dtype=torch.double).to(device)
    Rx = torch.zeros(4, 1, dtype=torch.double).to(device)
    Rx_data = torch.zeros(1, T-1, dtype=torch.double).to(device)
    x_t = torch.zeros(2, T, dtype=torch.double).to(device)
    u_t = torch.zeros(1, T-1, dtype=torch.double).to(device)
    cnt = 0
    for i in range(0, N):
        x[i, 0] = x0[i]
        x_hat[i, 0] = x0[i]
        x_t[i, 0] = x0[i]
    for i in range(0, T-1):
        if i == 0:
            mark[:, i] = 1
            Rx_data[:, i] = 1
        else:
            Rx[0] = x[0, [i]]
            Rx[1] = x[1, [i]]
            Rx[2] = x_hat[0, [0]]
            Rx[3] = x_hat[1, [0]]
            if 0 < Rx.T@L2@Rx+L1@Rx+L0:
                x_hat[:, [0]] = x[:, [i]]
                mark[:, i] = 1
                cnt += 1
            else:
                mark[:, i] = 0
            Rx_data[:, i] = torch.sigmoid(Rx.T@L2@Rx+L1@Rx+L0)
        u[:, [i]] = K@x_hat
        x[:, [i+1]] = f(x[:, [i]], u[:, [i]], h)+w[:, [i]]
    for i in range(0, T-1):
        cost = cost + x[:, i].T@Q@x[:, i]+u[:, i]@R@u[:, i]+lam*mark[:, i]

    cost = cost + x[:, T-1].T@Qf@x[:, T-1]

    imgname = Name+"_ev"+str(No)
    for i in range(T-1):
        u_t[:, i] = K@x_t[:, i]
        for j in range(N):
            x_t[j, i+1] = f(x_t[:, i], u_t[:, i], h)[j]+w[j, i]
        mark2[:, i] = 1
    for i in range(0, T-1):
        cost2 += x_t[:, i].T@Q@x_t[:, i]+u_t[:, i]@R@u_t[:, i]+lam*mark2[:, i]
    cost2 += x_t[:, T-1].T@Qf@x_t[:, T-1]

    return cost.item(), mark.sum().item(), cost2.item()


# ?----------------------------------------------------------------トレーニング
cstr.x0 = x0
cstr.xf = xf
cstr.beta_val = beta_val
cstr.h = h
cstr.lr = lr
cstr.lam = lam
cstr.batchSize = Batch
cstr.datasize = datasize
cstr.utrain()
# ?----------------------------------------------------------------変数
x = torch.zeros((N, T), dtype=torch.double).to(device)
x_hat = torch.zeros((N, T-1), dtype=torch.double).to(device)
u = torch.zeros(1, T-1, dtype=torch.double).to(device)
mark = torch.zeros(1, T-1, dtype=torch.double).to(device)
K = cstr.K.clone()
delta = cstr.delta.clone()
Rx = torch.zeros(4, 1, dtype=torch.double).to(device)
Rx_data = torch.zeros(1, T-1, dtype=torch.double).to(device)
cnt = 0  # 更新回数
# ?----------------------------------------------------------------
L2 = cstr.L.clone()
L1 = cstr.M.clone()
L0 = cstr.Mo.clone()
Q = cstr.Q.clone()
Qf = cstr.Qf.clone()
R = cstr.R.clone()
best = cstr.best
print("K", K)
print("L2", L2)
print("L1", L1)
print("L0", L0)
# ?----------------------------------------------------------------データの保存


if not os.path.exists(directory):
    os.makedirs(directory)
with open(data, "w") as file:
    file.write("Title:\n")
    file.write(f"{imgname}\n")
with open(data, "a") as file:  # 追加モード
    file.write("\nL0:\n")
    for value in L0:
        file.write(f"{value}\n")
with open(data, "a") as file:  # 追加モード
    file.write("\nL1:\n")
    for value in L1:
        file.write(f"{value}\n")
with open(data, "a") as file:  # 追加モード
    file.write("\nL2:\n")
    for value in L2:
        file.write(f"{value}\n")
with open(data, "a") as file:  # 追加モード
    file.write("\nK:\n")
    for value in K:
        file.write(f"{value}\n")
# ?----------------------------------------------------------------シミュレーション
counter = 0
total_cost = 0
total_cnt = 0
w_list = []

for i in range(parameter.evolution):
    seedinit(i)
    w = cstr.w_for_evaluation(T-1)
    w_list.append(w)

for w in w_list:
    counter += 1
    cost,  cnt, cost2 = calculate_cost(
        w, T, N, x0, Q, R, Qf, lam, K, h, imgname, counter)

    with open(data, "a") as file:  # 追加モード
        file.write("cost_ev"+str(counter)+"\n")
        file.write(f"{cost}\n")
    with open(data, "a") as file:  # 追加モード
        file.write("cost2_ev"+str(counter)+"\n")
        file.write(f"{cost2}\n")
    with open(data, "a") as file:
        file.write("cnt_ev"+str(counter)+"\n")
        file.write(f"{cnt}\n")

    total_cost += cost
    total_cnt += cnt
average_cost = total_cost / counter
average_cnt = total_cnt / counter

with open(data, "a") as file:  # 追加モード
    file.write("cost_AVE\n")
    file.write(f"{average_cost}\n")
with open(data, "a") as file:
    file.write("cnt_AVE\n")
    file.write(f"{average_cnt}\n")
parameters_content = f"""# Generated Parameters
T= {T}
L2 = {L2.tolist()}
L1 = {L1.tolist()}
L0 = {L0.tolist()}
K = {K.tolist()}
"""

parameters_file_path = imgname+"result.py"
with open(parameters_file_path, "w") as file:
    file.write(parameters_content)
# ?----------------------------------------------------------------時間
time.sleep(2)  # 例として2秒間の遅延

# 処理終了後の時刻
end_time = time.time()

# 経過時間を計算
elapsed_time = end_time - start_time
print(f"Time: {elapsed_time} /s")
with open(data, "a") as file:  # Append mode
    file.write(f"\nTime: {elapsed_time} /s\n")
time.sleep(5)  # 5秒間待機するダミーの処理

# 処理終了後にビープ音を鳴らす
winsound.Beep(1000, 1000)
