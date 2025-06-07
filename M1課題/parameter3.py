import numpy as np
import matplotlib.pyplot as plt

# --- パラメータ設定 ---
lambda_val = 1

# --- 初期条件 ---
z_0 = 2.5e6
x1_0 = 1e-8
x2_0 = 1e-11 * 1.414
Omega_r0 = 0.9989 

# --- 数値計算パラメータ ---
h = 0.01
num_steps = 4000

# --- 初期代入 ---
z = z_0
N_0 = -np.log(1 + z)
x1 = x1_0
x2 = x2_0
Omega_r = Omega_r0
N = N_0

# --- リスト ---
N_list = [N]
x1_list = [x1]
x2_list = [x2]
Omega_r_list = [Omega_r]

# --- 自律方程式の定義 ---
def dx1dN(x1, x2, Omega_r):
    return x1 / 2 * (3 * x1**2 - 3 * x2**2 - 3 + Omega_r) + np.sqrt(6) / 2 * lambda_val * x2**2

def dx2dN(x1, x2, Omega_r):
    return x2 / 2 * (3 + 3 * x1**2 - 3 * x2**2 + Omega_r - np.sqrt(6) * lambda_val * x1)

def dOmega_rdN(x1, x2, Omega_r):
    return Omega_r * (Omega_r - 1 + 3 * x1**2 - 3 * x2**2)

# --- 数値積分（4次ルンゲクッタ法） ---
for _ in range(num_steps):
    # Runge-Kutta 4次
    k1_x1 = h * dx1dN(x1, x2, Omega_r)
    k1_x2 = h * dx2dN(x1, x2, Omega_r)
    k1_Omega_r = h * dOmega_rdN(x1, x2, Omega_r)

    k2_x1 = h * dx1dN(x1 + k1_x1/2, x2 + k1_x2/2, Omega_r + k1_Omega_r/2)
    k2_x2 = h * dx2dN(x1 + k1_x1/2, x2 + k1_x2/2, Omega_r + k1_Omega_r/2)
    k2_Omega_r = h * dOmega_rdN(x1 + k1_x1/2, x2 + k1_x2/2, Omega_r + k1_Omega_r/2)

    k3_x1 = h * dx1dN(x1 + k2_x1/2, x2 + k2_x2/2, Omega_r + k2_Omega_r/2)
    k3_x2 = h * dx2dN(x1 + k2_x1/2, x2 + k2_x2/2, Omega_r + k2_Omega_r/2)
    k3_Omega_r = h * dOmega_rdN(x1 + k2_x1/2, x2 + k2_x2/2, Omega_r + k2_Omega_r/2)

    k4_x1 = h * dx1dN(x1 + k3_x1, x2 + k3_x2, Omega_r + k3_Omega_r)
    k4_x2 = h * dx2dN(x1 + k3_x1, x2 + k3_x2, Omega_r + k3_Omega_r)
    k4_Omega_r = h * dOmega_rdN(x1 + k3_x1, x2 + k3_x2, Omega_r + k3_Omega_r)

    x1 += (k1_x1 + 2*k2_x1 + 2*k3_x1 + k4_x1) / 6
    x2 += (k1_x2 + 2*k2_x2 + 2*k3_x2 + k4_x2) / 6
    Omega_r += (k1_Omega_r + 2*k2_Omega_r + 2*k3_Omega_r + k4_Omega_r) / 6

    N += h
    N_list.append(N)
    x1_list.append(abs(x1))  # logプロット用に絶対値
    x2_list.append(abs(x2))
    Omega_r_list.append(Omega_r)

# --- Nからz+1への変換 ---
# N = -ln(z+1) なので、z+1 = exp(-N)
z_plus_1_list = np.exp(-np.array(N_list))

# --- 理想解の計算 ---
N_array = np.array(N_list)
ideal_x1 = x1_0 * np.exp(-1 * (N_array - N_0))  # 初期値補正を忘れずに
ideal_x2 = x2_0 * np.exp(2 * (N_array - N_0))  # 初期値補正を忘れずに

# --- プロット ---
fig, ax = plt.subplots(figsize=(8,6))
ax.loglog(z_plus_1_list, np.abs(x1_list), label=r"Numerical $x_1$", color="cyan")
ax.loglog(z_plus_1_list, np.abs(x2_list), label=r"Numerical $x_2$", color="magenta")
ax.loglog(z_plus_1_list, ideal_x1, '--', label=r"Ideal $x_1(N) = x_1(0) e^{-N}$", alpha=0.6, color="blue")
ax.loglog(z_plus_1_list, ideal_x2, '--', label=r"Ideal $x_2(N) = x_2(0) e^{2N}$", alpha=0.6, color="red")

ax.set_xlabel(r"$z+1$", fontsize=14)
ax.set_ylabel(r"$x_1, x_2$", fontsize=14)
ax.set_xlim(1, z_0+1)  # z=0（現在）から初期(z=2.5e6)までカバー
ax.legend()
ax.invert_xaxis()  # 赤方偏移が小さい方向に右に進むように
plt.show()

# --- N=0に一番近いインデックスを探す ---
N_array = np.array(N_list)
idx_closest_to_zero = np.argmin(np.abs(N_array))  # N=0に最も近い点

# --- そのインデックスのx1, x2を取得 ---
x1_now = x1_list[idx_closest_to_zero]
x2_now = x2_list[idx_closest_to_zero]
Omega_r_now = Omega_r_list[idx_closest_to_zero]

# --- x1^2 + x2^2を計算して表示 ---
sum_of_squares = x1_now**2 + x2_now**2
print(f"At z=0 (N~0): x1^2 + x2^2 = {sum_of_squares:.6e}, $\Omega_r$ = {Omega_r_now:.6e}")