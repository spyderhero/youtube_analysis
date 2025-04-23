import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.title("初期値の影響：スカラー場初期値と宇宙進化の関係")

# --- λ 設定 ---
lambda_val = st.sidebar.slider(
    r"スカラー場の勾配 $\lambda$",
    min_value=-float(np.sqrt(2)),
    max_value=float(np.sqrt(2)),
    value=1.0,
    step=0.1,
)

# --- 初期値のパターン ---
x1_0_list = [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
x2_0_list = [1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]

z_0 = 2.5 * 10**6
Omega_r0 = 0.9989
h = 0.01
num_steps = 4000

# --- 微分方程式定義 ---
def dx1dN(x1, x2, Omega_r):
    return x1 / 2 * (3 * x1**2 - 3 * x2**2 - 3 + Omega_r) + np.sqrt(6) / 2 * lambda_val * x2**2

def dx2dN(x1, x2, Omega_r):
    return x2 / 2 * (3 + 3 * x1**2 - 3 * x2**2 + Omega_r - np.sqrt(6) * lambda_val * x1)

def dOmega_rdN(x1, x2, Omega_r):
    return Omega_r * (Omega_r - 1 + 3 * x1**2 - 3 * x2**2)

# --- プロット ---
fig, ax = plt.subplots(figsize=(8, 6))

colors = ['red', 'green', 'blue', 'purple', 'brown', 'cyan', 'orange', 'gray', 'pink']
labels = []

for i, x1_0 in enumerate(x1_0_list):
    for j, x2_0 in enumerate(x2_0_list):
        # 初期値
        x1, x2 = x1_0, x2_0
        Omega_r = Omega_r0
        Omega_m = 1 - x1**2 - x2**2 - Omega_r
        z = z_0

        # リスト初期化
        z_list = [z]
        Omega_DE_list = [x1**2 + x2**2]

        # 数値積分
        for _ in range(num_steps):
            k1_x1 = h * dx1dN(x1, x2, Omega_r)
            k1_x2 = h * dx2dN(x1, x2, Omega_r)
            k1_Omega_r = h * dOmega_rdN(x1, x2, Omega_r)

            k2_x1 = h * dx1dN(x1 + k1_x1 / 2, x2 + k1_x2 / 2, Omega_r + k1_Omega_r / 2)
            k2_x2 = h * dx2dN(x1 + k1_x1 / 2, x2 + k1_x2 / 2, Omega_r + k1_Omega_r / 2)
            k2_Omega_r = h * dOmega_rdN(x1 + k1_x1 / 2, x2 + k1_x2 / 2, Omega_r + k1_Omega_r / 2)

            k3_x1 = h * dx1dN(x1 + k2_x1 / 2, x2 + k2_x2 / 2, Omega_r + k2_Omega_r / 2)
            k3_x2 = h * dx2dN(x1 + k2_x1 / 2, x2 + k2_x2 / 2, Omega_r + k2_Omega_r / 2)
            k3_Omega_r = h * dOmega_rdN(x1 + k2_x1 / 2, x2 + k2_x2 / 2, Omega_r + k2_Omega_r / 2)

            k4_x1 = h * dx1dN(x1 + k3_x1, x2 + k3_x2, Omega_r + k3_Omega_r)
            k4_x2 = h * dx2dN(x1 + k3_x1, x2 + k3_x2, Omega_r + k3_Omega_r)
            k4_Omega_r = h * dOmega_rdN(x1 + k3_x1, x2 + k3_x2, Omega_r + k3_Omega_r)

            x1 += (k1_x1 + 2*k2_x1 + 2*k3_x1 + k4_x1) / 6
            x2 += (k1_x2 + 2*k2_x2 + 2*k3_x2 + k4_x2) / 6
            Omega_r += (k1_Omega_r + 2*k2_Omega_r + 2*k3_Omega_r + k4_Omega_r) / 6

            z = z / np.exp(h)
            Omega_DE = x1**2 + x2**2
            z_list.append(z)
            Omega_DE_list.append(Omega_DE)

        z_arr = np.array(z_list)
        Omega_arr = np.array(Omega_DE_list)

        # z=0での位置に合わせて正規化
        index_shift = np.argmin(np.abs(Omega_arr - 0.68))
        z_shift = z_arr[index_shift]
        shifted_z = z_arr / z_shift

        color = colors[(i * len(x2_0_list) + j) % len(colors)]
        label = f"x1_0={x1_0:.0e}, x2_0={x2_0:.0e}"
        ax.plot(shifted_z, Omega_arr, label=label, color=color)
        labels.append(label)

        # z=0に最も近いインデックスを取得して出力
        index_z0 = np.argmin(np.abs(np.array(Omega_DE_list) - 0.68))
        Omega_z0 = Omega_arr[index_z0]
        st.write(f"x1_0={x1_0:.0e}, x2_0={x2_0:.0e} のときの Ω_phi(z=0) = {Omega_z0:.4f}")
ax.set_xscale("log")
ax.set_xlim(1e-1, 1e6)
ax.set_ylim(0, 1.5)
ax.set_xlabel(r"$z+1$", fontsize=14)
ax.set_ylabel(r"$\Omega_\phi$", fontsize=14)
ax.legend(fontsize=8)

st.pyplot(fig)
