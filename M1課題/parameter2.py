import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.header("初期値の影響：スカラー場初期値と宇宙進化の関係")

st.subheader("1)$x_2$の初期値を固定して$x_1$の初期値を変動させたときのそれに伴う宇宙進化の変動")

# --- λ 設定 ---
lambda_val = st.sidebar.slider(
    r"スカラー場の勾配 $\lambda$",
    min_value=-float(np.sqrt(2)),
    max_value=float(np.sqrt(2)),
    value=1.0,
    step=0.1,
)

# --- 初期値のパターン ---
x1_0_list = [10**(-10), 10**(-9), 10**(-8)]
x2_0_list = [10**(-11)]  # 固定

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

# --- プロット準備 ---
fig, ax = plt.subplots(figsize=(8, 6))
colors = ['red', 'green', 'blue']
labels = []

for i, x1_0 in enumerate(x1_0_list):
    for j, x2_0 in enumerate(x2_0_list):
        x1, x2 = x1_0, x2_0
        Omega_r = Omega_r0
        Omega_m = 1 - x1**2 - x2**2 - Omega_r
        z = z_0

        z_list = []
        x1_list = []
        x2_list = []
        Omega_DE_list = []
        Omega_r_list = []
        Omega_m_list = []
        w_DE_list = []
        w_eff_list = []

        for _ in range(num_steps):
            z_list.append(z)
            Omega_DE = x1**2 + x2**2
            Omega_m = 1 - Omega_r - Omega_DE
            w_DE = (x1**2 - x2**2) / Omega_DE if Omega_DE > 1e-10 else -1
            w_eff = Omega_r / 3 + Omega_DE * w_DE

            x1_list.append(x1)
            x2_list.append(x2)
            Omega_DE_list.append(Omega_DE)
            Omega_r_list.append(Omega_r)
            Omega_m_list.append(Omega_m)
            w_DE_list.append(w_DE)
            w_eff_list.append(w_eff)

            # Runge-Kutta
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

        # 正規化
        z_arr = np.array(z_list)
        Omega_arr_DE = np.array(Omega_DE_list)
        Omega_arr_r = np.array(Omega_r_list)
        Omega_arr_m = np.array(Omega_m_list)
        w_arr_DE = np.array(w_DE_list)
        w_arr_eff = np.array(w_eff_list)
        x1_arr = np.array(x1_list)
        x2_arr = np.array(x2_list)

        index_shift = np.argmin(np.abs(Omega_arr_DE - 0.68))
        z_shift = z_arr[index_shift]
        shifted_z = z_arr / z_shift

        color = colors[i % len(colors)]
        label = f"x1_0={x1_0:.0e}, x2_0={x2_0:.0e}"
        ax.plot(shifted_z, Omega_arr_r, linestyle="--", color=color, label=f"$\Omega_{{r}}$ ({label})")
        ax.plot(shifted_z, Omega_arr_m, linestyle="-.", color=color, label=f"$\Omega_{{m}}$ ({label})")
        ax.plot(shifted_z, Omega_arr_DE, linestyle=(0,(3, 5, 5)), color=color, label=f"$\Omega_{{\phi}}$ ({label})")
        ax.plot(shifted_z, w_arr_DE, linestyle=":", color=color, label=f"$w_{{DE}}$ ({label})")
        ax.plot(shifted_z, w_arr_eff, linestyle="solid", color=color, label=f"$w_{{eff}}$ ({label})")
        ax.plot(shifted_z, x1_arr, linestyle=(0, (5, 3, 1, 3, 1, 3)), color=color, alpha=0.5, label=f"$x_1$ ({label})")
        ax.plot(shifted_z, x2_arr, linestyle=(5, (10, 3)), color=color, alpha=0.5, label=f"$x_2$ ({label})")

        Omega_DE_z0 = Omega_arr_DE[index_shift]
        Omega_r_z0 = Omega_arr_r[index_shift]
        st.write(f"$x_1 (初期値)$={x1_0:.0e}, $x_2 (初期値)$={x2_0:.0e} のときの $\Omega_\phi (z=0)$ = {Omega_DE_z0:.4f}, $\Omega_r (z=0)$ = {Omega_r_z0:.4f}")

ax.set_xscale("log")
ax.set_xlim(1e-1, 1e6)
ax.set_ylim(-1.1, 1.5)
ax.set_xlabel(r"$z+1$", fontsize=14)
ax.set_ylabel(r"Each Parameter value", fontsize=14)
ax.legend(bbox_to_anchor=(1, 0), loc='lower right', borderaxespad=1, fontsize=6)
st.pyplot(fig)

st.subheader("2)$x_1$の初期値を固定して$x_2$の初期値を変動させたときのそれに伴う宇宙進化の変動")

# --- 初期値のパターン ---
x1_0_list = [10**(-10)]
x2_0_list = [10**(-11), 10**(-10), 10**(-9)]
#x2の初期値を固定してx1の初期値を変動させると、グラフは微動だにしない。しかし、x1の初期値を固定してx2の初期値を変動させると、グラフはめっちゃ動く。
#めっちゃ動くのはOmega_r=Omega_mとなるzが変動することにつながる。
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

# --- プロット準備 ---
fig, ax = plt.subplots(figsize=(8, 6))
colors = ['red', 'green', 'blue']
labels = []

for i, x1_0 in enumerate(x1_0_list):
    for j, x2_0 in enumerate(x2_0_list):
        x1, x2 = x1_0, x2_0
        Omega_r = Omega_r0
        Omega_m = 1 - x1**2 - x2**2 - Omega_r
        z = z_0

        z_list = []
        x1_list = []
        x2_list = []
        Omega_DE_list = []
        Omega_r_list = []
        Omega_m_list = []
        w_DE_list = []
        w_eff_list = []

        for _ in range(num_steps):
            z_list.append(z)
            Omega_DE = x1**2 + x2**2
            Omega_m = 1 - Omega_r - Omega_DE
            w_DE = (x1**2 - x2**2) / Omega_DE if Omega_DE > 1e-10 else -1
            w_eff = Omega_r / 3 + Omega_DE * w_DE

            x1_list.append(x1)
            x2_list.append(x2)
            Omega_DE_list.append(Omega_DE)
            Omega_r_list.append(Omega_r)
            Omega_m_list.append(Omega_m)
            w_DE_list.append(w_DE)
            w_eff_list.append(w_eff)

            # Runge-Kutta
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

        # 正規化
        z_arr = np.array(z_list)
        Omega_arr_DE = np.array(Omega_DE_list)
        Omega_arr_r = np.array(Omega_r_list)
        Omega_arr_m = np.array(Omega_m_list)
        w_arr_DE = np.array(w_DE_list)
        w_arr_eff = np.array(w_eff_list)
        x1_arr = np.array(x1_list)
        x2_arr = np.array(x2_list)

        index_shift = np.argmin(np.abs(Omega_arr_DE - 0.68))
        z_shift = z_arr[index_shift]
        shifted_z = z_arr / z_shift

        color = colors[j % len(colors)]
        label = f"x1_0={x1_0:.0e}, x2_0={x2_0:.0e}"
        ax.plot(shifted_z, Omega_arr_r, linestyle="--", color=color, label=f"$\Omega_{{r}}$ ({label})")
        ax.plot(shifted_z, Omega_arr_m, linestyle="-.", color=color, label=f"$\Omega_{{m}}$ ({label})")
        ax.plot(shifted_z, Omega_arr_DE, linestyle=(0,(3, 5, 5)), color=color, label=f"$\Omega_{{\phi}}$ ({label})")
        ax.plot(shifted_z, w_arr_DE, linestyle=":", color=color, label=f"$w_{{DE}}$ ({label})")
        ax.plot(shifted_z, w_arr_eff, linestyle="solid", color=color, label=f"$w_{{eff}}$ ({label})")
        ax.plot(shifted_z, x1_arr, linestyle=(0, (5, 3, 1, 3, 1, 3)), color=color, alpha=0.5, label=f"$x_1$ ({label})")
        ax.plot(shifted_z, x2_arr, linestyle=(5, (10, 3)), color=color, alpha=0.5, label=f"$x_2$ ({label})")

        Omega_DE_z0 = Omega_arr_DE[index_shift]
        Omega_r_z0 = Omega_arr_r[index_shift]
        st.write(f"$x_1 (初期値)$={x1_0:.0e}, $x_2 (初期値)$={x2_0:.0e} のときの $\Omega_\phi (z=0)$ = {Omega_DE_z0:.4f}, $\Omega_r (z=0)$ = {Omega_r_z0:.4f}")

ax.set_xscale("log")
ax.set_xlim(1e-1, 1e6)
ax.set_ylim(-1.1, 1.5)
ax.set_xlabel(r"$z+1$", fontsize=14)
ax.set_ylabel(r"Each Parameter value", fontsize=14)
ax.legend(bbox_to_anchor=(1, 0), loc='lower right', borderaxespad=1, fontsize=6)
st.pyplot(fig)