import streamlit as st #streamlit run (このファイル名).py で起動
import numpy as np
import matplotlib.pyplot as plt

st.title("宇宙の進化：スカラー場と密度パラメータの関係")
st.write("高赤方偏移から低赤方偏移までのスカラー場と輻射・物質・暗黒エネルギーの進化を数値的に再現したグラフは以下のようになる。")

# --- パラメータ設定 ---
lambda_val = st.sidebar.slider(
    r"スカラー場の勾配 $\lambda$",
    min_value=-float(np.sqrt(2)),
    max_value=float(np.sqrt(2)),
    value=1.0,
    step=0.1,
)

# --- 初期値設定 ---
z_0 = 2.5 * 10**6
x1_0 = 1e-10
x2_0 = 1e-11
Omega_r0 = 0.9989
Omega_m0 = 1 - x1_0**2 - x2_0**2 - Omega_r0

# --- 数値計算設定 ---
h = 0.01
num_steps = 4000

# --- 初期値代入 ---
z = z_0
x1 = x1_0
x2 = x2_0
Omega_r = Omega_r0
Omega_m = Omega_m0

# --- 各変数の保存用リスト ---
z_list = [z]
x1_list = [x1]
x2_list = [x2]
Omega_DE_list = [x1**2 + x2**2]
Omega_r_list = [Omega_r]
Omega_m_list = [Omega_m]
w_DE_list = []
w_eff_list = []

# 微分方程式
def dx1dN(x1, x2, Omega_r):
    return x1 / 2 * (3 * x1**2 - 3 * x2**2 - 3 + Omega_r) + np.sqrt(6) / 2 * lambda_val * x2**2

def dx2dN(x1, x2, Omega_r):
    return x2 / 2 * (3 + 3 * x1**2 - 3 * x2**2 + Omega_r - np.sqrt(6) * lambda_val * x1)

def dOmega_rdN(x1, x2, Omega_r):
    return Omega_r * (Omega_r - 1 + 3 * x1**2 - 3 * x2**2)

# 数値積分（4次ルンゲクッタ法）
for _ in range(num_steps):
    # Runge-Kutta計算
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

    x1 += (k1_x1 + 2 * k2_x1 + 2 * k3_x1 + k4_x1) / 6
    x2 += (k1_x2 + 2 * k2_x2 + 2 * k3_x2 + k4_x2) / 6
    Omega_r += (k1_Omega_r + 2 * k2_Omega_r + 2 * k3_Omega_r + k4_Omega_r) / 6

    z = z / np.exp(h)

    # 計算
    Omega_DE = x1**2 + x2**2
    Omega_m = 1 - Omega_DE - Omega_r
    w_DE = (x1**2 - x2**2) / Omega_DE if Omega_DE > 1e-10 else -1
    w_eff = Omega_r / 3 + Omega_DE * w_DE

    # リストに追加
    z_list.append(z)
    x1_list.append(x1)
    x2_list.append(x2)
    Omega_r_list.append(Omega_r)
    Omega_m_list.append(Omega_m)
    Omega_DE_list.append(Omega_DE)
    w_DE_list.append(w_DE)
    w_eff_list.append(w_eff)

# --- シフト処理（z=0時点を Ω_DE ≈ 0.68 と定義） ---
index_shift = np.argmin(np.abs(np.array(Omega_DE_list) - 0.68)) # 最も1に近いインデックス
z_shift = z_list[index_shift]
shifted_z_list = np.array(z_list) / z_shift

# --- プロット ---
fig, ax = plt.subplots(figsize=(8, 6))
# 安全のため長さを揃える
plot_length = min(len(shifted_z_list), len(Omega_r_list), len(Omega_DE_list), len(Omega_m_list), len(w_eff_list), len(w_DE_list))

# プロット
ax.plot(shifted_z_list[:plot_length], Omega_r_list[:plot_length], label=r"$\Omega_r$", color='slateblue')
ax.plot(shifted_z_list[:plot_length], Omega_m_list[:plot_length], label=r"$\Omega_m$", color='orange')
ax.plot(shifted_z_list[:plot_length], Omega_DE_list[:plot_length], label=r"$\Omega_\phi$", color='red')
ax.plot(shifted_z_list[:plot_length], w_eff_list[:plot_length], label=r"$w_{eff}$", color='springgreen')
ax.plot(shifted_z_list[:plot_length], w_DE_list[:plot_length], label=r"$w_\phi$", color='dodgerblue')

ax.set_xscale("log")
ax.set_xlim(1e-1, 1e6)
ax.set_ylim(-1.1, 1.1)
ax.set_xlabel(r"$z+1$", fontsize=14)
ax.set_ylabel("Each Parameter Values", fontsize=14)
ax.tick_params(which="both", direction="in", top=True, right=True)
ax.legend()

st.pyplot(fig)

# --- 最終出力 ---
st.markdown("### z=0 における各密度パラメータ")
st.write(f"$Ω_r$ = {Omega_r_list[index_shift]:.5f}")
st.write(f"$Ω_m$ = {Omega_m_list[index_shift]:.5f}")
st.write(f"$Ω_\phi$ = {Omega_DE_list[index_shift]:.5f}")
st.write(f"合計 = {Omega_r_list[index_shift] + Omega_m_list[index_shift] + Omega_DE_list[index_shift]:.5f}")

# --- Ω_r = Ω_m となるz ---
ratio_list = np.array(Omega_r_list) / np.array(Omega_m_list)
index_equal = np.argmin(np.abs(ratio_list - 1))
z_equal = shifted_z_list[index_equal]
st.write(f"$Ω_r$ / $Ω_m$ = 1 が成立する赤方偏移 z ≈ {z_equal:.2f}")