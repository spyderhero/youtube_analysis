import numpy as np
import matplotlib.pyplot as plt

# 定数
sqrt6 = np.sqrt(6)

# 関数定義
def f1(lmbda):
    inner = -15 * lmbda**2 + 64
    return -0.5 + 0.5 / lmbda * np.sqrt(inner) 

def f2(lmbda):
    inner = -15 * lmbda**2 + 64
    return -0.5 - 0.5 / lmbda * np.sqrt(inner) 

# プロット範囲
lmbda_vals = np.linspace(-20, 20, 1000)
f1_vals = np.array([f1(l) for l in lmbda_vals])
f2_vals = np.array([f2(l) for l in lmbda_vals])

plt.figure(figsize=(10, 6))
plt.plot(lmbda_vals, f1_vals, label='$g_1$(λ)', color='blue')
plt.plot(lmbda_vals, f2_vals, label='$g_2$(λ)', color='red')
plt.axhline(0, color='black', linestyle='--', label='y = 0')
plt.xlabel('λ')
plt.ylabel('y')
plt.legend()

plt.show()
