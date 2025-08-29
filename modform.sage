import numpy as np
import matplotlib.pyplot as plt


alpha = float(2 * pi)
beta = float(2 * pi)
gamma = float(12)
for p in primes(100000):
    alpha *= (1 - p - 2 * p^2 + p^4) / (p - 2 * p^2 + p^4)
    beta *= (-1 + p^2 + p^3) / (p * (-1 + p + p^2))
    gamma *= p * (1 + p) / (-1 + p + p^2)


def nu(r):
    # \nu(r) = \prod_{p | r} (1 + p^2 / (1 - p - 2 * p^2 + p^4))
    return prod((1 + p^2 / (1 - p - 2 * p^2 + p^4)) for p in prime_divisors(r))


def Mk(y, k=2):
    # Density function for weight k modular forms
    if k % 2 == 1:
        raise ValueError("k must be even")

    s = 0
    for r in range(1, int(2 * sqrt(y)) + 1):
        if k == 2:
            pass
        s += nu(r) * sqrt(4 * y - r^2) * chebyshev_U(k - 2, r / 2 / sqrt(y))
    s *= alpha * (-1)^(k/2 - 1) / (k - 1)
    s += beta * sqrt(y) / (k - 1)
    if k == 2:
        s -= gamma * y
    return s


def dyadic_func(y):
    a = (4/9) * (2^(3/2) - 1) * beta
    b = (2/3) * gamma
    c = (2/3) * alpha
    if 0 <= y <= 1/4:
        return a * sqrt(y) - b * y
    elif 1/4 < y <= 1/2:
        return a * sqrt(y) - b * y + c * pi * y^2 - c * (1 - 2 * y) * sqrt(y - 1/4) - 2 * c * y^2 * arcsin(1 / (2 * y) - 1)
    elif 1/2 < y <= 1:
        return a * sqrt(y) - b * y + 2 * c * y^2 * (arcsin(1 / y - 1) - arcsin(1 / (2 * y) - 1)) - c * (1 - 2 * y) * sqrt(y - 1/4) + 2 * c * (1 - y) * sqrt(2 * y - 1)
    else:
        raise ValueError("Invalid input")


def fig3(k=8):
    # Plot of Mk
    print(f"Figure 3: Plot of Mk for k = {k}")
    plt.subplots(figsize=(15, 2))

    y_max = 5.0
    y_div = 100
    y_pts = [i / y_div for i in range(int(y_max * y_div))]
    Mk_pts = [Mk(y, k=k) for y in y_pts]
    plt.plot(y_pts, Mk_pts, color='blue', label=rf'$M_{k}(y)$')

    plt.legend(loc='upper right')
    plt.axhline(0, xmax=y_max, color='black', linewidth=1)
    plt.savefig(f"./plots/modform/fig3_k={k}.png")
    plt.close()


def fig4():
    print("Figure 4")
    plt.subplots(figsize=(15, 2))

    y_max = 1.0
    y_div = 500
    y_pts = [i / y_div for i in range(int(y_max * y_div))]
    dyadic_pts = [dyadic_func(y) for y in y_pts]
    plt.plot(y_pts, dyadic_pts, color='blue', label=r'$\frac{2}{3} \int_1^2 u \mathcal{M}_2(y/u) \mathrm{d} u$')

    plt.legend(loc='upper right')
    plt.xticks([0, 1/4, 1/2, 1])
    plt.axhline(0, xmax=y_max, color='black', linewidth=1)
    plt.savefig(f"./plots/modform/fig4.png")
    plt.close()


if __name__ == "__main__":
    # fig3(k=8)
    # fig3(k=24)
    fig4()
