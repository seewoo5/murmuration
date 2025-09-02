import numpy as np
import matplotlib.pyplot as plt
from bisect import bisect_left, bisect_right
from scipy.integrate import quad
from tqdm import tqdm


x = SR.var('x')


# Complex characters, use Lemma 2.6 to speed-up calculation.
def P(y, X=2^10, c=2):
    conductors = prime_range(X, int(c * X) + 1)
    P_even = 0.0
    P_odd = 0.0
    p = next_prime(int(y * X))
    for N in conductors:
        if N != p:
            P_even += (N - 1) / N * cos(2 * pi * p / N) + 1 / N
            P_odd += - (N - 1) / N * sin(2 * pi * p / N)
    P_even = P_even * float(log(X)) / X
    P_odd = P_odd * float(log(X)) / X
    return P_even, P_odd


def P_tilde(y, X=2000, delta=0.51):
    conductors = prime_range(X, X + int(X^delta) + 1)
    P_even = 0.0
    P_odd = 0.0
    p = next_prime(int(y * X))
    for N in conductors:
        if N != p:
            P_even += (N - 1) / N * cos(2 * pi * p / N) + 1 / N
            P_odd += - (N - 1) / N * sin(2 * pi * p / N)
    P_even = P_even * float(log(X)) / (X^delta)
    P_odd = P_odd * float(log(X)) / (X^delta)
    return P_even, P_odd


# Real characters
phi_cache = {}

def local_average_plots(func, X, delta, y_max, y_div):
    # If `func(p)` is a function defined on primes p
    # we compute local average of `func(p)` over the primes p
    # in the interval [yX, yX + X^delta].
    ps = prime_range(0, y_max * X + X ^ delta + 1)
    func_vals_partial_sum = []
    func_smooth_vals_partial_sum = []
    psum = 0
    psum_smooth = 0
    for p in tqdm(ps):
        fp, fp_smooth = func(p)
        psum += fp
        psum_smooth += fp_smooth
        func_vals_partial_sum.append(psum)
        func_smooth_vals_partial_sum.append(psum_smooth)
    ys = [i / y_div for i in range(1, int(y_max * y_div))]
    local_avg_plots = []
    local_smooth_avg_plots = []
    for y in ys:
        # Find the interval [yX, yX + X^delta]
        left = bisect_left(ps, y * X)
        right = bisect_right(ps, y * X + X ** delta)
        local_sum = func_vals_partial_sum[right - 1] - func_vals_partial_sum[left]
        local_sum_smooth = func_smooth_vals_partial_sum[right - 1] - func_smooth_vals_partial_sum[left]
        local_avg_plots.append(local_sum * float(log(X)) / (X ^ (delta + 1)))
        local_smooth_avg_plots.append(local_sum_smooth * float(log(X)) / (X ^ (delta + 1)))
    return local_avg_plots, local_smooth_avg_plots


def P_kronecker(p, X, phi, start=1.0, end=2.0):
    # Equation (1.5) when \Phi(x) = characteristic function of [start, end]
    # or smooth function supported on [start, end]
    P = 0.0
    P_smooth = 0.0
    d_start = int(start * X)
    if d_start % 2 == 0:
        d_start += 1
    d_end = int(end * X) + 1

    for d in srange(d_start, d_end, 2):  # over odd squarefree d's
        if not d.is_squarefree():
            continue
        kr = kronecker(8 * d, p)
        P += kr
        if kr == 1:
            P_smooth += phi_cache[(phi, d / X)]
        elif kr == -1:
            P_smooth -= phi_cache[(phi, d / X)]
    P *= float(sqrt(p))
    P_smooth *= float(sqrt(p))
    return P, P_smooth


def phi_tilde(phi_supp_start, phi_supp_end, phi):
    return lambda xi: quad(lambda x: (np.cos(2 * pi * xi * x) + np.sin(2 * pi * xi * x)) * phi(x), phi_supp_start, phi_supp_end)[0]


def phi_tilde_char(phi_supp_start, phi_supp_end):
    return lambda xi: (np.sin(2 * pi * xi * phi_supp_end - pi / 4) - np.sin(2 * pi * xi * phi_supp_start - pi / 4)) / (sqrt(2) * pi * xi)


def M_density(y, phi_supp_start, phi_supp_end, phi, delta=2/3):
    N = 20
    M_func = 0
    for a in srange(1, N+1, 2):
        if not a.is_squarefree():
            continue
        s = 0
        for m in srange(1, N+1):
            s += (-1)^m * cos(pi * m^2 * x / (a^2 * y) - pi / 4)
        M_func += moebius(a) * s / a^2
    M_func *= 1/sqrt(2)
    M_density = numerical_integral(M_func * phi, phi_supp_start, phi_supp_end)[0]
    return M_density


def M_density_char(y, phi_supp_start, phi_supp_end):
    N = 20
    M_sum = 0
    for a in srange(1, N + 1, 2):
        s = 0
        for m in range(1, N + 1):
            s += (-1)^m * phi_tilde_char(phi_supp_start, phi_supp_end)(m^2 / (2 * a^2 * y))
        M_sum += moebius(a) * s / a^2
    M_sum *= 1/2
    return M_sum


def fig1_top(X=2^10, c=2):
    print("Figure 1 (Top)")
    plt.subplots(figsize=(12, 6))

    y_max = 10.0
    y_div = 200
    y_pts = [i / y_div for i in range(int(y_max * y_div))]
    P_pts = [P(y, X=X, c=c) for y in y_pts]

    P_density_even_func = lambda y: quad(lambda x: np.cos(2 * np.pi * y / x), 1, c)[0]
    P_density_odd_func = lambda y: -quad(lambda x: np.sin(2 * np.pi * y / x), 1, c)[0]
    plt.plot(y_pts, [P_density_even_func(y) for y in y_pts], color='green', label=rf'$\lim_{{X \to \infty}} P_{{+}}(y, X, {c})$')
    plt.plot(y_pts, [P_density_odd_func(y) for y in y_pts], color='orange', label=rf'$\lim_{{X \to \infty}} P_{{-}}(y, X, {c})$')

    plt.scatter(y_pts, [p[0] for p in P_pts], color='blue', label=rf'$P_{{+}}(y, {X}, {c})$', s=1)
    plt.scatter(y_pts, [p[1] for p in P_pts], color='red', label=rf'$P_{{-}}(y, {X}, {c})$', s=1)

    plt.legend(loc='upper right')
    plt.axhline(0, xmax=y_max, color='black', linewidth=1)
    plt.savefig(f"./plots/dirichlet/fig1_top_X={X}_c={float(c)}.png")
    plt.close()


def fig1_bottom(X=2002, delta=0.51):
    print("Figure 1 (Bottom)")
    plt.subplots(figsize=(12, 6))

    y_max = 2.0
    y_div = 200
    y_pts = [i / y_div for i in range(int(y_max * y_div))]
    P_tilde_pts = [P_tilde(y, X=X, delta=delta) for y in y_pts]

    P_tilde_density_even_func = lambda y: np.cos(2 * pi * y)
    P_tilde_density_odd_func = lambda y: -np.sin(2 * pi * y)
    plt.plot(y_pts, [P_tilde_density_even_func(y) for y in y_pts], color='green', label=rf'$\lim_{{X \to \infty}} \widetilde{{P}}_{{+}}(y, X, {delta:.2f})$')
    plt.plot(y_pts, [P_tilde_density_odd_func(y) for y in y_pts], color='orange', label=rf'$\lim_{{X \to \infty}} \widetilde{{P}}_{{-}}(y, X, {delta:.2f})$')

    plt.scatter(y_pts, [p[0] for p in P_tilde_pts], color='blue', label=rf'$\widetilde{{P}}_{{+}}(y, {X}, {delta:.2f})$', s=1)
    plt.scatter(y_pts, [p[1] for p in P_tilde_pts], color='red', label=rf'$\widetilde{{P}}_{{-}}(y, {X}, {delta:.2f})$', s=1)

    plt.legend(loc='upper right')
    plt.axhline(0, xmax=y_max, color='black', linewidth=1)
    plt.savefig(f"./plots/dirichlet/fig1_bottom_X={X}_delta={float(delta):.2f}.png")
    plt.close()


def fig23(X=2^16, delta=2/3):
    # Plot Figure 2 and 3 at once
    print("Figure 2")
    plt.subplots(figsize=(12, 6))

    phi_plus = exp(-1 / (1 - 4 * (x - 1.5)^2))
    start_plus = 1.0
    end_plus = 2.0
    phi_minus = exp(-1 / (1 - 4 * (x + 1.5)^2))
    start_minus = -2.0
    end_minus = -1.0
    y_max = 2.0
    y_div = 100
    y_pts = [i / y_div for i in range(1, int(y_max * y_div))]

    # cache phi values
    d_start_plus = int(start_plus * X)
    d_end_plus = int(end_plus * X)
    d_start_minus = int(start_minus * X)
    d_end_minus = int(end_minus * X)

    for d in tqdm(srange(d_start_plus, d_end_plus + 1), desc="Caching phi+ values"):
        if d == d_start_plus or d == d_end_plus:
            phi_cache[(phi_plus, d / X)] = 0
        else:
            phi_cache[(phi_plus, d / X)] = phi_plus.subs(x = d / X)
    for d in tqdm(srange(d_start_minus, d_end_minus + 1), desc="Caching phi- values"):
        if d == d_start_minus or d == d_end_minus:
            phi_cache[(phi_minus, d / X)] = 0
        else:
            phi_cache[(phi_minus, d / X)] = phi_minus.subs(x = d / X)

    P_kronecker_plus_pts, P_kronecker_smooth_plus_pts = local_average_plots(
        lambda p: P_kronecker(p, X=X, phi=phi_plus, start=start_plus, end=end_plus), X=X, delta=delta, y_max=y_max, y_div=y_div
    )
    P_kronecker_minus_pts, P_kronecker_smooth_minus_pts = local_average_plots(
        lambda p: P_kronecker(p, X=X, phi=phi_minus, start=start_minus, end=end_minus), X=X, delta=delta, y_max=y_max, y_div=y_div
    )

    P_kronecker_smooth_density_plus_pts = []
    P_kronecker_smooth_density_minus_pts = []
    for y in tqdm(y_pts, desc="M_density computation"):
        P_kronecker_smooth_density_plus_pts.append(M_density(y, start_plus, end_plus, phi_plus))
        P_kronecker_smooth_density_minus_pts.append(M_density(y, start_minus, end_minus, phi_minus))

    plt.plot(y_pts, P_kronecker_smooth_density_plus_pts, color='green', label=rf'$\lim_{{X \to \infty}} M_{{\Phi_+}}(y, X, {delta})$')
    plt.plot(y_pts, P_kronecker_smooth_density_minus_pts, color='orange', label=rf'$\lim_{{X \to \infty}} M_{{\Phi_-}}(y, X, {delta})$')

    plt.scatter(y_pts, P_kronecker_smooth_plus_pts, color='blue', label=rf'$M_{{\Phi_+}}(y, {X})$', s=1)
    plt.scatter(y_pts, P_kronecker_smooth_minus_pts, color='red', label=rf'$M_{{\Phi_-}}(y, {X})$', s=1)

    plt.legend(loc='upper right')
    plt.axhline(0, xmax=y_max, color='black', linewidth=1)
    plt.savefig(f"./plots/dirichlet/fig2_X={X}_delta={float(delta):.2f}.png")
    plt.close()

    print("Figure 3")
    plt.subplots(figsize=(12, 6))

    plt.plot(y_pts, [M_density_char(y, start_plus, end_plus) for y in y_pts], color='green', label=rf'$\lim_{{X \to \infty}} M_{{\Phi_+}}(y, X, {delta})$')
    plt.plot(y_pts, [M_density_char(y, start_minus, end_minus) for y in y_pts], color='orange', label=rf'$\lim_{{X \to \infty}} M_{{\Phi_-}}(y, X, {delta})$')

    plt.scatter(y_pts, P_kronecker_plus_pts, color='blue', label=rf'$M_{{\Phi_+}}(y, {X}, {delta})$', s=1)
    plt.scatter(y_pts, P_kronecker_minus_pts, color='red', label=rf'$M_{{\Phi_-}}(y, {X}, {delta})$', s=1)

    plt.legend(loc='upper right')
    plt.axhline(0, xmax=y_max, color='black', linewidth=1)
    plt.savefig(f"./plots/dirichlet/fig3_X={X}_delta={float(delta):.2f}.png")
    plt.close()


if __name__ == "__main__":
    fig1_top()
    fig1_bottom()
    fig23(X=2^16)  # Original plot uses X = 2^19, but it may take more than a day with current implementation (on a macbook)
