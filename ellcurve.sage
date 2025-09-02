import polars as pl
import matplotlib.pyplot as plt
from tqdm import tqdm
from lmf import db


ec_db = db.ec_curvedata


def query_data(N1, N2, r=None):
    # Elliptic curves over Q of conductor N1 <= N <= N2
    if r is None:
        return ec_db.search({"conductor": {"$gte": N1, "$lte": N2}})
    return ec_db.search({"conductor": {"$gte": N1, "$lte": N2}, "rank": r})


def fig1():
    # Figure 1 of HLOP
    np = 1000
    pmax = 7919
    ns = list(range(1, 1001))

    print("Murmuration of elliptic curves of conductor in [7500, 10000] and rank r = 0, 1")
    plt.subplots(figsize=(12, 6))

    data_r0 = list(query_data(7500, 10000, r=0))
    data_r1 = list(query_data(7500, 10000, r=1))
    avgs_r0 = vector([0] * 1000)
    avgs_r1 = vector([0] * 1000)
    cnt_r0 = 0
    cnt_r1 = 0
    isog_labels = set()

    for ec in tqdm(data_r0):
        if ec['lmfdb_iso'] in isog_labels:
            continue
        isog_labels.add(ec['lmfdb_iso'])
        ec_sage = EllipticCurve(QQ, ec['ainvs'])
        avgs_r0 += vector(ec_sage.aplist(pmax))
        cnt_r0 += 1
    avgs_r0 /= cnt_r0
    
    for ec in tqdm(data_r1):
        if ec['lmfdb_iso'] in isog_labels:
            continue
        isog_labels.add(ec['lmfdb_iso'])
        ec_sage = EllipticCurve(QQ, ec['ainvs'])
        avgs_r1 += vector(ec_sage.aplist(pmax))
        cnt_r1 += 1
    avgs_r1 /= cnt_r1

    plt.scatter(ns, avgs_r0, color='blue', label='rank 0', s=1)
    plt.scatter(ns, avgs_r1, color='red', label='rank 1', s=1)
    plt.legend(loc='upper right')
    plt.axhline(0, xmax=1000, color='black', linewidth=1)
    plt.savefig(f"./plots/ellcurve/fig1_top_r01.png")
    plt.close()


    print("Murmuration of elliptic curves of conductor in [5000, 10000] and rank r = 0, 2")
    plt.subplots(figsize=(12, 6))

    data_r0 = list(query_data(5000, 10000, r=0))
    data_r2 = list(query_data(5000, 10000, r=2))
    avgs_r0 = vector([0] * 1000)
    avgs_r2 = vector([0] * 1000)
    cnt_r0 = 0
    cnt_r2 = 0
    isog_labels = set()

    for ec in tqdm(data_r0):
        if ec['lmfdb_iso'] in isog_labels:
            continue
        isog_labels.add(ec['lmfdb_iso'])
        ec_sage = EllipticCurve(QQ, ec['ainvs'])
        avgs_r0 += vector(ec_sage.aplist(pmax))
        cnt_r0 += 1
    avgs_r0 /= cnt_r0
    
    for ec in tqdm(data_r2):
        if ec['lmfdb_iso'] in isog_labels:
            continue
        isog_labels.add(ec['lmfdb_iso'])
        ec_sage = EllipticCurve(QQ, ec['ainvs'])
        avgs_r2 += vector(ec_sage.aplist(pmax))
        cnt_r2 += 1
    avgs_r2 /= cnt_r2

    plt.scatter(ns, avgs_r0, color='blue', label='rank 0', s=1)
    plt.scatter(ns, avgs_r2, color='green', label='rank 2', s=1)
    plt.legend(loc='upper right')
    plt.axhline(0, xmax=1000, color='black', linewidth=1)
    plt.savefig(f"./plots/ellcurve/fig1_bottom_r02.png")
    plt.close()


def ec_murmuration_dyadic(X, y_max=1.0):
    # Reproduce Sutherland's computation
    # We assume parity conjecture and compute root number as (-1)^r
    # Non-CM only
    print(f"Murmuration of elliptic curves of conductor in [{X}, {2*X})")
    plt.subplots(figsize=(24, 6))

    data = list(query_data(X, 2*X-1))
    isog_labels = set()
    np = prime_pi(int(X * y_max))
    avgs_even = vector([0] * np)
    avgs_odd = vector([0] * np)
    cnt_even = 0
    cnt_odd = 0
    for ec in tqdm(data):
        if ec['lmfdb_iso'] in isog_labels:
            continue
        if ec['cm'] != 0:  # Non-CM only
            continue
        isog_labels.add(ec['lmfdb_iso'])
        ec_sage = EllipticCurve(QQ, ec['ainvs'])
        if ec['rank'] % 2 == 0:
            avgs_even += vector(ec_sage.aplist(int(X * y_max)))
            cnt_even += 1
        else:
            avgs_odd += vector(ec_sage.aplist(int(X * y_max)))
            cnt_odd += 1
    avgs_even /= cnt_even
    avgs_odd /= cnt_odd

    ys = [p / X * y_max for p in primes(X)]

    plt.scatter(ys, avgs_even, color='blue', label=f"even", s=1)
    plt.scatter(ys, avgs_odd, color='red', label=f"odd", s=1)
    plt.legend(loc='upper right')
    plt.axhline(0, xmax=y_max, color='black', linewidth=1)
    plt.savefig(f"./plots/ellcurve/dyadic_X={X}_ymax={y_max:.2f}.png")
    plt.close()


def ec_with_local_avg(X, y_max=1.0, gammas=[1/5]):
    # With X^gamma of local averaging

    data = list(query_data(X, 2*X-1))
    max_gamma = max(gammas)
    print(f"Murmuration of elliptic curves of conductor in [{X}, {2*X}) with local averaging, gamma={gammas}")

    isog_labels = set()
    pmax = next_prime(int(X * y_max))
    pmax_more = next_prime(int(X * y_max + X^max_gamma) + 1)
    np = prime_pi(pmax)
    np_more = prime_pi(pmax_more)
    avgs = vector([0] * np_more)

    for ec in tqdm(data):
        if ec['lmfdb_iso'] in isog_labels:
            continue
        if ec['cm'] != 0:
            continue
        isog_labels.add(ec['lmfdb_iso'])
        ec_sage = EllipticCurve(QQ, ec['ainvs'])
        if ec['rank'] % 2 == 0:
            avgs += vector(ec_sage.aplist(pmax_more))
        else:
            avgs -= vector(ec_sage.aplist(pmax_more))
    avgs /= len(isog_labels)

    # Primes list and truncate to those < pmax
    primes_all = list(prime_range(pmax_more + 1))
    ap_vals = list(avgs)  # index-aligned with primes_all
    primes_core = list(prime_range(pmax + 1))  # primes <= pmax

    # Sliding window local average over [p, p + X^gamma] for each gamma in gammas
    for gamma in gammas:
        window_avgs = [0.0] * np
        window_sum = 0.0
        pmax_gamma = next_prime(int(X * y_max + X^gamma) + 1)
        primes_all = list(prime_range(pmax_gamma + 1))
        m = len(primes_all)
        max_shift = X^gamma
        right = 0
        for i, p in enumerate(primes_core):
            # Advance right pointer while prime within window
            while right < m and primes_all[right] <= p + max_shift:
                window_sum += ap_vals[right]
                right += 1
            # Remove left element (current p) after using it in average for this i, but
            # first compute average over indices [i, right-1]
            count = right - i
            if count > 0:
                window_avgs[i] = window_sum / count
            else:
                window_avgs[i] = 0.0  # fallback (should not occur)
            # Slide window: subtract current p contribution before next i
            window_sum -= ap_vals[i]

        plt.subplots(figsize=(24, 6))
        ys = [p / X for p in primes_core]

        plt.scatter(ys, ap_vals[:np], color='green', label=f"without local avg", s=1)
        plt.scatter(ys, window_avgs, color='purple', label=rf"with local avg ($\gamma$={float(gamma):.2f})", s=2.5)
        plt.legend(loc='upper right')
        plt.axhline(0, xmax=y_max, color='black', linewidth=1)
        plt.savefig(f"./plots/ellcurve/local_avg_X={X}_ymax={y_max:.2f}_gamma={float(gamma):.2f}.png")
        plt.close()


if __name__ == "__main__":
    # fig1()
    # ec_murmuration_dyadic(2^12)
    # ec_murmuration_dyadic(2^13)
    # ec_murmuration_dyadic(2^14)
    # ec_murmuration_dyadic(2^15)
    # ec_murmuration_dyadic(2^16)  # about 4 hours with a macbook
    # ec_with_local_avg(2^12, gammas=[1/5, 1/4, 1/3, 1/2])
    # ec_with_local_avg(2^13, gammas=[1/5, 1/4, 1/3, 1/2])
    # ec_with_local_avg(2^14, gammas=[1/5, 1/4, 1/3, 1/2])
    # ec_with_local_avg(2^15, gammas=[1/5, 1/4, 1/3, 1/2])
    ec_with_local_avg(2^16, gammas=[1/5, 1/4, 1/3, 1/2])
