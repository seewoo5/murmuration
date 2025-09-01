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
    print(f"Murmuration of elliptic curves of conductor in ({X}, {2*X}]")
    plt.subplots(figsize=(24, 6))

    data = list(query_data(X+1, 2*X))
    isog_labels = set()
    np = prime_pi(int(X * y_max))
    avgs_even = vector([0] * np)
    avgs_odd = vector([0] * np)
    cnt_even = 0
    cnt_odd = 0
    for ec in tqdm(data):
        if ec['lmfdb_iso'] in isog_labels:
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


if __name__ == "__main__":
    fig1()
    # ec_murmuration_dyadic(2^16)  # about 4 hours with a macbook
