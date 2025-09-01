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
    ec_murmuration_dyadic(2^16)  # about 4 hours with a macbook
