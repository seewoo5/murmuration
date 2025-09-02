# Murmuration

Reproducing murmuration plots in literature

## Requirements

- SageMath >= 10.6

The following python libraries have to be installed inside SageMath shell:

- numpy
- matplotlib
- tqdm

```
sage --sh
pip install numpy matplotlib tqdm
exit
```

## Elliptic curves

- He-Lee-Oliver-Podznyakov, "Murmuration of Elliptic Curves", Experimental Mathematics 2024

    - Reproduce Figure 1

- Sutherland, "Letter to Rubinstein and Sarnak"

    - Reproduce figures in page 2 and 3

```
sage ellcurve.sage
```

The generated plots will be saved under the directory `plots/ellcurve`.

## Dirichlet character

- Lee-Oliver-Podznyakov, "Murmuration of Dirichlet Characters", IMRN 2025

    - Reproduce Figure 1, 2, and 3.

```
sage dirichlet.sage
```

The generated plots will be saved under the directory `plots/dirichlet`.

## Modular form

- Zubrilina, "Murmuration", Inventiones 2025

    - Reproduce Figure 2, 3, and 4.

```
sage modform.sage
```

The generated plots will be saved under the directory `plots/modform`.
