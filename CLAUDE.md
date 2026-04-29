# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Is

A Bayesian probabilistic modeling playground exploring the [Pangolin](https://github.com/justindomke/pangolin) library. Each `.py` and `.ipynb` file is a standalone example or experiment — there is no shared library or application structure.

## Environment Setup

This project uses [Pixi](https://pixi.sh) for dependency management (conda-forge based). Python 3.14 is required.

```bash
pixi install        # install dependencies
pixi shell          # activate environment
```

## Running Examples

```bash
# Plain Python scripts
python eight_schools.py
python framingham.py

# Marimo reactive notebooks (interactive)
marimo edit sleepstudy.py
marimo edit model_selection.py

# Jupyter notebook
jupyter notebook sleepstudy.ipynb
```

## Key Libraries

| Library | Role |
|---------|------|
| `pangolin` (as `pg`) | Core probabilistic programming DSL |
| `pangolin.interface` (as `pi`) | Distribution constructors (`normal`, `beta`, `bernoulli`, etc.) |
| `pangolin.blackjax` | MCMC inference via BlackJax (`sample_arviz` for posterior sampling) |
| `bambi` | High-level Bayesian regression with formula syntax |
| `arviz` (as `az`) | Posterior diagnostics and visualization |
| `jax` | Underlying numerical backend (JAX 0.9) |
| `marimo` | Reactive notebook format (`.py` files that run as notebooks) |

## Pangolin Patterns

Models follow this structure:

```python
import pangolin as pg
import pangolin.interface as pi

# Define priors
alpha = pi.normal(0, 10)
beta = pi.normal(0, 1)

# Define likelihood (observed nodes)
y = pi.normal(alpha + beta * x, 1)

# Sample posterior
idata = pg.blackjax.sample_arviz(
    [alpha, beta],   # latent variables to sample
    [y],             # observed variables
    [y_obs],         # observed values
    num_samples=1000,
    num_warmup=1000,
)
```

Hierarchical models use `pi.vmap` for vectorized random effects over groups.
