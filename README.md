# hydro2dRP
*ChatGPT was used to organize this Readme file.*

<!-- ![Ryan-vanderHeijden/hydro2dRP](hydro2dRP_square_logo_dark.png) -->
<img src='hydro2dRP_square_logo_dark.png' width='400' height='400'>


---

# Gumbel Copula Utilities

Tools for **bivariate extreme value analysis** using the **Gumbel copula**, including joint return periods, iso-return-period contours, and Kendall risk contours.
This code is intended for **internal use** in exploratory analysis, model development, and visualization.

---

## Features

* Gumbel copula **CDF and PDF**
* **Joint return periods**

  * AND case
  * OR case
  * Conditional return period
* **Iso–return-period contours** in copula space
* **Transforms** from copula space to real values
* **Kendall risk contours** and bootstrap confidence bands
* Sampling from the Gumbel copula (Marshall–Olkin algorithm)
* Utilities for:

  * Distribution fitting for marginals (AIC-based)
  * Colored line plotting based on likelihood with missing data handling

---

## Mathematical background (brief)

Let $$( U, V \in (0,1) )$$ be marginal probability integral transforms.

The **Gumbel copula** is
$$C(u, v) = \exp\left(\left[(-\log u)^\theta + (-\log v)^\theta\right]^{1/\theta}\right), \quad \theta \ge 1$$


Common return period definitions used here:

* **AND case**:
$$T_{AND} = \frac{1}{P(U > u, V > v)} = \frac{1}{1 - u - v + C(u,v)}$$

* **OR case**:
$$T_{OR} = \frac{1}{P(U > u \cup V > v)} = \frac{1}{1 - C(u,v)}$$

* **Kendall return period**:
$$T_{K}(u,v) = \frac{1}{1−K(C(u,v))}$$

---

## Overview of return period types

| Contour type | What is fixed              | Interpretation          |
| ------------ | -------------------------- | ----------------------- |
| AND          | $$P(X>x,Y>y)$$             | simultaneous exceedance |
| OR           | $$P(X>x \text{ or } Y>y)$$ | system failure          |
| Kendall      | rank of $$C(u,v)$$         | joint extremeness       |


---
## Installation / Requirements

Clone the repo and import directly.

### Dependencies

* Python ≥ 3.9
* `numpy`
* `scipy`
* `matplotlib`

Example:

```bash
pip install numpy scipy matplotlib
```

---

## Basic usage

```python
import numpy as np
from gumbel_copula_2dRP import (
    gumbel_copula,
    return_period_OR,
    iso_rp_OR,
)
```

### Evaluate the copula

```python
u = np.linspace(0.01, 0.99, 200)
v = np.linspace(0.01, 0.99, 200)
theta = 2.0

C = gumbel_copula(u[:, None], v[None, :], theta)
```

---

### OR-type return period

```python
T_or = return_period_OR(0.95, 0.95, theta)
print(T_or)
```

---

### Iso–return-period contour (OR case)

```python
u = np.linspace(0.01, 0.99, 500)
T = 100  # years

v = iso_rp_OR(u, T, theta)
```

---

## Kendall risk contours

### Sampling

```python
from gumbel_copula_2drp import sample_gumbel

U, V = sample_gumbel(n=50_000, theta=2.5)
C_sim = gumbel_copula(U, V, theta=2.5)
C_sorted = np.sort(C_sim)
```

### Kendall level for return period T

```python
from copula import kendall_level

c_T = kendall_level(C_sorted, T=100)
```

### Kendall isoline

```python
from gumbel_copula_2drp import gumbel_kendall_isoline

u = np.linspace(0.01, 0.99, 500)
v = gumbel_kendall_isoline(u, c_T, theta=2.5)
```

---

## Plotting helpers

### Colored line with gaps preserved

```python
from gumbel_copula_2drp import colored_line
import RP_plotting as rp_plot
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

lc = rp_plot.colored_line(u, v, c=v)
ax.add_collection(lc)
ax.autoscale()

plt.show()
```

---

## Notes & limitations

* Only **bivariate** Gumbel copula is implemented
* No parameter estimation routines (for theta) are included
* Numerical issues may arise for:

  * u or v extremely close to 0 or 1
  * very large theta
* This code assumes **continuous marginals**

---

## References

* Nelsen, R. B. (2006). *An Introduction to Copulas*
* Salvadori, G., et al. (2011). *Multivariate Return Periods*
* Coles, S. (2001). *An Introduction to Statistical Modeling of Extreme Values*

---

## Contact / ownership

Internal code — contact me for questions or modifications.

---
