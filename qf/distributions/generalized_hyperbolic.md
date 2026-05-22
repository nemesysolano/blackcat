# Generalized Hyperbolic Inference #

## Random Number Generation.
The primary library is **SciPy** (scipy.stats.genhyperbolic), but to get the absolute maximum performance for generating tens of millions of synthetic paths, you should exploit its underlying mathematical structure using your C++/Cython stack.

Here is the breakdown of the available tools and how to optimize them.

### 1. The Standard: scipy.stats.genhyperbolic

SciPy natively supports the Generalized Hyperbolic (GH) distribution.

```python
from scipy.stats import genhyperbolic

# p = lambda, a and b are shape/skewness parameters derived from alpha and beta
samples = genhyperbolic.rvs(p=lambda_val, a=alpha_val, b=beta_val, loc=mu, scale=delta, size=10_000_000)

```

**The Performance Reality:**
SciPy does not rely on a slow inverse-transform method for this. Under the hood, it uses a C-optimized implementation of the **Hörmann and Leydold (2014)** algorithm to generate the necessary underlying variables. However, because genhyperbolic inherits from SciPy's generic rv_continuous class, the Python-level API overhead and input validation can create a bottleneck when generating massive Monte Carlo datasets.

### 2. The High-Performance Approach: Variance-Mean Mixture

Because you are building a heavy backtesting pipeline and are proficient in C++ and Cython, you can bypass the generic SciPy wrapper entirely. You can build a significantly faster generator by exploiting the fact that the GH distribution is a **normal variance-mean mixture**.

Instead of asking a library to generate GH numbers directly, you generate the mixing variable and standard normals separately. This is highly parallelizable and strips out the Python overhead.

**The Mathematics:**
A random variable $X \sim GH(\lambda, \alpha, \beta, \delta, \mu)$ can be constructed exactly as:

$$X = \mu + \beta W + \sqrt{W} Z$$

Where:

* $Z \sim N(0, 1)$ (Standard Normal)
* $W \sim GIG(\lambda, \delta, \gamma)$ (Generalized Inverse Gaussian distribution), where $\gamma = \sqrt{\alpha^2 - \beta^2}$

**The Cython Execution Strategy:**
To maximize throughput for your synthetic DNN training data:

1. **Generate the GIG variables:**
Use scipy.stats.geninvgauss.rvs() to generate the $W$ array. SciPy's GIG generator is heavily optimized in C, so calling this once to generate a massive, contiguous 1D NumPy array is quite fast.


2. **Generate the Normal variables:**
Use NumPy's np.random.standard_normal() to generate the $Z$ array. This execution is effectively instantaneous.


3. **Assemble the Mixture via Cython:**
Compute the final $X = μ + β W + \sqrt{W} Z$ equation. While you can do this using standard NumPy vectorization, compiling this specific arithmetic step in a Cython prange loop with nogil enabled allows you to multi-thread the calculation across all CPU cores.


By explicitly decoupling the Generalized Inverse Gaussian generation from the normal mixture assembly, you remove the object-oriented bloat and achieve bare-metal performance for your synthetic data pipeline.

### 3. The R Ecosystem (Fallback)

If you ever find the SciPy GIG generation too slow for your required scale, the absolute fastest production-grade GH generators exist in the R ecosystem—specifically the ghyp and GeneralizedHyperbolic packages. Some quantitative teams use the rpy2 Python library to bind these R packages into their data pipelines specifically for the Monte Carlo generation step, though your native Cython mixture approach usually renders this unnecessary.

---