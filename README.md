# Black Cat #

## The Price and Volume Series ##

Let $X=\{x(1),...,x(t-1), x(t),...\}$ and $V=\{v(1),...,v(t-1), v(t),...\}$ two stochastic processes where $x(t)$ and $v(t)$ are strictly positive. For convenience we denot $X$ and $V$ as price and volume series respectively.

### Difference Functions ###

Let's define some functions that will simplify further definitions based on $X$ and $V$

#### Percentage Difference $Δ_{\%}(a,b)$ ####

The percentage difference between $a$ and $b$ denoted as $Δ(a,b)$ is defined as:

$Δ_{\%}(a,b)= \frac{b-a}{|a|+|b| + 0.000009}$

#### Serial Difference $Δ_T(ξ(t))$ ####

Pretend that $Ξ = \{ξ(1),...,ξ(t-1), ξ(t),...\}$ is a stochastic process. The serial difference associated to $ξ(t)$ is denoted as $Δ_T(ξ(t))$. It is defined as:

$Δ_T(ξ(t))=Δ_{\%}(ξ(t-T), ξ(t))$

#### Serial Ratio  ####

Let  $Ξ = \{ξ(1),...,ξ(t-1), ξ(t),...\}$ is a stochastic process; the **serial ratio** for $ξ(t) \in Ξ$ is defined as

$ρ(ξ(t))= \frac{ξ(t)}{ξ(t-1)}$

## The Price-Time Angles ##

### The Closest Extreme ###

These concepts refer to finding the nearest prior occurrence of a high or low price that is **structurally higher** or **lower** than the current price at ${t}$. The search is always backward in time.

#### Closest Higher High (${h_↑(t)}$) ####
$h_↑(t) = h(t - i_{h↑}) \quad \text{where} \quad i_{h↑} = \min \{j \in \mathbb{Z}^+ \mid h(t-j) > h(t)\}$

#### Closest Lower High (${h_↓(t)}$) ####
$h_↓(t) = h(t - i_{h↓}) \quad \text{where} \quad i_{h↓} = \min \{j \in \mathbb{Z}^+ \mid h(t-j) < h(t)\}$

#### Closest Higher Low (${l_↑(t)}$) ####
$l_↑(t) = l(t - i_{l↑}) \quad \text{where} \quad i_{l↑} = \min \{j \in \mathbb{Z}^+ \mid l(t-j) > l(t)\}$

#### Closest Lower Low (${l_↓(t)}$) ####
$l_↓(t) = l(t - i_{l↓}) \quad \text{where} \quad i_{l↓} = \min \{j \in \mathbb{Z}^+ \mid l(t-j) < l(t)\}$

---

To ensure the geometry remains stable and free from the "zero-degree" or "90-degree" traps, the normalization factors are now defined using the standardized indices ($i_{h↑}, i_{h↓}, i_{l↑}, i_{l↓}$).

#### 1. Time Lookback Base $B(t)$
This factor represents the maximum temporal distance to any of the four structural pivots, ensuring all time-ratios are bounded in $[0, 1]$.
$B(t) = \max\{i_{h↑}, \space i_{h↓}, \space i_{l↑}, \space i_{l↓}\}$

#### 2. Normalized Time Vector $b(t)$
The relative temporal proximity of each structural point.
$b(t) = \left\{\frac{i_{h↑}}{B(t)}, \space \frac{i_{h↓}}{B(t)}, \space \frac{i_{l↑}}{B(t)}, \space \frac{i_{l↓}}{B(t)}\right\}$

#### 3. Price Range Base $C(t)$
This factor represents the maximum price distance to the structural levels, ensuring all price-ratios are bounded in $[0, 1]$.
$C(t) = \max\{h_↑(t)-h(t), \space h(t)-h_↓(t), \space l_↑(t)-l(t), \space l(t)-l_↓(t)\}$

#### 4. Normalized Price Vector $c(t)$
The relative price proximity to each structural point.
$c(t) = \left\{\frac{h_↑(t)-h(t)}{C(t)}, \space \frac{h(t)-h_↓(t)}{C(t)}, \space \frac{l_↑(t)-l(t)}{C(t)}, \space \frac{l(t)-l_↓(t)}{C(t)}\right\}$

By dividing the normalized time component by the normalized price component, we derive the four **Price-Time Angles** that govern the structural geometry at time $t$:

$Θ_k(t) = \arctan\left(\frac{b_k(t)}{c_k(t) + \epsilon}\right) \quad \text{for } k \in \{1, 2, 3, 4\}$

*Note: A small epsilon ($\epsilon$) is recommended in implementation to prevent division by zero if the current price is exactly at a structural level.*

## The Volume-Time Angles ##

Serial volume difference ($|Δ_1(v(t))|$) swallows a lot of noice; so we decided to try geometrical patterns (angles) instead.

#### Closest Higher Volume (${v_↑(t)}$) ####

$v_↑(t) = h(t - i_{v↑}) \quad \text{where} \quad i_{v↑} = \min \{j \in \mathbb{Z}^+ \mid v(t-j) > v(t)\}$

#### Closest Lower Volume (${v_↑(t)}$) ####

$v_↓(t) = v(t - i_{v↓}) \quad \text{where} \quad i_{v↓} = \min \{j \in \mathbb{Z}^+ \mid v(t-j) < v(t)\}$

----
Similarly to what we did for the **price-time angles**, we have to define **normalized time vector b(t)** and **normalized volume vector c(t)**.

#### 1. Time Lookback Base $B(t)$ ####

$B(t) = \max\{i_{v↑}, \space i_{v↓}\}$

#### 2. Normalized Time Vector $b(t)$ ####

$b(t) = \left\{\frac{i_{v_↑}}{B(t)}, \space \frac{i_{v_↓}}{B(t)} \right\}$

#### 3. Volume Range Base $C(t)$ ####

$C(t) = \max\{v_↑(t)-v(t), \space v(t)-v_↓(t)\}$

#### 4. Normalized Volume Vector $c(t)$ ####

$c(t) = \left\{\frac{v_↑(t)-v(t)}{C(t)}, \space \frac{v(t)-v_↓(t)}{C(t)}\right\}$

----

By dividing the normalized time component by the normalized volume component, we derive the two **Volume-Time Angles** that govern the structural geometry at time $t$:

$Φ_k(t) = \arctan\left(\frac{b_k(t)}{c_k(t) + \epsilon}\right) \quad \text{for } k \in \{1, 2\}$

## Wavelets ##

### Price-Time Wavelets ###
Consider the four price-time angles ${θ_1(t)}$, ${θ_2(t)}$, ${θ_3(t)}$ and ${θ_4(t)}$ ruling at time ${t}$. The **price-time wavelet $W(t)$** function
is a periodic non-linear function defined as

$W(t) =\frac{\sum^4_{i=1} (\cos (θ_i(t)) + \sin (θ_i(t)))}{4\sqrt{2}}$.

This wavelet function is useful to sketch the market structure at a given point in time $t$. 
Let's move further and define the family of multivariative probability 
functions $Ω(t)$ for $\{Θ_1(t), Θ_2(t), Θ_3(t), Θ_4(t)\}$. At any given time $t$ the four angles are bound to the $[0,\frac{π}{2}]$ interval, therefore
we just need to isolate the normalization constant $A$ in the following equation:

$A \int^{π/2}_{0} \int^{π/2}_{0} \int^{π/2}_{0} \int^{π/2}_{0} ω^2⋅dΘ_1⋅dΘ_2⋅dΘ_3⋅dΘ_4 = 1$, where

$ω = \frac{\sum^4_{i=1} (\cos (θ_i) + \sin (θ_i))}{4\sqrt{2}}$.

After calculating the quadruple integral and isolating $A$, we get

$A = \frac{128}{\pi^4 + 2\pi^3 + 48\pi^2}$. Moreover, our probability function 

$Ω(Θ_1, Θ_2, Θ_3, Θ_4)=\frac{128}{\pi^4 + 2\pi^3 + 48\pi^2}ω^2$

As we wanted, we just found a family of functions $Ω(t)=Ω(Θ_1(t), Θ_2(t), Θ_3(t), Θ_4(t))$. For $i \in [1...k]$ we can use model a neural network to
forecast 

$\overrightarrow Ω(t) = Ω(t)⋅Δ_T(ξ(t)) - Ω(t-1)⋅Δ_T(ξ(t-1))$ given 

$\{\overrightarrow Ω(t-k),...,\overrightarrow Ω(t-1)\}$ as input features where $k > 0$.

### Volume-Time Wavelets ###
Consider the two volume-time angles ${φ_1(t)}$ and ${φ_2(t)}$ ruling at time ${t}$. The **volume-time wavelet $V(t)$** function
is a periodic non-linear function defined as

$V(t) =\frac{\sum^2_{i=1} (\cos (φ_i(t)) + \sin (φ_i(t)))}{2\sqrt{2}}$.

Let's define another family of multivariative probability 
functions $H(t)$ for $\{φ_1(t), φ_2(t)\}$. At any given time $t$ the four angles are bound to the $[0,\frac{π}{2}]$ interval,
therefore we just need to isolate the normalization constant $A$ in the following equation:

$A \int^{π/2}_{0} \int^{π/2}_{0} h^2⋅dφ_1⋅dφ_2 = 1$, where

$h = \frac{\sum^2_{i=1} (\cos (φ_i) + \sin (φ_i))}{2\sqrt{2}}$.

After calculation the double integral and isolating $A$, we get

$A = \frac{16}{\pi^2 + 2\pi + 16}$ and the probability function we want is

$H(φ_1, φ_2)=\frac{16}{\pi^2 + 2\pi + 16}h^2$

As we wanted, we just found a family of functions $H(t)=H(φ_1(t), φ_2(t))$. For $i \in [1...k]$ we can use model a neural network to
forecast 

$\overrightarrow H(t) =  H(t)⋅Δ_T(ξ(t)) - H(t-1)⋅Δ_T(ξ(t-1))$ given 

$\{\overrightarrow H(t-k),...,\overrightarrow H(t-1)\}$ as input features where $k > 0$.

## Squared Standarized Volume ##

The **squared standarized volume** $V(t)$ is required to calculate _liquidity cap_ in backtesting simulation.

$\hat V(t) = [\frac{\hat v(t) - v(t)} {\hat v(t) + v(t)}]^2 $ where

$\hat v(t) = \frac{\sum^{k-1}_0 v(t-k)} {k}$ and

$σ_v(t) = \sqrt {\frac{\sum^{k-1}_0  (\hat v(t)-v(t-k))^2}{k-1}}$

## Time invariant features ##

These are features which do not depend on time and enhance feature sets used to train our models.

### Scaled Market Cap ###

The **scaled market cap** is defined as

$\hat K = \begin{cases}
    \frac{\log_{10}(\text{K})}{13},\:\textit{for stocks} \\
    0,\:\textit{otherwise}
\end{cases}$

where $K$ is the market cap for stocks.

### Scaled Beta ###

The **scaled beta** is defined as 

$\hat B = \begin{cases}
    \log(b+1),\:\textit{for stocks} \\ \\
    0,\:\textit{otherwise}
\end{cases}$

where $b$ is stock beta.


# References #
[The deep learning book](https://www.deeplearningbook.org)
[Fractional Derivatives](https://www.sciencedirect.com/science/article/pii/S0377042714000065)