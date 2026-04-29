# Black Cat #

## The Price and Volume Series ##

Let $X=\{x(1),...,x(t-1), x(t),...\}$ and $V=\{v(1),...,v(t-1), v(t),...\}$ two stochastic processes where $x(t)$ and $v(t)$ are strictly positive. For convenience we denote $X$ and $V$ as price and volume series respectively.

### Difference Functions ###

Let's define some functions that will simplify further definitions based on $X$ and $V$

#### Percentage Difference $Δ_{\%}(a,b)$ ####

The percentage difference between $a$ and $b$ denoted as $Δ(a,b)$ is defined as:

$Δ_{\%}(a,b)= |b - a|·\frac{b-a}{(|a|+|b|)^2 + 0.000009}$

#### Serial Difference $Δ_T(ξ(t))$ ####

Pretend that $Ξ = \{ξ(1),...,ξ(t-1), ξ(t),...\}$ is a stochastic process. The serial difference associated to $ξ(t)$ is denoted as $Δ_T(ξ(t))$. It is defined as:

$Δ_T(ξ(t))=Δ_{\%}(ξ(t-T), ξ(t))$

#### Serial Ratio  ####

Let  $Ξ = \{ξ(1),...,ξ(t-1), ξ(t),...\}$ is a stochastic process; the **serial ratio** for $ξ(t) \in Ξ$ is defined as

$ρ(ξ(t))= \frac{ξ(t)}{ξ(t-1)}$

#### Log Difference ####

The **logarithmic difference** (log difference) between $a$ and $b$ denoted as $l(a,b)$ is defined as:

$l(a,b) = \ln(\frac{b}{a})$.

On the same lines, the **logarithmic serial difference** (log-serial difference or log return) $L(t)$ is

$L(t) = l(ξ(t),ξ(t-1))$

## The Price-Time Angles ##

### The Closest Extreme ###

These concepts refer to finding the nearest prior occurrence of a high or low price that is **structurally higher** or **lower** than the current price at ${t}$. The search is always backward in time.

#### Closest Higher High (${h_↑(t)}$) ####
$h_↑(t) = h(t - i_{h_↑}) \quad \text{where} \quad i_{h_↑} = \min \{j \in \mathbb{Z}^+ \mid h(t-j) > h(t)\}$

#### Closest Lower High (${h_↓(t)}$) ####
$h_↓(t) = h(t - i_{h_↓}) \quad \text{where} \quad i_{h_↓} = \min \{j \in \mathbb{Z}^+ \mid h(t-j) < h(t)\}$

#### Closest Higher Low (${l_↑(t)}$) ####
$l_↑(t) = l(t - i_{l_↑}) \quad \text{where} \quad i_{l_↑} = \min \{j \in \mathbb{Z}^+ \mid l(t-j) > l(t)\}$

#### Closest Lower Low (${l_↓(t)}$) ####
$l_↓(t) = l(t - i_{l_↓}) \quad \text{where} \quad i_{l_↓} = \min \{j \in \mathbb{Z}^+ \mid l(t-j) < l(t)\}$

---

To ensure the geometry remains stable and free from the "zero-degree" or "90-degree" traps, the normalization factors are now defined using the standardized indices ($i_{h_↑}, i_{h_↓}, i_{l_↑}, i_{l_↓}$).

#### 1. Time Lookback Base $B(t)$
This factor represents the maximum temporal distance to any of the four structural pivots, ensuring all time-ratios are bounded in $[0, 1]$.
$B(t) = \max\{i_{h_↑}, \space i_{h_↓}, \space i_{l_↑}, \space i_{l_↓}\}$

#### 2. Normalized Time Vector $b(t)$
The relative temporal proximity of each structural point.
$b(t) = \left\{\frac{i_{h_↑}}{B(t)}, \space \frac{i_{h_↓}}{B(t)}, \space \frac{i_{l_↑}}{B(t)}, \space \frac{i_{l_↓}}{B(t)}\right\}$

#### 3. Price Range Base $C(t)$
This factor represents the maximum price distance to the structural levels, ensuring all price-ratios are bounded in $[0, 1]$.
$C(t) = \max\{|h_↑(t)-h(t)|, \space |h_↓(t)-h(t)|, \space |l_↑(t)-l(t)|, \space |l_↓(t)-l(t)|\}$

#### 4. Normalized Price Vector $c(t)$
The relative price proximity to each structural point.
$c(t) = \left\{\frac{h_↑(t)-h(t)}{C(t)}, \space \frac{h_↓(t)-h(t)}{C(t)}, \space \frac{l_↑(t)-l(t)}{C(t)}, \space \frac{l_↓(t)-l(t)}{C(t)}\right\}$

By dividing the normalized time component by the normalized price component, we derive the four **Price-Time Angles** that govern the structural geometry at time $t$:

$Θ_k(t) = \arctan\left(\frac{b_k(t)}{c_k(t) + \epsilon}\right) \quad \text{for } k \in \{1, 2, 3, 4\}$

*Note: A small epsilon ($\epsilon$) is recommended in implementation to prevent division by zero if the current price is exactly at a structural level.*

## The Volume-Time Angles ##

Serial volume difference ($|Δ_1(v(t))|$) swallows a lot of noise; so we decided to try geometrical patterns (angles) instead.

#### Closest Higher Volume (${v_↑(t)}$) ####

$v_↑(t) = v(t - i_{v↑}) \quad \text{where} \quad i_{v_↑} = \min \{j \in \mathbb{Z}^+ \mid v(t-j) > v(t)\}$

#### Closest Lower Volume (${v_↓(t)}$) ####

$v_↓(t) = v(t - i_{v↓}) \quad \text{where} \quad i_{v_↓} = \min \{j \in \mathbb{Z}^+ \mid v(t-j) < v(t)\}$

----
Similarly to what we did for the **price-time angles**, we have to define **normalized time vector b(t)** and **normalized volume vector c(t)**.

#### 1. Time Lookback Base $B(t)$ ####

$B(t) = \max\{i_{v_↑}, \space i_{v_↓}\}$

#### 2. Normalized Time Vector $b(t)$ ####

$b(t) = \left\{\frac{i_{v_↑}}{B(t)}, \space \frac{i_{v_↓}}{B(t)} \right\}$

#### 3. Volume Range Base $C(t)$ ####

$C(t) = \max\{v_↑(t)-v(t), \space v(t)-v_↓(t)\}$

#### 4. Normalized Volume Vector $c(t)$ ####

$c(t) = \left\{\frac{v_↑(t)-v(t)}{C(t)}, \space \frac{v_↓(t)-v(t)}{C(t)}\right\}$

----

By dividing the normalized time component by the normalized volume component, we derive the two **Volume-Time Angles** that govern the structural geometry at time $t$:

$Φ_k(t) = \arctan\left(\frac{b_k(t)}{c_k(t) + \epsilon}\right) \quad \text{for } k \in \{1, 2\}$

## Wavelets ##

### Price-Time Wavelets ###
Consider the four price-time angles ${θ_1(t)}$, ${θ_2(t)}$, ${θ_3(t)}$ and ${θ_4(t)}$ ruling at time ${t}$. The **price-time wavelet $W(t)$** function
is a periodic non-linear function defined as

$W(t) =\frac{\sum^4_{i=1} (\cos (θ_i(t)) + \sin (θ_i(t)))}{4\sqrt{2}}$.

This wavelet function is useful to sketch the market structure at a given point in time $t$. 
Let's move further and define the family of multivariate probability 
functions $Ω(t)$ for $\{Θ_1(t), Θ_2(t), Θ_3(t), Θ_4(t)\}$. 

The $Θ_1$, $Θ_3$ and $Θ_2$, $Θ_4$ angles are bound to the $[0,\frac{π}{2}]$ and $[-\frac{π}{2}, 0]$ intervals respectively, therefore
we just need to isolate the normalization constant $A$ in the following equation:

$A_Ω \int^{π/2}_{0} \int^{π/2}_{0} \int^{π/2}_{0} \int^{π/2}_{0} ω^2⋅dΘ_1⋅dΘ_2⋅dΘ_3⋅dΘ_4 = 1$, where

$ω = \frac{\sum^4_{i=1} (\cos (θ_i) + \sin (θ_i))}{4\sqrt{2}}$.

After calculating the quadruple integral and isolating $A$, we get

$A_Ω = \frac{128}{\pi^4 + 2\pi^3 + 48\pi^2}$. Moreover, our probability function 

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
functions $H(t)$ for $\{φ_1(t), φ_2(t)\}$. The $φ_1$ and $φ_2$ angles are bound to  $[0,\frac{π}{2}]$ and $[-\frac{π}{2}, 0]$ intervals respectively,
therefore we just need to isolate the normalization constant $A$ in the following equation:

$A_H \int^{π/2}_{0} \int^{π/2}_{0} h^2⋅dφ_1⋅dφ_2 = 1$, where

$h = \frac{\sum^2_{i=1} (\cos (φ_i) + \sin (φ_i))}{2\sqrt{2}}$.

After calculating the double integral and isolating $A$, we get

$A_H = \frac{16}{\pi^2 + 2\pi + 16}$ and the probability function we want is

$H(φ_1, φ_2)=\frac{16}{\pi^2 + 2\pi + 16}h^2$

As we wanted, we just found a family of functions $H(t)=H(φ_1(t), φ_2(t))$. For $i \in [1...k]$ we can use model a neural network to
forecast 

$\overrightarrow H(t) =  H(t)⋅Δ_T(ξ(t)) - H(t-1)⋅Δ_T(ξ(t-1))$ given 

$\{\overrightarrow H(t-k),...,\overrightarrow H(t-1)\}$ as input features where $k > 0$.

## Fractional Features ##

Fractional features are numeric features that can be modeled using fractional derivatives:

$D^s_tZ(t)= \frac{1}{Γ(1-s)}\frac{d}{dt}\int^{t}_0\frac{Z(τ)}{(t-τ)^s}dτ$

We are not expecting that $D^s_tZ(t)$ boils down to an analytical solution because we will carelessly plug any $Z(t)$; therefore
we need to discretize (quantize) $D^s_tZ(t)$  using historical values of $Z(t)$:

$D^s_tZ(t) ≈ \sum^{N-1}_{k=0} w_k Z(t-k)$, where

$w_k = \begin{cases}
    1\text{ } \text{when } k = 0  \\\\
    w_{k-1}\frac{k-1-s}{k} \text{ when } k > 0
\end{cases}$


### Log Acceleration ###

Consider the series of log returns $\{L(t-(N-1)),...,L(t)\}$. Roughly speaking, if $L(t)= \ln(\frac{ξ(t)}{ξ(t-1)})$ 
gauges momentum at time $t$, we can approximate the acceleration at time $t$ as $L(t) - L(t-1)$. Worth to notice that **momentum signum** and
**acceleration signum** represent different ideas; the former indicate direction of motion whilst the later represent whether the 
particle is accelerating or decelerating. Let $Λ_L(t) = L(t) - L(t-1)$ denote the **log acceleration** of a particle at time $t$ which can be 
estimated with fractional derivative as:

$Λ_L(t) ≈ \hat Λ_L(t) = \hat D^s_tL(t) = \sum^{N}_{k=1} w_k L(t-k)$ where $0 < s < 1$.

Once we calculate the $\hat Λ_L(t)$' we can also estimate $s$ and subsequently the weights for the discrete integration formula required to reverse
engineer $\hat L(t)$ momentum from acceleration:

$w_k(-s) = \begin{cases}
    1\text{ } \text{when } k = 1  \\\\
    w_{k-1}\frac{k-1+s}{k} \text{ when } k > 1
\end{cases}$

Now we are in position to to estimate $\hat L(t)$ as

$ \hat L(t) = \sum^{N}_{k=1} |w_k(-s)|Λ_L(t-k)$; where $w_k(-s)$ are the integral weights (where s is replaced by −s in the recursive step).

Enter into market according following depending all three sign match as illustrated in the table below:

Signal Type|$L(t)$|$\hat L(t)$|$\hat Λ_L$|$Λ_L(t)$|Potential State (V)|Physical Interpretation|
-----------|------|------------|--------|------|-------------------|-----------------------|
Strong Bullish|+|+|+|+|Tunneling|All signs align. Kinetic pressure exceeds the potential barrier $V$. High probability breakout.
Strong Bearish|−|−|−|−|Tunneling|Momentum and memory are in phase. The price particle is escaping the well to the downside.
Mean Reversion (Long)|−|−|+|+|Hard Boundary|Price is at the bottom of the well. Potential $V$ is high, forcing a reversal in $\hat Λ$.
Mean Reversion (Short)|+|+|−|−|Hard Boundary|Price is at the ceiling. The wavelet force $ω(t)$ is overpowering the upward inertia.
Incoherent Noise|±|±|∓|∓|Damping|The fractional memory ($\hat L$) and local acceleration ($Λ$) are out of phase. No Trade.
Fake-Out Warning|+|+|+|−|Decoherence|Memory predicts a jump ($\hat Λ$), but the classical price ($Λ$) is stalling. The particle is trapped.

---

The formula for $\hat L(t)$  is the discrete version of Riemann-Liouville Fractional Integral:

$_aI^s_tf(t) = \frac{1}{Γ(s)} \int^t_a(t-τ)^{s-1}f(τ)dτ$

### Wavelet Acceleration $Λ_W(t)$ ###
Consider again the series of log returns $\{L(t-(N-1)),...,L(t)\}$. Now we'll bind wavelets to log returns using this formula:

$Λ_W(t) = \operatorname sign(L(t))⋅(\frac{Ω(t)+H(t) - (Ω(t-1)+H(t-1))}{2})$ where $Ω(t)$ and $H(t)$ where previously defined in _Wavelets_ section. 

An alternative method to estimate $Λ(t)$ is to feed a neural network targeting $Λ(t)$ with last $\{Δ_W(t-(N-1)),...,Δ_W(t-1)\}$ wavelet accelerations. However, this neural model is a predictor unlike **log acceleration**'s which is a regressor. 

## Standard Schrödinger Equation ##

The time-dependent Schrödinger equation (TDSE) is the fundamental equation of motion in _non-relativistic quantum mechanics_. It describes how the quantum state of a physical system —represented by the wave function $Ψ(x,t)$— evolves over time.

$iℏ \frac{∂Ψ(x,t)}{∂t} = \frac {-ℏ^2}{2m} \frac{∂^2Ψ(x,t)}{∂x^2} + V(x,t)Ψ(x,t)$.

Each term in the equation represents a specific physical quantity or mathematical operation.

### 1. Energy Operator (Left Side) ###

$iℏ \frac{∂Ψ(x,t)}{∂t}$ 

This term represents the _total energy_ of the system acting on the wave function   .

- $i$: The imaginary unit ($\sqrt {-1}$), which is essential for describing the phase and wave-like nature of matter.
- $ℏ$: The reduced Plank's constant. This is the fundamental scale of quantum world, approximately $1.054×10^{-34} J$.
- $\frac{∂}{∂t}$: The partial derivative repecto to time. This shows that the equation is first order in time, meaning that if we know the state of the system now, we can determine its state at any future time.

### 2. Kinetic Energy Term ###

$\frac {-ℏ^2}{2m} \frac{∂^2Ψ(x,t)}{∂x^2}$

This term represents the _kinetic energy_ of the particle.

- $m$: The mass of the particle.
- $\frac{∂^2}{∂x^2}$: The second partial derivative (the Laplacian in higher dimensions). In physics, this represents the curvature of the wave function.

### 3. Kinetic Energy Term ###

$V(x,t)Ψ(x,t)$

This represents the _potential energy_ environment in which the particle moves. 

- $V(x,t)$: This function describes the external forces acting on the particle (e.g., gravity, electric fields or a physical container.
- If $V$ changes over time, then the energy of the system is not converved. If $V$ depends only on position $V(x)$, the equation can often be simplified using separation of variables

### 4. The Wave Function ###

$Ψ(x,t)$

Known as the _probability amplitude_ environment in which the environment moves. 

- While $Ψ$ itself is a complex value function and not directly observable, its absolute square ${|Ψ(x,t)|}^2$ provides the _probability density_ of finding the particle at position $x$ at time $t$.

### Black Cat's Box Model $V(t)$ ###

Suppose that $\{x_1,...,x_n,...\}$ is a stream of stock prices. Remeber that ${h_↑(t)}$ and ${l_↓(t)}$ are respectively the closest higher high and lowest lower low to the current price at time $t$ (namely $x_t$).

Given this context, let's define the black cat's box model:

$V(x,t) = \begin{cases}
    Δ_{\%}(x_t,h_↑(t)) \text{ when } \operatorname sign(L(t)) + \operatorname sign(ρ(t)) = 2  \\\\
    0 \text{ when } B(t) = 0  \\\\
    Δ_{\%}(l_↓(t),x_t) \text{ when } \operatorname sign(L(t)) + \operatorname sign(ρ(t)) = -2 
\end{cases}$,

where $ρ(t) = Δ_{\%}(c(t)-o(t), h(t)-l(t))$ (candle's **body-to-range ratio**).

We notice that this box model is fully time dependent, so let's drop $x$ to have a simpler model:

$V(t) = \begin{cases}
    Δ_{\%}(x_t,h_↑(t)) \text{ when } \operatorname sign(L(t)) + \operatorname sign(ρ(t)) = 2  \\\\
    0 \text{ when } B(t) = 0  \\\\
    Δ_{\%}(l_↓(t),x_t) \text{ when } \operatorname sign(L(t)) + \operatorname sign(ρ(t)) = -2 
\end{cases}$.

---

Next logical step is to replace $V(x,t)$ with $V(t)$ in the time dependend Schrödinger equation:

$iℏ \frac{∂Ψ(x,t)}{∂t} = \frac {-ℏ^2}{2m} \frac{∂^2Ψ(x,t)}{∂x^2} + V(t)Ψ(x,t)$.

Moreover, let's define $Ψ(x, t) = Ψ(t) = \sqrt {\frac {Ω(t)+H(t)}{2}} e^{iL(t)}$ (where $Ω(t)$ ad $H(t)$ are previously defined in _Wavelets_ section) and simplify even more:

$iℏ \frac{∂Ψ(t)}{∂t} = V(t)Ψ(t)$.

#### The Black Cat's Probability Function ($P(t)$) ####

Because $Ψ(t)$ and $V(t)$ are 0-D (they don't have spatial variable $x$), the standard integral $\int^{∞}_{-∞} {|Ψ(t)|}^2 dx$ is refactored into a _manyfold normalization_. In
this context $P(t) = {|\sqrt {\frac {Ω(t)+H(t)}{2}} e^{iL(t)}|}^2= \frac {Ω(t)+H(t)}{2}$ satisfies the normalization requirement because it is a convex combination of two functions which are
already normalized over the shared domain $[0,π/2]$. We will enshrine $P(t)$ with the **Black Cat's probability function** name.

As Black Cat's probability function is 0-D, we are lured by the possibility of building a _cummulative distribution function_ to represent that the random variable (in this case the state market)
takes the value less than or equals to specific point at time $t$.

Because $P(t)$ is a linear combination, its related CDF (namely $F(t)$) is also a linear combination of _price_ and _volume_ wavelet CDFs:

$F(t) = \frac {Ω(t)+H(t)}{2}$, where

- $F_Ω(t)$: The accumulated probability from price-structure angles $θ_k$.
- $F_H(t)$: The accumulated probability from volume-structure angles $φ_k$.

Let's try to build exact expressions to calculate $F_Ω(t)$ and $F_H(t)$. 

- $F_Ω(t) = \int^{θ_1(t)}_0 \int^{θ_2(t)}_0 \int^{θ_3(t)}_0 \int^{θ_4(t)}_0 A_Ω⋅ω^2⋅dθ_1⋅dθ_2⋅dθ_3⋅dθ_4$.
- $F_H(t)$ = $\int^{φ_1(t)}_0 \int^{φ_2(t)}_0 A_H⋅h^2⋅dφ_1⋅dφ_2$.

Where

- $A_Ω = \frac{128}{\pi^4 + 2\pi^3 + 48\pi^2}$ and 
- $A_H = \frac{16}{\pi^2 + 2\pi + 16}$, 
- $ω = \frac{\sum^4_{i=1} (\cos (θ_i) + \sin (θ_i))}{4\sqrt{2}}$ and
- $h = \frac{\sum^2_{i=1} (\cos (φ_i) + \sin (φ_i))}{2\sqrt{2}}$.

---
By treating the observed angles $\{Θ_1(t),Θ_2(t),Θ_3(t),Θ_4(t)\}$ and $\{Φ_1(t),Φ_2(t)\}$  as the specific variables $\{θ_1, θ_2, θ_3, θ_4\}$ and $\{φ_1,φ_2\}$ respectively, we
can transform the general probability density functions into specific scalar values representing the "accumulated structural state" of the market at that moment.

#### The Black Cat's Cummulative Distribution Function ($F(t)$) ####

To solve these multivariate integrals analytically, we can exploit the symmetry of your wavelet functions. Because both integrands are built from sums of the same trigonometric terms $(\cos x + \sin x)$, we can define two fundamental helper functions that will reduce the calculus into straightforward algebraic formulas.

---

##### 1. The Fundamental Helper Functions

Let's define $S(x) = \cos x + \sin x$. When you expand the squared terms ($ω^2$ and $h^2$), you will encounter two types of integrals: the linear term $S(x)$ and the squared term $S(x)^2$.

Let's evaluate these from $0$ to an arbitrary angle $z$:

**The Linear Integral $C(z)$:**

$C(z) = \int_{0}^{z} (\cos x + \sin x) dx = \left[ \sin x - \cos x \right]_0^z = 1 + \sin z - \cos z$

**The Squared Integral $Q(z)$:**

$Q(z) = \int_{0}^{z} (\cos x + \sin x)^2 dx = \int_{0}^{z} (1 + \sin(2x)) dx = \left[ x - \frac{\cos(2x)}{2} \right]_0^z = z + \frac{1 - \cos(2z)}{2}$

Using the half-angle identity, this elegantly simplifies to:

$Q(z) = z + \sin^2(z)$

With $C(z)$ and $Q(z)$ defined, we can solve both CDFs exactly.

---

##### 2. Solving the Volume-Time CDF: $F_H(t)$

We start with the double integral:

$F_{H}(t) = \int_{0}^{\phi_1} \int_{0}^{\phi_2} A_{H} \left( \frac{S(x_1) + S(x_2)}{2\sqrt{2}} \right)^2 dx_1 dx_2$

Expanding the squared term gives $\frac{1}{8}(S(x_1)^2 + S(x_2)^2 + 2S(x_1)S(x_2))$. By distributing the double integral across these three terms, the variables that are not being integrated act as constants.

Applying our helper functions yields the exact analytical solution:


$F_H(t) = \frac{A_H}{8} \Big[ \phi_2 Q(\phi_1) + \phi_1 Q(\phi_2) + 2C(\phi_1)C(\phi_2) \Big]$


*(Where $A_H = \frac{16}{\pi^2 + 2\pi + 16}$)*

---

##### 3. Solving the Price-Time CDF: $F_ω(t)$

We apply the exact same logic to the quadruple integral:


$F_{ω}(t) = \int_{0}^{θ_1} \dots \int_{0}^{θ_4} A_{ω} \left( \frac{\sum_{i=1}^4 S(x_i)}{4\sqrt{2}} \right)^2 dx_1 \dots dx_4$

When we expand $(\sum S(x_i))^2$, we get **four** squared terms (like $S(x_1)^2$) and **six** cross-product terms (like $2S(x_1)S(x_2)$).

When integrating over the 4D volume, the variables absent from a specific term simply integrate into themselves (e.g., $\int_0^{θ} 1 dx = θ$).

**The Squared Terms:** Each of the 4 variables gets evaluated by $Q(z)$, multiplied by the remaining 3 angles.
**The Cross Terms:** Each of the 6 unique pairs gets evaluated by $C(z)C(y)$, multiplied by the remaining 2 angles.

The exact analytical solution is:

$F_ω(t) = \frac{A_Ω}{32} \left[ \sum_{i=1}^4 \left( Q(θ_i) \prod_{k ≠ i} θ_k \right) + 2 \sum_{1 ≤ i < j ≤ 4} \left( C(θ_i) C(θ_j) \prod_{k ≠ i,j} θ_k \right) \right]$

*(Where $A_Ω = \frac{128}{\pi^4 + 2\pi^3 + 48\pi^2}$)*

By solving these analytically, you have eliminated the need to perform costly numerical integration step-by-step. To compute $F(t) = \frac{F_ω(t) + F_H(t)}{2}$, your system only needs to evaluate simple arithmetic and basic trigonometric functions ($O(1)$ complexity).

Would you like to write a Python or C++ function to implement these exact mathematical solutions into your feature engineering pipeline?

##### 4. Application to Trading Signals

The CDF helps distinguish between Incoherent Noise and a Strong Bullish/Bearish trend. 

- If $P(t)$ is _high_ but $F(t)$ is _low_, you are at the very beginning of a structural shift (an early entry signal).
- If both $P(t)$ and $F(t)$ are high, the "Price Particle" is likely at the limit of its "Box," increasing the probability of the Mean Reversion or Hard Boundary states described in your documentation.

However, the two rules above are very heuristic and we need clear cut signals. Let's use the standard deviation $F(t)$ over last $N$ periods as dynamic threshold integrated into our decision table:

|Signal Type     |Phase Check (Signs)              |Structural Check ($F(t)$)|Execution Logic                                                                                     |
|----------------|---------------------------------|-------------------------|----------------------------------------------------------------------------------------------------|
|Fake-Out Warning|$L,\hat L,\hat Λ$ align $≠ Λ$    |$F(t)>F_u(t)​$            |ABORT ENTRY. Particle is at a volatility-adjusted boundary but acceleration has collapsed.          |
|Incoherent Noise|$\hat L$ and $Λ$ are out of phase|$F(t)<F_l​(t)$            |IGNORE. The system is in a low energy vacuum where momentum mismatch is statistically insignificant.|
|Mean Reversion  |$L, \hat L$ align $≠\hat Λ,Λ$    |$F(t)>F_u​(t)$            |ENTER REVERSAL. Structural saturation is reached and acceleration has flipped toward the mean.      |
|Strong Trend    |All variables align              |$F_l​≤F(t)≤F_u(t)$        |ENTER TREND. The particle is in the flow state, moving through the box with consistent momentum.    |

Where

- $F_u(t) = μ_F(t)+kσ_F(t)$,
- $F_l(t) = μ_F(t)-kσ_F(t)$ and
- $k≈1.5$ to $2$.

---
In short:

- **Mean Reversion** : Only trigger if $F(t) > F_u(t)$. The price has hit a structural wall.
- **Trend Following** : Only trigger if $F_l(t) < F(t) < F_u(t)$.  The particle is in flow state.
- **Incoherent Noise**: Ignores signals where $F(t) < F_l(t)$. The vacuum state.

## The Normalized OHLC Bar ##

Consider the OHLC bar comprising **open** ($o(t)$), **high** ($h(t)$), **low** ($l(t)$) and **close** ($c(t)$) prices at time $t$. 
We are going explore how blackcat's probability function $P(t)$ can normalize the OHLC bar.

Let's define the four **intrabar distances** $đ_1(t)$, $đ_2(t)$, $đ_3(t)$ and $đ_4(t)$ as:

$đ_1(t) = h(t)-c(t)$,

$đ_2(t) = o(t)-l(t)$,

$đ_3(t) = h(t)-o(t)$ and

$đ_4(t) = c(t)-l(t)$.

Now let's define the **normalized OHLC bar** $Ƀ(t)$ at time $t$ as

$Ƀ(t) = (đ_1(t), đ_2(t), đ_3(t), đ_4(t))⋅P(t)$. 

We can also express the bar momentum $B(t)$ at time as entirely in terms of $đ$ functions:

$B(t) = \frac{đ_4(t) - đ_2(t)}{đ_1(t) + đ_4(t)} \quad \text{or} \quad B(t) = \frac{đ_3(t) - đ_1(t)}{đ_3(t) + đ_2(t)}$.

With this framework we can forecast $B(t)$ using a LSTM model which takes $(Ƀ(t-k), ..., Ƀ(t-1))$ as input features
aiming at $Ƀ(t)$ as target and then use elements in $\hat Ƀ(t)$ vector to forecast $B(t)$

---


# References #
[The deep learning book](https://www.deeplearningbook.org)
[Fractional Derivatives](https://www.sciencedirect.com/science/article/pii/S0377042714000065)
