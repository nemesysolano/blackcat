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
Consider the four price-time angles ${θ_1(t)}$, ${θ_2(t)}$, ${θ_3(t)}$ and ${θ_4(t1)}$ ruling at time ${t}$. The **price-time wavelet $W(t)$** function
is a periodic non-linear function defined as

$W(t) =\frac{\sum^4_{i=1} (\cos θ_i(t) + \sin θ_i(t))^2}{8}$.

Following the same lines, consider the two volume-time angles ${Φ_1(t)}$ and ${Φ_2(t)}$ ruling at time $t$. The **volume-time wavelet $V(t)$** function
is a periodic non-linear function defined as

$V(t) = \frac{\sum^2_{i=1} (\cos Φ_i(t) + \sin Φ_i(t))^2}{4}$.

The $\tanh (32⋅|W(t)|⋅|V(t)|)$ formula is called **market power** which, albeit not actually tradable, will be provided as input to the ensemble model as measurement of trend stability$ 

## Price Momentum ##

**Price momentum measures** the _velocity_ of price changes as opposed to the actual price levels themselves and (unlike price) is a directional quantity. In this work
we are going two define three kinds of price momentum.

### Volume-Price Momentum ####

Volume-Price momentum is defined as

$y(t) = Δ_1(x(t))⋅|Δ_1(v(t))| $ where

$Δ_1$ is $Δ_T$ with $T=1$

### Price-Time Wavelet Momemtum ###

Volume-Price momentum is defined as

$w(t) = Δ_1(x(t))⋅W(t)$

### Volume-Time Wavelet Momemtum ###

Volume-Price momentum is defined as

$v(t) = Δ_1(v(t))⋅V(t)$

### Average Wavelet Force ###

The average wavelet momentum is defined as

$a(t) = \frac{W(t) + V(t)}{2}$

### Bar Direction ###

Consider the OHLC bar at time $t$; the **bar momentum** is defined as 

$\vec b = Δ_{\%}(c(t) - o(t), h(t) - l(t))$, where

$o(t),\space  l(t),\space  h(t)$ and $c(t)$ are open, low, high and close price at time $t$ respectively.


## Price Acceleration Model ##

Let $X=\{x(1),...,x(t-1), x(t),...\}$ a sequence of momenta belonging to the same type (Volume-Price, Price-Time Wavelet, Volume-Time Wavelet or Average Wavelet). A DNN model
that forecast

 $x(t)-x(t-1)$ from past $\{x(t-k)-x(t-(k-1)),...,x(t-2)-x(t-1)\}$ is called **price acceleration model**.

## Risk Management and Position Sizing ##

### The Risk Assestment Reports ###

Trainining process for price acceleration models generate the **risk assestment reports** sketched below

|Ticker|MSE|MAE|Match %|Diff %|Pct Diff%|Edge|  tradable    |
|------|---|---|-------|------|---------|----|--------------|
|AAAAA |0.#|0.#|0.#    |0.#   |0.#      |99  | True or False|

The meaning of the "Ticker", "MSE", "MAE" columns must be obvious to the reader. Let's describe the other columns:

1. $\text{Match \%}$: Directional matches ratio (i.e How many times expected direction matches actual direction).
2. $\text{Diff \%}$: Directional mismatches ratio (i.e How many times expected direction does not match actual direction).
3. $\text{Pct Diff\%}$: Absolute difference between **Match %** and **Diff %**.
4. $\text{Edge}$: $\max (\text{Match \%},\text{Diff \%}) 100 - 50$
5. Tradable: True if $\text{Edge} > 6$

---
Trading simulators must check $\mathbf{sign}(\text{Match \%}-\text{Diff \%})$ and handle it as contrarian signal (_sell_ when positive, _buy_ when negative)
when it is negative.

To calculate your precise Stop Loss (SL) and Take Profit (TP) levels, you need to combine the model's prediction magnitude, its historical error (MAE), and the statistical edge (edge_diff).
Here is the step-by-step assembly for a ticker, using the values found in your report-pricevol.csv.
#### 1. Define the "Unit of Risk" ($d_{risk}$) ####
The MAE (Mean Absolute Error) represents the average distance the model is "off" by. To avoid being stopped out by normal model noise, your stop loss distance should be a multiple of this error.

* Formula: $d_{risk} = k \cdot MAE$

* Recommended $k$: $1.5$ to $2.0$ (Higher $k$ reduces stop-outs but requires a larger move to hit TP).

#### 2. Determine the Target Reward Ratio ($R$) ####
Use your edge_diff to find the minimum Reward-to-Risk ratio needed to break even. To ensure a positive expectancy, we add a Profit Buffer (e.g., $0.2$ to $0.5$).

* Break-even Floor: $R_{floor} = \frac{50 - \text{edge\_diff}}{50 + \text{edge\_diff}}$ (Kelly Size)

* Target Ratio: $R_{target} = R_{floor} + \text{Buffer}$

#### 3. Calculate the Price Levels ####
Now, apply these distances to the current price ($P_{curr}$) based on the model's predicted direction (the sign of $y_{pred}$ from your script).

* For a Long Trade ($y_{pred} > 0$):

  * Stop Loss Price = $P_{curr} \cdot (1 - d_{risk})$

  * Take Profit Price = $P_{curr} \cdot (1 + (d_{risk} \cdot R_{target}))$

* For a Short Trade ($y_{pred} < 0$):

  * Stop Loss Price = $P_{curr} \cdot (1 + d_{risk})$

  * Take Profit Price = $P_{curr} \cdot (1 - (d_{risk} \cdot R_{target}))$

----

#### Worked Example: AAOI ####
Based on your latest report, AAOI has an MAE of 0.8580 and an edge_diff of 19. Let's assume the current price is $20.00.

1. Calculate Risk Distance: Using $k=1.5$, $d_{risk} = 1.5 \cdot 0.00858 \approx 0.0128$ ($1.28\%$).

2. Calculate Reward Ratio: * $R_{floor} = \frac{50-19}{50+19} = \frac{31}{69} \approx 0.45$.

   * Add a $0.55$ buffer for a clean 1:1 Reward:Risk ratio ($R_{target} = 1.0$).
     * Set Levels (assuming Long):
       * Stop Loss: $20.00 \cdot (1 - 0.0128) = \mathbf{\$19.74}$
       * Take Profit: $20.00 \cdot (1 + 0.0128) = \mathbf{\$20.26}$
     * Set Levels (assuming Short):
       * Stop Loss: $20.00 \cdot (1 + 0.0128) = \mathbf{\$20.26}$
       * Take Profit: $20.00 \cdot (1 - 0.0128) = \mathbf{\$19.74}$

### Final Sanity Check with Angles ###

#### Entry Rules Table ####
|Trade Direction ($y_{\text{pred}}$)|Structural Requirement (The Filter)|Pivot Geometry Logic                          |Execution Action     |
|-----------------------------------|-----------------------------------|----------------------------------------------|---------------------|
| Momentum Up (+)                   | $θ_{l↑} > θ_{h↓}$                 |Higher Low is closer/stronger than Lower High.|BUY (LONG)           |
| Momentum Up (+)                   | $θ_{l↑} < θ_{h↓}$                 |Price is hitting a descending ceiling.        |PASS (or Reduce Size)|
| Momentum Down (-)                 | $θ_{h↓} > θ_{l↑}$                 |Lower High is closer/stronger than Higher Low.|SELL (SHORT)         |
| Momentum Down (-)                 | $θ_{h↓} < θ_{l↑}$                 |Price is hitting a rising floor.              |PASS (or Reduce Size)|

When we talk about "reducing size" in a strategy, it usually refers to Position Sizing—choosing to buy fewer shares our maximum allowed capital—rather than opening ten separate small trades.
Instead of opening multiple orders, you calculate a Volume Factor ($V_f$) based on our sanity checks:

#### Volume Factor Table ####
|Condition|Volume Factor|Final Shares to Buy                    |
|---------|-------------|---------------------------------------|
|Momentum + Structure   |1.0          |`Max_Kelly_Size`         |
|Momentum only          |0.5          |`Max_Kelly_Size * 0.5`   |
|Divergence             |0.0          |Do not trade             |

The `Max_Kelly_Size` (for know on denoted as $K_{\max}$) is defined as

$K_{\max} = \frac{\text{edge\_diff}}{50}$


# References #
[The deep learning book](https://www.deeplearningbook.org)

# Scripts #
