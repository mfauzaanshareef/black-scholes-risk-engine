# General Assumptions
- We assume a constant risk free rate and constant volatility.
- The underlying stock doesn't pay any dividends.
- Frictionless markets: there are no transaction costs.
- Continuous trading: we can buy and sell any amount of stock, even frictional amounts.
# Method 1

### 1. Assume that the price of the stock follows a GBM

$$
dS = \mu S dt + \sigma S dW \qquad (1)
$$

**Where:**
* $dS$: The change in the stock price.
* $\mu S dt$: The deterministic drift component (expected return of the stock), where $\mu$ is the drift rate.
* $\sigma S dW$: The stochastic component, where $\sigma$ is volatility of the stock price and $dW$ is a Wiener process (Brownian motion).

### 2. Define a function that gives the value of an option and apply Ito's Lemma

$$
V = V(S, t) \qquad (2)
$$

**Where:**
* $S$: The price of the underlying stock.
* $t$: Time.

Apply Ito's Lemma to the function $V$ to expand it using the Taylor's series (keeping terms up to order $dt$):

$$
dV = \frac{\partial V}{\partial t} dt + \frac{\partial V}{\partial S} dS + \frac{1}{2} \frac{\partial^2 V}{\partial S^2} (dS)^2 \qquad (3)
$$

We modeled the price of a stock in Step 1. Square it to get $(dS)^2$:

$$
(dS)^2 = (\mu S dt + \sigma S dW)^2 = \mu^2 S^2 dt^2 + 2\mu \sigma S^2 dt dW + \sigma^2 S^2 dW^2
$$

> **Notes (Things that cancel out):**
> * $dt^2 \to 0$ (Second-order term is negligible).
> * $dt dW \to 0$ (Term of order $dt^{1.5}$ is negligible).
> * $dW^2 \to dt$ (The variance of a Wiener process scales linearly with time).

So $(dS)^2$ becomes:

$$
(dS)^2 = \sigma^2 S^2 dt
$$

Now insert this result for $(dS)^2$ into Equation 3 to get:

$$
dV = \frac{\partial V}{\partial t} dt + \frac{\partial V}{\partial S} (\mu S dt + \sigma S dW) + \frac{1}{2} \frac{\partial^2 V}{\partial S^2} (\sigma^2 S^2 dt)
$$

Group $dt$ terms and $dW$ terms to get:

$$
dV = \left( \frac{\partial V}{\partial t} + \mu S \frac{\partial V}{\partial S} + \frac{1}{2} \sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} \right) dt + \left( \sigma S \frac{\partial V}{\partial S} \right) dW \qquad (4)
$$

We still have the $dW$ term meaning that randomness is still present.

### 3. Construct a portfolio of stock and option. Apply delta-hedging.

$$
\Pi = V - \Delta S \qquad (5)
$$

**Where:**
* $\Pi$: The value of the portfolio.
* $V$: The value of the option (Long position).
* $-\Delta S$: A short position of size $\Delta$ in the underlying stock.

This portfolio changes in value according to:

$$
d\Pi = dV - \Delta dS \qquad (6)
$$

Substitute $dS$ from Equation 1 and $dV$ from Equation 4 into Equation 6 to get:

$$
d\Pi = \left[ \left( \frac{\partial V}{\partial t} + \mu S \frac{\partial V}{\partial S} + \frac{1}{2} \sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} \right) dt + \sigma S \frac{\partial V}{\partial S} dW \right] - \Delta (\mu S dt + \sigma S dW)
$$

Group the $dt$ and $dW$ terms to get:

$$
d\Pi = \left( \frac{\partial V}{\partial t} + \mu S \frac{\partial V}{\partial S} + \frac{1}{2} \sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} - \Delta \mu S \right) dt + \left( \sigma S \frac{\partial V}{\partial S} - \Delta \sigma S \right) dW \qquad (7)
$$

Apply delta hedging. Set Delta equal to the change in the option price with respect to the change in the underlying stock price:

$$
\Delta = \frac{\partial V}{\partial S}
$$

Substitute this to the portfolio value equation (No. 7).

$$
d\Pi = \left( \frac{\partial V}{\partial t} + \frac{1}{2} \sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} \right) dt \qquad (8)
$$

> **Notes:**
> * The cancellation of the $dW$ term removes the randomness, making the portfolio deterministic.
> * The terms involving $\mu$ also cancel out because $\mu S \frac{\partial V}{\partial S} - \frac{\partial V}{\partial S} \mu S = 0$. The cancellation of the $\mu$ terms implies that the option price does not depend on the drift (expected return) of the stock.
> * Now the portfolio value only depends on the risk-free rate and volatility of the underlying stock.

### 4. Apply the no-arbitrage principle

Since the portfolio is now risk-free, it must earn the risk-free rate $r$. The change in value of the portfolio is:

$$
d\Pi = r \Pi dt
$$

Substitute the known values for $\Pi$ (Equation 5) and $d\Pi$ (Equation 8) and equate to get:

$$
\left( \frac{\partial V}{\partial t} + \frac{1}{2} \sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} \right) dt = r \left( V - S \frac{\partial V}{\partial S} \right) dt
$$

Divide by $dt$ and rearrange to get:

$$
\frac{\partial V}{\partial t} + \frac{1}{2} \sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} + r S \frac{\partial V}{\partial S} - rV = 0
$$

**The Black-Scholes PDE:**

$$
\underbrace{\frac{\partial V}{\partial t}}_{\text{Time Decay}} + \underbrace{r S \frac{\partial V}{\partial S}}_{\text{Drift Return}} + \underbrace{\frac{1}{2} \sigma^2 S^2 \frac{\partial^2 V}{\partial S^2}}_{\text{Convexity}} = \underbrace{rV}_{\text{Risk-Free Return}}
$$

> **Notes:**
> The time-decay of the option, plus the gains from the risk neutral drift (from bank account), plus the convexity benefits from stock price volatility, must add up to the risk-free return on the option's value.

# Method 2

- This method is less intuitive than method 1 but it requires less equation expansion.
- It is less intuitive because we don't see the cancellation of the randomness term $dW$ and drift term $\mu$.

### 1. Assume that the price of the stock follows a GBM

The stock price $S$ follows a Geometric Brownian Motion:

$$
dS = \mu S dt + \sigma S dW \qquad (1)
$$

**Where:**
* $dS$: Change in stock price.
* $\mu, \sigma$: Drift and volatility constants.
* $dW$: The random Wiener process.

### 2. Define a function that gives the value of an option

Let $V(S, t)$ be the price of a call option.

### 3. Define the portfolio

We construct a portfolio $\Pi$ consisting of one long option ($V$) and a short position in $\Delta$ shares of the underlying stock ($S$):

$$
\Pi = V - \Delta S \qquad (2)
$$

### 4. Define the change in that portfolio

We are interested in how the value of this portfolio changes over a tiny time step $dt$:

$$
d\Pi = dV - \Delta dS \qquad (3)
$$

> **Note:** 
> - In this method, we don't expand $dS$ yet. We keep it as a single term to make the algebra cleaner.

### 5. Apply Ito's Lemma to V(S,t)

To find $dV$, we use the Taylor Series expansion for Ito Calculus:

$$
dV = \frac{\partial V}{\partial t} dt + \frac{\partial V}{\partial S} dS + \frac{1}{2} \frac{\partial^2 V}{\partial S^2} (dS)^2 \qquad (4)
$$

### 6. Square the dS equation from GBM

We need the value of $(dS)^2$ for the equation above. From Equation (1):

$$
(dS)^2 = (\mu S dt + \sigma S dW)^2
$$

Ignoring terms higher than order $dt$ (since $dt^2 \to 0$ and $dt dW \to 0$) and using the rule that $dW^2 = dt$:

$$
(dS)^2 = \sigma^2 S^2 dt
$$

Substitute this back into the Ito's Lemma equation (Equation 4):

$$
dV = \frac{\partial V}{\partial t} dt + \frac{\partial V}{\partial S} dS + \frac{1}{2} \sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} dt \qquad (5)
$$

### 7. Replace dV in the change of portfolio function

Now substitute Equation (5) into our portfolio change equation (Equation 3):

$$
d\Pi = \left( \frac{\partial V}{\partial t} dt + \frac{\partial V}{\partial S} dS + \frac{1}{2} \sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} dt \right) - \Delta dS
$$

Group the terms by $dt$ and $dS$:

$$
d\Pi = \left( \frac{\partial V}{\partial t} + \frac{1}{2} \sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} \right) dt + \left( \frac{\partial V}{\partial S} - \Delta \right) dS \qquad (6)
$$

### 8. Cancel out dS (Delta Hedging)

We want to eliminate risk. The risk in this equation is entirely contained in the $dS$ term (because $dS$ contains the random $dW$).

To make the risk vanish, we set the coefficient of $dS$ to zero:

$$
\frac{\partial V}{\partial S} - \Delta = 0 \quad \Rightarrow \quad \Delta = \frac{\partial V}{\partial S}
$$

If we choose this specific $\Delta$, the $dS$ term disappears from Equation (6). The equation becomes:

$$
d\Pi = \left( \frac{\partial V}{\partial t} + \frac{1}{2} \sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} \right) dt \qquad (7)
$$

> **Notes**
> - We don't have to expand $dS$ into $\mu$ and $dW$ and cancel them separately. By cancelling $dS$, we removed both the randomness ($dW$) and the drift ($\mu$) in one move.

### 9. Define the return of the risk-free portfolio

Since the portfolio has no risk (no $dS$ term), it must earn the risk-free interest rate $r$.

$$
d\Pi = r \Pi dt
$$

Substitute the original definition of the portfolio $\Pi = V - \Delta S$ (Equation 2) into this:

$$
d\Pi = r (V - \Delta S) dt
$$

Since we know $\Delta = \frac{\partial V}{\partial S}$, this becomes:

$$
d\Pi = r \left( V - S \frac{\partial V}{\partial S} \right) dt \qquad (8)
$$

### 10. Equate both dPi equations

Now we simply equate the result from our Ito derivation (Equation 7) with the result from the risk-free argument (Equation 8):

$$
\left( \frac{\partial V}{\partial t} + \frac{1}{2} \sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} \right) dt = r \left( V - S \frac{\partial V}{\partial S} \right) dt
$$

Divide by $dt$ and rearrange all terms to one side:

$$
\frac{\partial V}{\partial t} + \frac{1}{2} \sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} + r S \frac{\partial V}{\partial S} - rV = 0
$$

The Black-Scholes PDE:

$$
\underbrace{\frac{\partial V}{\partial t}}_{\text{Time Decay}} + \underbrace{r S \frac{\partial V}{\partial S}}_{\text{Drift Return}} + \underbrace{\frac{1}{2} \sigma^2 S^2 \frac{\partial^2 V}{\partial S^2}}_{\text{Convexity}} - rV = 0
$$