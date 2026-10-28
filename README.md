# SCRL course project: return seeking and residual risk control

This repository contains a course proposal and research recommendations for portfolio reinforcement learning using IntelliQuant as a potential data and backtesting platform.

**Recommended question:** Can a separately trained risk controller improve the out-of-sample return–risk trade-off of a frozen return-seeking policy beyond volatility targeting, conventional portfolio optimization, and joint reward training?

The central idea is to separate return seeking from risk management, then test whether that separation is useful when prices are exogenous but portfolio holdings, costs, wealth, and drawdown depend on actions. Model-free versus model-based training is a controlled comparison within this project. AR/VAR and recurrent market models are supporting methods, rather than independent research contributions.

**Status:** Research design and proposal draft only. No trading environment, trained policies, data export, or empirical results have been implemented or validated. Platform assessment was checked against public documentation on **October 28, 2026**, with a read-only inspection of the signed-in strategy overview and a sample editor. Account-specific entitlements, runtime behavior, and data quality remain unverified.

## Repository contents

| File | Purpose |
| --- | --- |
| [StochRL26_ID_Name_Proposal.tex](StochRL26_ID_Name_Proposal.tex) | Original course template; specifies a two-page body limit, excluding references. |
| [StochRL26_ResidualRisk_Proposal_Draft.tex](StochRL26_ResidualRisk_Proposal_Draft.tex) | Draft research proposal following the template's sections. |
| [README.md](README.md) | Expanded rationale, platform assessment, experimental design, and implementation recommendations. |

The draft still has author/student-ID placeholders. The built-in LaTeX compiler could not start because of an environment setup failure; successful compilation and compliance with the two-page body limit have **not** been verified. This README expands the reasoning and does not replace the submission-length proposal.

## 1. Why trading can still require sequential control

For a small price-taking trader, it is reasonable to assume an allocation decision does not change the market price process. It does not follow that the action leaves the entire state unchanged.

Let market observations be represented by history $h_t$ and let $z_t$ contain the account state. The structural assumptions are

$$
P(x_{t+1}\mid h_t,z_t,a_t)=P(x_{t+1}\mid h_t),
\qquad z_{t+1}=F(z_t,a_t,x_{t+1}).
$$

The first equality assumes the account carries no additional market information beyond the observed market history. A finite feature window is an approximate information state, not a proven Markov state.

| Exogenous market information | Controlled account state |
| --- | --- |
| Prices, returns, trading volume | Holdings and portfolio weights |
| Market indicators and historical features | Cash and wealth |
| Publicly available fundamentals | Previous positions and future turnover costs |
| Market regime proxies | Running peak wealth and drawdown |

Holding a position today affects what it costs to rebalance tomorrow. A loss changes drawdown and can affect a later risk constraint. These effects motivate sequential control even without market impact.

There is also an important limiting case: with action-independent future observations, no costs, no account-dependent feasible sets, and an additive objective whose period utility depends only on the current market information and allocation, each decision reduces to conditional one-step utility maximization. Autocorrelated returns alone do not invalidate that decomposition. This is a useful diagnostic and a modest theoretical observation, not a major novelty claim.

**Recommendation:** Include costs and account state explicitly, and compare against a one-step allocator. Do not justify RL solely by saying that prices form a time series.

## 2. What to take from the supplied paper

The supplied paper is [ResSafe: Learning Safety Filtering with Residual Reinforcement Learning for Humanoids](https://arxiv.org/abs/2609.15988). Its relevant architecture freezes a task policy and trains a residual action correction. It is not simply a method that learns several reward weights in parallel. Its robotics objective includes survival and intervention costs, and the authors do not claim hard safety guarantees for the learned filter.

The proposed financial adaptation is our research recommendation: first learn net-return-seeking allocations, then learn risk corrections while limiting degradation in the original objective. Robotics safety results do not establish financial risk guarantees.

Keep three different meanings of “weights” separate:

| Quantity | Meaning | Suggested treatment |
| --- | --- | --- |
| Portfolio weights $w_t$ | Capital allocated to assets and cash | Policy output |
| Reward coefficients | Relative scaling of return, risk, and intervention terms | Small preregistered validation grid or explicit constraints |
| Network parameters | Parameters of nominal policy, residual policy, and critics | Optimize during the relevant training stage |

Separate critics can estimate return and risk in parallel. Separate residual experiments can also share a frozen nominal policy. Neither arrangement automatically resolves conflicts among objectives. Jointly updating both actors would be a different experimental condition because it removes the frozen reference.

## 3. Recommended formulation

### Assets, observations, and actions

Start with **3–5 liquid Korean ETFs plus cash**, long-only, without leverage. Choose an economically diverse universe using eligibility and liquidity information available at the start of the experiment, rather than selecting today's successful funds retrospectively. Exact symbols depend on verified listing dates and history coverage.

Use a compact feature set: lagged returns, trailing volatility, a few momentum measures, current holdings, cash fraction, and drawdown. Avoid a large technical-indicator search. Fundamentals are optional for a later stock-universe extension; an ETF allocation project does not need every field the platform provides.

Actions specify target weights on the asset-plus-cash simplex. The simulator converts targets into feasible orders, including lot sizes, cash constraints, and execution timing. Target weights are not guaranteed to equal realized weights after next-open price gaps and rounding.

### Stage A: train the nominal policy

Use one continuous-allocation algorithm, initially PPO, to maximize mean net log growth:

$$
g_{t+1}=\log(V_{t+1}/V_t),
\qquad J(\pi)=\mathbb{E}\left[\frac1T\sum_{t=0}^{T-1}g_{t+1}\right].
$$

Wealth must come from self-financing account updates with costs included. This objective is a precise interpretation of “return foremost”; expected log growth is not identical to expected arithmetic profit. State the chosen objective consistently in the proposal and evaluation.

Then freeze the nominal network $\pi_0$. During corrected rollouts, query it on the **actual corrected account state**, including the positions produced by previous residual decisions.

### Stage B: learn a residual risk controller

Let $w_t^0=\pi_0(h_t,z_t)$. A residual policy proposes a correction:

$$
w_t=\operatorname{Proj}_{\Delta^{K+1}}
\left(w_t^0+\delta_\phi(h_t,z_t,w_t^0)\right).
$$

A bounded correction is a useful implementation option. Projection enforces target allocation limits; order execution still needs its own feasibility checks.

For an estimated conditional covariance matrix $\widehat\Sigma_t$, including the chosen cash model, define

$$
C(\phi)=\mathbb{E}\left[\frac1T\sum_t
\left(w_t^\top\widehat\Sigma_t w_t+
\eta\|w_t-w_t^0\|_2^2\right)\right].
$$

The suggested primary problem is

$$
\min_\phi C(\phi)
\quad\text{subject to}\quad
J(\pi_\phi)\ge J(\pi_0)-\epsilon.
$$

Fix $\epsilon$ using training/validation data, in mean daily net log-return units. It allows some return sacrifice in exchange for risk reduction. A strict zero-tolerance profit-first objective may leave little freedom to improve risk.

An implementable Lagrangian is

$$
\mathcal L(\phi,\lambda)=C(\phi)+
\lambda\left[J(\pi_0)-\epsilon-J(\pi_\phi)\right],
\qquad \lambda\ge0.
$$

Minimize over the policy parameters and increase the multiplier when the estimated return floor is violated. Estimate nominal and corrected performance on comparable episodes. Normalize quantities before selecting learning rates; variance and log-return terms can have very different numerical scales.

This is approximate constrained optimization with estimation and optimization error. Freezing the nominal network does **not** preserve its returns automatically, and satisfying the empirical constraint does not guarantee the return floor on unseen data.

Variance alone can favor an all-cash allocation. The return floor, cash-exposure reporting, and matched-risk comparisons are essential to distinguish useful control from merely taking less risk.

### Controlled reward alternatives

| Objective or cost | Role | Main limitation |
| --- | --- | --- |
| Net log growth | Nominal objective | Sensitive to accounting and cost assumptions |
| $w_t^\top\widehat\Sigma_t w_t$ | Primary conditional variance proxy | Depends on covariance quality; not realized full-horizon variance |
| $\max(0,-r_{p,t+1})^2$ | Downside-risk ablation | Focuses on losses rather than all volatility |
| $\max(0,d_{t+1}-d_t)$ | Drawdown-increment ablation | Its sum is not maximum drawdown |
| Terminal maximum drawdown or episode loss CVaR | Optional path/tail-risk extension | Requires appropriate trajectory-level estimation |
| Squared residual size | Preserve nominal behavior | Small allocation changes do not guarantee small performance changes |

Do not treat per-step squared returns as an exact implementation of horizon variance, or average per-step Sharpe ratios as the portfolio Sharpe ratio. Use a small, predefined set of reward ablations. A CVaR extension must identify the loss variable, horizon, and confidence level; daily-return CVaR and terminal-wealth CVaR are different objectives.

## 4. Model-free and model-based comparison

### Model-free core

Train the residual policy on historical episodes with explicit account transitions. PPO may collect new trajectories by replaying market windows under its current policy. Replaying a known path is not collecting a new independent market realization.

Under the price-taking assumption, observed asset returns and a specified execution model permit simulated evaluation of alternative allocations on the same market path. This makes ordinary historical portfolio replay different from settings where only the reward for one logged action is known. It does not reveal actual unobserved fills or remove market-impact uncertainty.

### Model-based extension

Learn the exogenous market process and retain deterministic, explicit account accounting:

1. Fit an AR model for a single return series, or a regularized VAR for multiple assets.
2. Sample **joint** residual vectors or blocks to retain cross-asset dependence; use a suitable volatility model if needed.
3. Generate short, coherent market scenarios using only the training period.
4. Roll portfolio states through those scenarios and use the transitions for residual-policy optimization.
5. Evaluate the frozen policy on untouched real market periods.

A small probabilistic GRU/RNN or ensemble can be added after the VAR experiment works. A point forecast alone does not specify a useful distribution for risk estimation. Check dependence, volatility, tail behavior, and multi-step errors as well as mean prediction error.

**An AR/RNN forecast used only as a feature is forecast-assisted model-free RL, not by itself model-based RL.** A forecast-driven optimizer without a learned policy is a model-based control baseline. Keep these labels distinct.

For a focused comparison, hold the nominal policy fixed across model-free and model-based residual variants. Use common random numbers or shared historical paths for paired policy comparisons. Synthetic rollouts can improve training coverage under the fitted model, but cannot manufacture independent evidence or guarantee unseen crash coverage.

## 5. Novelty and the standard of evidence

The following are insufficient novelty claims by themselves:

- Applying PPO to prices.
- Adding volatility, drawdown, or Sharpe terms to a reward.
- Comparing AR and RNN forecasts.
- Combining a return model with a risk module.
- Replaying the same price path under different allocations.

A stronger course contribution is a careful answer to **when modular risk learning adds value in an exogenous market with controlled account dynamics**. Possible evidence includes better trade-offs after costs, less tuning sensitivity, robustness across regimes, and transfer to unseen nominal-policy checkpoints. Reusability must be tested; it is not implied by modular architecture.

Risk-aware RL and model-guided portfolio constraint correction already exist. Relevant starting points include [lexicographic multi-objective RL](https://www.ijcai.org/proceedings/2022/0476.pdf), [percentile risk-constrained RL](https://www.jmlr.org/papers/v18/15-636.html), and [neural-progressive hedging](https://proceedings.mlr.press/v180/ghosh22a.html). These motivate baselines and terminology; the literature check so far is not an exhaustive novelty review.

## 6. IntelliQuant assessment

### Verified capabilities and boundaries

| Area | Evidence and assessment |
| --- | --- |
| Strategy workflow | The inspected Studio interface offers script/block strategies, templates, a date range, initial capital, a backtest control, charts, metrics, logs, and holdings views. These controls were observed; a backtest was not executed. |
| JavaScript strategies | The [basic strategy guide](https://www.intelliquant.ai/help/manual/1) documents the strategy lifecycle. This supports a JavaScript backtesting workflow; it does not establish Python training support. |
| Market history | The [market-data guide](https://www.intelliquant.ai/help/manual/2) describes historical price access and adjusted closes; the [Stock reference](https://www.intelliquant.ai/help/ref/1) documents additional fields. Coverage and adjustment behavior need dataset-level validation. |
| Fundamentals | The [financial-data guide](https://www.intelliquant.ai/help/manual/3) documents quarterly financial fields and delayed availability. Verify announcement timing and restatement treatment before using them for research. |
| Accounts | The [accounts and orders guide](https://www.intelliquant.ai/help/manual/4) describes cash, positions, equity, orders, and multiple accounts. This is useful for replication of portfolio accounting. |
| Costs and execution | [IQEnvironment](https://www.intelliquant.ai/help/ref/15) documents configurable commissions, taxes, and execution modes. Match these settings explicitly across implementations. |
| Data extraction and training | A supported raw-data bulk export, external Python API, neural training runtime, and model-import facility have not been verified. Result downloads are not proof of raw training-data export. |

### Membership implications

The [membership page](https://www.intelliquant.ai/membership) currently lists:

| Plan | Backtest history | Runs/day | Excel results |
| --- | --- | --- | --- |
| BASIC | About two years, ending one month before the present | 10 | No |
| PRO | Korea: up to 10 years; US: up to 5 years, within advertised coverage | 100 | Yes |
| PREMIUM | Advertised Korea coverage from 2000 and US coverage from 2010 | Unlimited | Yes |

Listed monthly prices are KRW 30,000 for PRO and KRW 100,000 for PREMIUM. These are public plan terms, not confirmed entitlements for the signed-in account. Actual security histories depend on listing dates. BASIC is suitable for workflow exploration, but its short window is weak support for a deep-RL study across regimes. Paid access still does not establish raw-data export rights or functionality.

### Execution and data pitfalls

The environment reference distinguishes next-day open, next-day average-price, and same-day close execution. Prefer a close-to-next-open decision timeline. Using completed close-of-day information and assuming an immediately executable same-day close can introduce an optimistic timing assumption. Do not use the next day's range or closing price as an observation when placing an order at its open.

Do not assume documented default commission/tax values match the intended historical market or present account. Configure and record them. Add a clearly stated slippage assumption where appropriate. Verify corporate actions, ETF distributions, adjusted versus raw prices, suspended trading, and delisted instruments. Adjusted returns and cash distributions must not be double counted.

### Recommended role in the project

**Conditional recommendation:** Use IntelliQuant for strategy prototyping and final backtest replication. Use a local Python environment for training and reproducible experiments, provided the required data can be obtained through a supported and permitted route.

Proposed workflow:

```text
Verified data access and usage rights
                 |
       Versioned local dataset
                 |
   Local account simulator and training
                 |
       Frozen policy and settings
                 |
   Compact JavaScript implementation
                 |
IntelliQuant replication and discrepancy audit
```

Small network inference in JavaScript is a possible implementation route, **not a verified platform feature**. Check script-size limits, runtime, numerical parity, persistent state, and supported language/library features before committing to this route. If inference is impractical, a distilled policy is a separately evaluated approximation; it cannot silently replace the trained policy in the reported results.

If training-data extraction is unavailable, train on a separately licensed dataset and describe IntelliQuant as an external backtest check. Reconcile differences in data sources and fills rather than attributing them to the learning algorithm. Do not build a proposal whose essential data pipeline depends on an unverified export capability.

### First feasibility questions

- What history and download permissions does the actual account have?
- Does Excel export contain only strategy results, or the asset-level OHLCV/features needed for training?
- Is research extraction supported and permitted, and can the data be retained locally? Can it be redistributed in a public repository?
- Are delisted histories, historical eligibility, and announcement-time fundamentals available?
- How are adjusted prices, dividends, splits, and ETF distributions represented?
- Can a compact frozen policy run within script and runtime limits?
- Can a simple fixed-allocation strategy be reproduced locally with matching cash, holdings, and costs?

These are pending checks, not claims that the platform lacks the capabilities.

## 7. Baselines and ablations

| Method | What it tests |
| --- | --- |
| Equal-weight portfolio | Whether complexity beats a simple allocation rule |
| Uncorrected nominal PPO | Benefit and cost of intervention |
| Joint return–risk PPO | Whether two-stage training helps beyond a weighted reward |
| Nominal plus volatility targeting | Whether gains are explained by exposure reduction |
| Nominal plus minimum-deviation covariance optimizer | Whether an ordinary optimization filter suffices |
| Forecast plus one-step allocation | Whether prediction and myopic optimization explain the result |
| Direct policy optimization through differentiable replay, if feasible | Whether sequential optimization needs policy-gradient RL specifically |
| Model-free residual PPO | Core proposed method |
| VAR-scenario residual PPO | Added value of a learned market simulator |
| GRU-scenario residual PPO, optional | Whether added model capacity helps after simpler methods work |

Keep information, leverage limits, timing, costs, training periods, and tuning budgets comparable. Report both stages' total training compute. Match volatility or exposure using validation-calibrated controls; do not rescale test returns with hindsight and present them as an executable strategy.

Priority ablations are removing costs/path dependence, removing the residual intervention penalty, varying the return tolerance, comparing risk definitions, and testing frozen versus joint training. Transfer across nominal checkpoints and conditioning on forecast uncertainty are extensions. Avoid a full Cartesian sweep of algorithms, predictors, rewards, and asset universes.

## 8. Evaluation protocol

1. **Chronological splits:** Separate training, validation, and untouched test blocks. Use walk-forward folds where history permits. A 6/2/2-year split is only an example conditional on actual coverage; expanding-window folds may use data more efficiently.
2. **Causal preprocessing:** Fit scalers and learned predictors on training data. Rolling indicators/covariances can update during testing using observations available by the decision time, following a fixed rule. Do not retune on test outcomes.
3. **Horizon handling:** Purge training labels or scenario targets that overlap validation/test targets. Using past training-period observations as the initial test feature history is legitimate; using future test returns to fit a model is not.
4. **Account continuity:** Specify initial positions and cash and whether portfolios carry across test blocks. Do not hide liquidation, startup, or reset costs.
5. **Execution audit:** Form decisions after close and execute at the next open. Define how opening gaps, integer orders, unfilled orders, dividends, and cash interest are treated.
6. **Metrics:** Report net growth/CAGR, annualized volatility, Sharpe with a stated cash benchmark, maximum drawdown, turnover, cash exposure, realized tail-loss statistics, and return-floor violations.
7. **Uncertainty:** Run multiple training seeds and use paired moving-block bootstrap intervals where appropriate. Preserve dependence in resampling. Seeds measure optimization variability, not independent market evidence. Interpret Sharpe and tail estimates cautiously with short samples.
8. **Stress tests:** Increase costs/slippage, examine adverse held-out regimes, and test forecast misspecification. Model-generated stress is supplementary evidence.
9. **Selection discipline:** Choose reward coefficients, tolerances, checkpoints, and model capacity using validation data. Report the selection procedure and budget.
10. **Replication:** Compare local and IntelliQuant portfolio paths and investigate discrepancies in weights, trades, fees, and equity before comparing headline metrics.

Success means a repeatable improvement in the relevant return–risk frontier after costs and against conventional controls. If a standard optimizer or volatility rule matches the residual policy, that is an informative result about the limits of learned risk control.

## 9. Suggested five-week plan

| Period | Deliverable | Decision gate |
| --- | --- | --- |
| Week 1 | Verified data path, asset universe, accounting specification, simple replication | If export is unavailable, adopt a licensed training source immediately. |
| Week 2 | Nominal PPO and simple allocation/risk baselines | Confirm causal execution and credible baseline behavior. |
| Week 3 | Joint-reward and residual PPO with return-floor monitoring | Establish the core comparison before adding a market model. |
| Week 4 | Walk-forward results and VAR scenario ablation | Add GRU only if the core results and uncertainty estimates are complete. |
| Week 5 | Stress tests, platform discrepancy audit, figures, and report | Freeze final settings and report negative findings as well as positive ones. |

The minimum viable project is one ETF universe, one nominal algorithm, joint versus residual training, strong non-RL risk controls, and credible held-out evaluation. A large RNN, multiple RL algorithms, CVaR optimization, and extensive transfer experiments should not all be prerequisites for finishing the course project.

## 10. References and open decisions

- [ResSafe, arXiv:2609.15988](https://arxiv.org/abs/2609.15988): architectural inspiration.
- [Skalse et al., Lexicographic Multi-Objective Reinforcement Learning, IJCAI 2022](https://www.ijcai.org/proceedings/2022/0476.pdf): prioritized objectives.
- [Chow et al., Risk-Constrained Reinforcement Learning with Percentile Risk Criteria, JMLR](https://www.jmlr.org/papers/v18/15-636.html): chance/CVaR constrained formulations.
- [Ghosh et al., Neural-progressive hedging, UAI 2022](https://proceedings.mlr.press/v180/ghosh22a.html): prior model-guided constraint correction with portfolio applications.
- [IntelliQuant Studio guide](https://support.intelliquant.ai/guide/iq-studio), [Stock API](https://www.intelliquant.ai/help/ref/1), [environment API](https://www.intelliquant.ai/help/ref/15), and [membership](https://www.intelliquant.ai/membership): platform evidence.

Before submission, settle the exact data source and rights, asset universe, date range, course emphasis, experiment budget, return tolerance, cash model, and primary risk metric. Complete the novelty review and verify the LaTeX layout. Keep reported facts, proposed methods, and untested hypotheses distinct throughout the proposal.
