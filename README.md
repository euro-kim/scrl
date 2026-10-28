# SCRL project: budgeting the return cost of risk interventions

**Research question:** Can a personalized residual risk controller reduce portfolio drawdown while explicitly limiting the positive net-return shortfall caused by its interventions?

Train a return-seeking portfolio policy, freeze it, and learn a risk correction. Simulate an untouched reference account and a corrected account on the same market path, each with separate holdings, wealth, and costs. The proposed component budgets the corrected account's **positive shortfall relative to the executable reference policy**, rather than relying only on small action corrections or an average-return floor.

This is a proposed **Type II application extension**, not an established first in the literature or a new general safety principle. Exact overlap with prior work still requires verification. Policy transfer is not a requirement. Model-based market scenarios are optional.

## Status and files

This repository contains research design only. No environment, data pipeline, trained policy, or empirical result has been implemented or validated. The IntelliQuant assessment uses public documentation and a read-only inspection of the strategy overview/sample editor on **October 28, 2026**. Account entitlements, data export, inference runtime, and data quality remain unverified.

| File | Purpose |
| --- | --- |
| [StochRL26_ID_Name_Proposal.tex](StochRL26_ID_Name_Proposal.tex) | Original course template, preserved. |
| [StochRL26_ResidualRisk_Proposal_Draft.tex](StochRL26_ResidualRisk_Proposal_Draft.tex) | Current proposal centered on paired accounts and an intervention-loss budget. |
| [README.md](README.md) | Expanded formulation, research rationale, experimental plan, and platform assessment. |

The proposal has name/student-ID placeholders. The built-in LaTeX compiler cannot start because of an environment setup failure. Compilation and actual PDF page count remain unverified; the source targets two pages including references. Update this status only after successful compilation and page-count inspection.

## Course requirements

The supplied slides require a course-related topic with **at least one novel component** and a **two-page** proposal containing:

1. Title.
2. Motivation and problem definition.
3. Literature review.
4. Main ideas and novelty.
5. Expected contributions.
6. Detailed plan.
7. References.

The slides allow Type I theory/algorithm projects, Type II applications to practical problems, and Type III combinations. This project targets Type II. The original template excludes references from its body limit, while the slide simply states two pages; the current draft conservatively targets two pages total. Reproducing a safety architecture on another dataset alone is not the claimed component.

## 1. Motivation: budget the controller's economic harm

A risk controller can reduce drawdown by holding cash and missing profitable trades. Small allocation changes need not have small financial consequences. An average-return constraint also allows profitable interventions to offset harmful ones.

For example, suppose reference-minus-controlled cumulative log-growth gaps in two equally weighted episodes are +0.03 and -0.03. The signed average gap is zero, but average **positive shortfall** is 0.015. The new constraint registers sacrificed return rather than cancelling it against another episode's gains.

The hypothesis is that explicitly budgeting this shortfall produces a useful held-out drawdown–shortfall frontier. It could instead cause excessive conservatism or perform no better than fine-tuning. That outcome must be reported.

## 2. Price-taking market and controlled account state

Start with **three Korean ETFs plus cash**, long-only, without leverage. Choose a diverse universe using historical eligibility and liquidity information rather than today's successful funds. Verify symbols and listing histories before fixing the experiment.

For market history $h_t$ and account state $z_t$, assume

$$
P(x_{t+1}\mid h_t,z_t,a_t)=P(x_{t+1}\mid h_t),
\qquad z_{t+1}=F(z_t,a_t,x_{t+1}).
$$

Market history is assumed to contain any market information carried by the account; finite-window features only approximate an information state. Orders do not move prices, but they change positions, cash, wealth, subsequent turnover costs, and drawdown.

Use close-of-day information to form orders executed at the next open. Target weights become feasible orders with explicit cash, lot-size, cost, and opening-gap treatment. Define commissions, taxes, slippage, distributions, adjusted/raw prices, cash interest, and terminal accounting. Never observe the next day's close/range when deciding at its open.

With exogenous observations, no costs, no account-dependent constraints, and an additive per-period utility, allocation can reduce to conditional one-step optimization. Time-series data alone does not justify sequential RL. Here, costs and path-dependent drawdown create the sequential problem.

## 3. Paired-account objective and residual policy

### Reference and controlled trajectories

Train a nominal PPO policy for net log growth, then freeze its parameters **and deployment action rule**. Prefer deterministic reference actions for paired evaluation.

For every fixed $T$-day episode $\omega$, initialize two accounts with identical capital and positions:

| Account | Action source | State evolution |
| --- | --- | --- |
| Untouched reference | $\pi_0(h_t,z_t^0)$ | Own holdings, wealth, fees, and drawdown |
| Controlled | Corrected recommendation of $\pi_0(h_t,z_t^\phi)$ | Own holdings, wealth, fees, and drawdown |

The reference must never inherit corrected holdings. The nominal recommendation within the controlled system must use the controlled account's actual state. Identical market paths do not imply identical account trajectories.

Define

$$
G_i(\omega)=\log\frac{V_T^i}{V_0^i},
\qquad L_\phi(\omega)=[G_0(\omega)-G_\phi(\omega)]_+.
$$

Growth includes costs. Fix the horizon before comparing budgets: cumulative log-return tolerance over an episode is different from a mean daily-return tolerance or a guaranteed arithmetic percentage-point return floor.

For controlled wealth, define

$$
d_t=1-\frac{V_t^\phi}{\max_{u\le t}V_u^\phi},
\qquad D_t=\max_{u\le t}d_u.
$$

The primary problem is

$$
\min_\phi\;\mathbb E[D_T]
\quad\text{subject to}\quad
\mathbb E[L_\phi]\le\epsilon.
$$

### Personalized residual controller

For a nominal recommendation $w_t^0=\pi_0(h_t,z_t^\phi)$, use

$$
w_t=\operatorname{Proj}_{\Delta^4}
\left[w_t^0+\delta_\phi(o_t,w_t^0)\right].
$$

The observation can include lagged returns, volatility, both current account states, nominal allocation, relative cumulative growth, and time remaining. The untouched reference is a causal shadow account, updated using observed prices and modeled costs. It needs no real reference-account trades or future prices. Future outcomes supply completed training feedback and evaluation metrics only.

Projection enforces target weight limits; actual order execution still requires account feasibility. Include an explicit zero-correction bypass. With matched initialization and action/execution rules, bypass reproduces the reference and gives zero shortfall, so it is a feasible candidate for any nonnegative budget. This is a basic feasibility property, not an optimization or financial safety guarantee.

### What the constraint means

The signed alternative is $\mathbb E[G_0-G_\phi]\le\epsilon$. Because $u\le[u]_+$, the positive-shortfall constraint implies the signed constraint for the same episode distribution and budget. The converse does not generally hold. This elementary relationship motivates the experiment and is not claimed as a new theorem.

The constraint controls **expected positive terminal shortfall**. It does not bound every episode, maximum interim underperformance, CVaR, drawdown itself, or future-market losses. Shortfall may reflect sacrificed upside even when both policies profit. A small relative shortfall does not establish that the nominal strategy is profitable in absolute terms.

## 4. Training and implementation

1. Split market data chronologically. Keep complete episode targets inside their own split.
2. Fit causal features, train a compact nominal PPO policy for net growth, select its checkpoint using validation data, and freeze it.
3. Sample a training market window and simulate reference and controlled accounts with separate transitions.
4. Compute maximum drawdown and terminal positive shortfall. Optimize

$$
\mathcal L(\phi,\lambda)=\mathbb E[D_T]+
\lambda\left(\mathbb E[L_\phi]-\epsilon\right),
\qquad \lambda\ge0.
$$

5. Minimize over policy parameters and increase the multiplier on estimated budget violations. Separate critics can estimate the two costs; critics alone do not enforce the constraint.
6. Use undiscounted finite episodes. Costs $D_{t+1}-D_t$ telescope to terminal maximum drawdown when $D_0=0$. Positive increments of **current** drawdown do not have that property. Compute shortfall from complete paired episodes and preserve terminal costs on failures.
7. Select a small budget grid and learning settings on validation data, then evaluate frozen policies on untouched market periods.

Terminal shortfall supplies credit to earlier actions through trajectory learning; it need not be a differentiable per-step reward. Define account-feasibility and failure handling to avoid invalid wealth logarithms or silently deleting losing episodes. Report costs of stochastic exploration during training.

## 5. Main ideas and novelty

The proposed component is **budgeting the economic consequence of personalized risk interventions through paired self-financing accounts**. It adapts counterfactual safety to a price-taking portfolio setting and uses positive benchmark-relative shortfall instead of a gain-cancelling signed floor.

| Prior work | Established idea | Proposed distinction |
| --- | --- | --- |
| [ResSafe](https://arxiv.org/abs/2609.15988) | Frozen task policy plus learned safety residual | Budget financial shortfall against an executable policy rather than relying on small allocation changes |
| [RiPO](https://arxiv.org/abs/2306.07013) | Barrier-function portfolio risk correction | Learn personalized episode-drawdown intervention with a positive-shortfall budget |
| [Do No Harm](https://proceedings.mlr.press/v242/vaskov24a.html) | Counterfactual harm relative to an alternative policy | Adapt comparison to net portfolio growth with two cost-bearing account trajectories |
| [Regret-Optimized Portfolio Enhancement](https://arxiv.org/html/2502.02619) | Portfolio enhancement using future-looking oracle regret | Use the frozen executable policy as reference and constrain its positive shortfall |

These are positioning differences, **not proof of a first-in-literature claim**. Benchmark-relative downside measures, risk-aware RL, and paired replay are established. Check exact formulation overlap before submission. Neither modularity, transferability, Korean data, nor an RNN alone is sufficient novelty.

The empirical questions are:

- Does positive-shortfall budgeting improve the held-out drawdown–shortfall frontier over a signed mean-return floor?
- Does a frozen nominal policy plus residual add value beyond fine-tuning under the same budget?
- Does learned adaptation beat simple risk overlays after costs and cash exposure are controlled?

The stricter constraint may sacrifice more return or fail to generalize. Report these possibilities directly.

## 6. Baselines and optional market models

| Method | Purpose |
| --- | --- |
| Unchanged nominal policy | Reference performance and zero-intervention candidate |
| Residual with signed mean-return floor | Isolate the constraint change |
| Residual with positive-shortfall budget | Proposed adaptation |
| Fine-tuning with the same positive-shortfall budget | Test the need for a separate correction network |
| Capacity-matched fine-tuning of both nominal and residual networks | Separate parameter freezing from extra capacity |
| Nominal plus volatility targeting / drawdown rule | Test whether learning is necessary |
| Covariance-based minimum-deviation optimizer, if feasible | Conventional optimization comparison |
| Equal weight | Contextual allocation baseline |

Initialize adaptations from identical nominal checkpoints and zero-output residuals. Match observations, execution, costs, and tuning budgets. Report trainable parameter counts and compute for pretraining and adaptation. Sweep validation-selected budgets: identical numeric tolerances do not guarantee identical realized risk or constraint satisfaction.

An action-norm penalty is an optional ablation, not the main preservation mechanism. Calibrate exposure-matched controls on validation data. Hindsight test-return rescaling is not an executable strategy.

**Model-free core:** train with historical market replay and explicit accounting. Paired paths reduce noise in comparisons under the simulator; they do not create independent market evidence or reveal real unobserved fills.

**Optional model-based extension:** fit a regularized VAR on training data and use short scenarios with jointly resampled residuals for residual-policy training. Keep the same nominal policy/account model. A probabilistic GRU/RNN follows only if resources permit. Forecast features alone do not constitute model-based RL; generated transitions must enter optimization. Synthetic paths are not additional independent test evidence.

## 7. IntelliQuant assessment

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

## 8. Evaluation and verification

1. **Chronology:** Use training, validation, and untouched test blocks, with walk-forward folds where coverage permits. Purge episode/forecast targets crossing boundaries; past observations may initialize causal features.
2. **Preprocessing:** Fit scalers/predictors without test information. Rolling features can update from past data under fixed rules. Do not retune policies inside test blocks.
3. **Episodes:** Fix horizon, positions, cash model, reset rules, and terminal accounting. Use non-overlapping test episodes where feasible and report continuous portfolio paths too. A terminal budget does not constrain maximum interim underperformance.
4. **Metrics:** Net growth/CAGR, maximum drawdown, volatility, Sharpe, mean positive shortfall, signed mean growth gap, shortfall frequency/severity, turnover, cash exposure, and empirical budget violations.
5. **Uncertainty:** Pair comparisons on identical market windows. Use multiple seeds and moving-block resampling where appropriate, recomputing path-dependent metrics. Seeds and overlapping windows are not independent market samples. Short histories cannot support strong tail guarantees.
6. **Selection:** Choose horizon, budget grid, checkpoints, and settings using validation data. Report relevant frontiers and tuning budgets. Distinguish empirical training violations from held-out violations.
7. **Stress:** Increase costs/slippage and examine adverse held-out regimes. Test model misspecification if scenario training is added.
8. **Replication:** Reconcile local/IntelliQuant weights, trades, fees, and equity before interpreting metric discrepancies. Evaluate a distilled policy as a separate approximation.

Planned implementation checks: bypass reproduces reference trades and wealth; account states are independent; identical growth gives zero shortfall; outperformance never creates negative shortfall; terminal losses survive episode completion/failure; and maximum-drawdown costs telescope correctly. These are verification targets, not tests already implemented or passed.

## 9. Detailed plan and expected contributions

| Period | Deliverable | Decision gate |
| --- | --- | --- |
| Week 1 | Data route, fixed ETF universe, causal paired-account simulator | Adopt separately licensed data if IntelliQuant extraction is unavailable. |
| Week 2 | Nominal PPO, shadow-account feedback, conventional controls | Verify costs, bypass equality, and no future-data observations. |
| Week 3 | Signed-floor residual, proposed residual, and fine-tuning controls | Separate constraint effects from architecture effects. |
| Week 4 | Held-out frontiers, seeds, uncertainty, and cost stress | Add VAR scenarios only after the core evidence is complete. |
| Week 5 | Platform discrepancy audit, figures, and final submission | Present positive and negative findings with their limits. |

Expected contributions are the financial intervention-budget formulation, paired-account implementation, controlled empirical comparison, and documented platform feasibility/replication findings. Success needs evidence beyond increased cash exposure. If fine-tuning or simple controls match the residual policy, that limits the case for residual learning and is a finding to report.

The minimum project uses one ETF universe, one nominal PPO policy, one drawdown objective, a small budget grid, and strong controls. Transfer, CVaR optimization, many RL algorithms, and large recurrent market models are not requirements.

## 10. References and open decisions

- [Qu et al., ResSafe, arXiv:2609.15988 (2026)](https://arxiv.org/abs/2609.15988): residual safety architecture.
- [Li, Huang, and Tam, Combining Reinforcement Learning and Barrier Functions for Adaptive Risk Management in Portfolio Optimization, arXiv:2306.07013 (2023)](https://arxiv.org/abs/2306.07013): direct portfolio-controller precedent.
- [Vaskov, Schwarting, and Baker, Do No Harm, L4DC 2024](https://proceedings.mlr.press/v242/vaskov24a.html): counterfactual safety foundation.
- [Karzanov et al., Regret-Optimized Portfolio Enhancement, arXiv:2502.02619 (2025)](https://arxiv.org/abs/2502.02619): related portfolio regret method.
- [Ghosh et al., Neural-progressive hedging, UAI 2022](https://proceedings.mlr.press/v180/ghosh22a.html): additional model-guided portfolio constraint precedent.
- [Chow et al., Risk-Constrained Reinforcement Learning with Percentile Risk Criteria, JMLR](https://www.jmlr.org/papers/v18/15-636.html): broader constrained-risk formulations.
- [IntelliQuant Studio guide](https://support.intelliquant.ai/guide/iq-studio), [Stock API](https://www.intelliquant.ai/help/ref/1), [environment API](https://www.intelliquant.ai/help/ref/15), and [membership](https://www.intelliquant.ai/membership): platform evidence.

Before submission, resolve literature overlap, dataset rights, ETF symbols, date range, episode horizon, cash assumptions, and validation-selected budgets. Fill author details and verify the compiled two-page layout. The slide's novel-component requirement is real; a candidate application extension should not be presented as established originality before that review.
