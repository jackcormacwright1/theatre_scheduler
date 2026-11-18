# Methodology

This document sets out how the theatre list booking models work: what is simulated, how each quantity is calculated, how scenarios are compared, how the code was checked, and what the results can and can't support. Every symbol is defined where it first appears, and the main calculations have small worked examples.

**All data is synthetic.** Procedure times, surgeon differences, demand, leave and waiting lists are invented, with values chosen to be plausible rather than taken from any organisation. The project started as an NHS piece of work that was deprioritised before completion; this repository is a from-scratch reimplementation and contains no NHS code or data. The results describe *mechanisms and trade-offs*, not the performance of any real service.

Code: [github.com/jackcormacwright1/theatre_scheduler](https://github.com/jackcormacwright1/theatre_scheduler)

---

## 1. The questions

1. **Booking one window (Part 1).** If each case's duration is uncertain, how many more patients can be booked onto a fixed set of theatre lists, at the same risk of lists overrunning, by modelling that uncertainty with Monte Carlo simulation rather than booking to average times? What does any gain depend on? And does a hierarchical duration model, which shares information between surgeons and procedures, help the planner keep to its own overrun limit?
2. **A year of booking (Part 2).** For four specialties, how many extra theatre lists are needed to treat 95% of patients within 6 weeks of being added to the waiting list? How does the answer depend on how that target is measured? What happens to theatre utilisation?
3. **Scenario comparison (Part 3, the dashboard).** For any two configurations of demand, capacity and booking rules, what is the difference in patients treated, waiting times, lists and theatre minutes, and is it distinguishable from noise? Starting from today's waiting list, when will the target be met, and how many extra lists would meet it by a chosen week?

The 6-week target is used as a working premise: if patients are treated within 6 weeks of the decision to operate, the remaining 12 weeks of the 18-week referral-to-treatment (RTT) standard are left for everything before that decision.

---

## 2. The synthetic world

Code: [`synthetic.py`](https://github.com/jackcormacwright1/theatre_scheduler/blob/main/src/theatre_sched/synthetic.py)

### 2.1 Specialties and procedures

Each specialty has six elective procedures. For each procedure $p$ the table gives the median case time $M_p$ in minutes (patient into theatre to patient out), the spread $\sigma_p$ on the log scale (explained in 2.2), and its share of the waiting list.

| Specialty | Procedure | Median $M_p$ (min) | Log-scale spread $\sigma_p$ | Share of referrals |
|---|---|---|---|---|
| General surgery | Skin lesion excision | 25 | 0.35 | 15% |
| | Haemorrhoidectomy | 35 | 0.30 | 10% |
| | Umbilical hernia repair | 45 | 0.30 | 15% |
| | Inguinal hernia repair | 60 | 0.30 | 30% |
| | Laparoscopic cholecystectomy | 80 | 0.35 | 20% |
| | Incisional hernia repair | 110 | 0.40 | 10% |
| Orthopaedics | Carpal tunnel release | 25 | 0.30 | 15% |
| | Knee arthroscopy | 45 | 0.30 | 15% |
| | Shoulder arthroscopy | 75 | 0.30 | 10% |
| | ACL reconstruction | 100 | 0.30 | 5% |
| | Total hip replacement | 120 | 0.25 | 25% |
| | Total knee replacement | 125 | 0.25 | 30% |
| Gynaecology | Operative hysteroscopy | 25 | 0.35 | 30% |
| | Diagnostic laparoscopy | 40 | 0.35 | 15% |
| | Laparoscopic ovarian cystectomy | 80 | 0.35 | 15% |
| | Vaginal hysterectomy and repair | 100 | 0.30 | 15% |
| | Laparoscopic excision of endometriosis | 120 | 0.45 | 10% |
| | Total laparoscopic hysterectomy | 150 | 0.35 | 15% |
| ENT | Grommet insertion | 15 | 0.35 | 20% |
| | Microlaryngoscopy | 25 | 0.35 | 15% |
| | Tonsillectomy | 40 | 0.30 | 25% |
| | Septoplasty | 60 | 0.30 | 15% |
| | Endoscopic sinus surgery | 90 | 0.35 | 15% |
| | Thyroidectomy | 130 | 0.30 | 10% |

The specialties differ mainly in case mix: orthopaedics is dominated by two-hour joint replacements, ENT has many very short cases, and gynaecology mixes short day cases with long, highly variable laparoscopic operations.

### 2.2 Case durations

Case durations are **lognormal**: the logarithm of a case's duration in minutes is normally distributed. This gives right-skewed times, with most cases near the usual time and a tail of slow ones, and is the usual model for surgical case durations (Strum et al., 2000).

For surgeon $s$ doing procedure $p$, the duration $D$ of a case is

$$\log D \sim \text{Normal}(\mu_{sp},\ \sigma_p^2), \qquad \mu_{sp} = \log M_p + \beta_s + \gamma_{sp}$$

where

- $\sim \text{Normal}(a, b^2)$ means "is drawn from a normal distribution with mean $a$ and standard deviation $b$";
- $\mu_{sp}$ is the average of $\log D$ for that surgeon and procedure, so $e^{\mu_{sp}}$ is their median time;
- $\beta_s \sim \text{Normal}(0, 0.15^2)$ is surgeon $s$'s overall speed: $\beta_s = 0.15$ means about 16% slower than the typical surgeon on every procedure (because $e^{0.15} = 1.16$);
- $\gamma_{sp} \sim \text{Normal}(0, 0.08^2)$ is a small surgeon-specific deviation for that procedure;
- $\sigma_p$ is the case-to-case spread from the table.

Two facts about the lognormal are used throughout. The **median** is $e^{\mu}$, and the **mean** is $e^{\mu + \sigma^2/2}$, slightly above the median because of the long right tail.

> **Worked example.** A procedure with median 60 minutes and $\sigma = 0.30$:
> mean $= 60 \times e^{0.30^2/2} = 60 \times e^{0.045} = 62.8$ minutes;
> 90th percentile $= 60 \times e^{1.2816 \times 0.30} = 60 \times e^{0.384} = 88.1$ minutes
> (1.2816 is the 90th percentile of the standard normal distribution). So one case in ten takes at least 47% longer than the median.

Each surgeon performs between four and six of their specialty's six procedures, chosen at random.

### 2.3 What the planner can see: historical cases

The true parameters are hidden. The planner sees only past cases. The number of past cases for each surgeon–procedure pair is

$$n_{sp} = \max\big(1,\ \text{round}(e^{Z})\big), \qquad Z \sim \text{Normal}(\log 15,\ 1^2)$$

so the median pair has 15 past cases, but some have one or two and some over a hundred. This variation matters: estimates based on three cases are much less reliable than estimates based on a hundred.

### 2.4 Random number streams

Each world is generated from a seed. The surgeon parameters, the history and the waiting list each draw from their own random stream (via `numpy.random.SeedSequence.spawn`). Changing one setting, such as the amount of history, therefore leaves everything else identical, so differences between scenarios reflect the setting rather than a reshuffled world.

---

## 3. The planner's duration models

Code: [`durations.py`](https://github.com/jackcormacwright1/theatre_scheduler/blob/main/src/theatre_sched/durations.py)

**Plug-in lognormal.** For each surgeon–procedure pair with past durations $x_1, \dots, x_n$:

$$\hat\mu_{sp} = \frac{1}{n}\sum_{k=1}^{n} \log x_k, \qquad \hat\sigma_{sp} = \text{standard deviation of } \log x_1, \dots, \log x_n$$

The hat ($\hat{\ }$) marks an estimate. If a pair has fewer than 3 cases, its own standard deviation is too unstable, and the procedure's **pooled** spread is used instead:

$$\hat\sigma_p = \sqrt{\frac{\sum_{\text{pairs}} (n-1)\,\text{sd}^2}{\sum_{\text{pairs}} (n-1)}}$$

pooled over the pairs for that procedure with at least 2 cases.

The planner then treats $\hat\mu$ and $\hat\sigma$ as if they were exact. That is what "plug-in" means: the uncertainty in the estimates themselves is ignored. Part 1 shows the consequences.

**Historical average.** The booking-to-averages rule uses the plain arithmetic mean of each pair's past durations, $\bar m_{sp} = \frac{1}{n}\sum x_k$: the "average time it takes that surgeon".

**Oracle.** For reference, an oracle planner uses the true $\mu_{sp}, \sigma_p$. It shows what perfect knowledge of durations would be worth.

### 3.1 Hierarchical model (partial pooling)

Code: `fit_hierarchical` in [`durations.py`](https://github.com/jackcormacwright1/theatre_scheduler/blob/main/src/theatre_sched/durations.py)

The plug-in model's weakness is that a pair's average from two cases is treated as exactly as reliable as one from two hundred. The hierarchical model (Gelman and Hill, 2007) treats each pair's true average $\theta_{sp}$ as uncertain, and estimates it from two sources: the pair's own cases, and what the rest of the data says about a surgeon of that speed doing that procedure.

**The model**, on the log scale ($y = \log$ of a case time in minutes):

$$y_{spk} = \theta_{sp} + \sigma_p\,\varepsilon_{spk}, \qquad \varepsilon_{spk} \sim \text{Normal}(0, 1)$$

$$\theta_{sp} \sim \text{Normal}\big(m_{sp},\ \tau^2\big), \qquad m_{sp} = \alpha_p + \beta_s$$

- $k$ indexes a pair's past cases. $\theta_{sp}$ is the pair's true average log time and $\sigma_p$ how much single cases of procedure $p$ vary around it.
- $m_{sp}$ is the **prior mean**: what you'd expect before seeing the pair's own cases. $\alpha_p$ is procedure $p$'s typical log time for an average surgeon, and $\beta_s$ is surgeon $s$'s speed (negative is faster), estimated from all their procedures, with the $\beta_s$ summing to zero.
- $\tau$ is how far pairs' true averages scatter around $m_{sp}$: the surgeon-specific part that the procedure and the surgeon's general speed don't explain.

**Fitting** is empirical Bayes by the method of moments (no MCMC), repeated until $\tau$ settles:

1. $\hat\sigma_p$ is the pooled within-pair spread, as in the plug-in model.
2. Fit the pair averages $\bar y_{sp}$ to $\alpha_p + \beta_s$ by weighted least squares, with weights $1/(\tau^2 + \hat\sigma_p^2/n_{sp})$: a pair's average is noisier the fewer cases it rests on.
3. Estimate $\tau^2$ from the residuals $r_{sp} = \bar y_{sp} - \hat m_{sp}$: their spread is $\tau^2$ plus sampling noise $\hat\sigma_p^2/n_{sp}$, so $\hat\tau^2 = \text{mean}\big(r_{sp}^2 \cdot N/(N-q) - \hat\sigma_p^2/n_{sp}\big)$, floored at $0.01^2$. $N$ is the number of pairs and $q$ the number of fitted coefficients.

**Each pair's estimate** is then a weighted average of its own average and the prior mean:

$$w_{sp} = \frac{V_{sp}}{V_{sp} + \hat\sigma_p^2 / n_{sp}} = \frac{n_{sp}}{n_{sp} + k_{sp}}, \qquad k_{sp} = \frac{\hat\sigma_p^2}{V_{sp}}$$

$$\hat\theta_{sp} = w_{sp}\,\bar y_{sp} + (1 - w_{sp})\,\hat m_{sp}, \qquad v_{sp} = w_{sp}\,\frac{\hat\sigma_p^2}{n_{sp}}$$

$V_{sp} = \hat\tau^2 + \text{se}(\hat m_{sp})^2$ is the prior variance: $\tau^2$ plus the uncertainty in the estimated prior mean itself. $w_{sp}$ is the weight on the pair's own data, and $k_{sp}$ is the number of cases at which the pair's own data and the prior get equal weight. $\hat\theta_{sp}$ is the posterior mean and $v_{sp}$ the posterior variance: how uncertain the pair's average still is.

> **Worked example.** A procedure's prior mean for a surgeon of this speed is 60 minutes. Single cases vary with $\sigma = 0.30$ on the log scale, and pairs scatter around the prior with $\tau = 0.10$, so $k = 0.30^2/0.10^2 = 9$ (ignoring the small uncertainty in the prior mean).
> - The surgeon has done the procedure **twice**, averaging 75 minutes. Then $w = 2/(2 + 9) = 0.18$, and $\hat\theta = 0.18\log 75 + 0.82\log 60$, which is **62.5 minutes**, much nearer 60 than 75. The posterior standard deviation is $\sqrt{0.18 \times 0.09/2} = 0.09$, so a 90% interval for the pair's typical time is about 54 to 73 minutes.
> - With **twenty** cases averaging 75 minutes, $w = 20/29 = 0.69$ and the estimate is **70.0 minutes** (90% interval about 64 to 77). The more evidence the pair has, the more it speaks for itself.

**Using it to plan.** The Monte Carlo draws carry the remaining uncertainty. In each draw $r$, the pair's average is drawn once, $\theta_{sp}^{(r)} \sim \text{Normal}(\hat\theta_{sp}, v_{sp})$, and shared by all of that pair's patients, whose case times are then $\exp\big(\theta_{sp}^{(r)} + \hat\sigma_p\,\varepsilon\big)$. Sharing matters: if a surgeon is in fact slower at a procedure than the data suggest, all four of their cases of it on one list run slow together, and the list total moves by more than independent errors would allow. A pair with no history at all gets the prior, $\theta_{sp} \sim \text{Normal}(\hat m_{sp}, V_{sp})$.

With no surgeon effect ($\beta_s = 0$), the prior mean is just the procedure's average across surgeons and $\tau$ also absorbs differences in surgeon speed. The surgeon effect is on by default because it shrinks towards a better prior: in 30 synthetic worlds with a median of 15 past cases per pair, the error in the pair averages (root mean square, log scale) was 0.112 from each pair's own cases, 0.095 with partial pooling towards the procedure average, and 0.084 with the surgeon effect.

---

## 4. Evaluating a list with Monte Carlo simulation

Code: [`evaluate.py`](https://github.com/jackcormacwright1/theatre_scheduler/blob/main/src/theatre_sched/evaluate.py)

A **list** $j$ is one theatre session of length $L_j$ minutes (480 for an all-day list) with $n_j$ booked cases, separated by changeovers of $c$ minutes (15 by default). Draw $S$ simulated durations for every patient, written $D_i^{(r)}$ for patient $i$ in draw $r = 1, \dots, S$. The total time the list takes in draw $r$ is

$$Z_j^{(r)} = \sum_{i \in j} D_i^{(r)} + c\,(n_j - 1)$$

and the Monte Carlo estimates are averages over draws:

| Quantity | Estimate |
|---|---|
| Chance the list overruns | $\hat P_j = \frac{1}{S}\sum_{r} \mathbf{1}\big[Z_j^{(r)} > L_j\big]$ |
| Monte Carlo standard error of $\hat P_j$ | $\sqrt{\hat P_j(1-\hat P_j)/S}$ |
| Expected overtime | $\frac{1}{S}\sum_r \max\big(Z_j^{(r)} - L_j,\ 0\big)$ |
| Utilisation counting changeovers | $\frac{1}{S}\sum_r \min\big(Z_j^{(r)}, L_j\big) / L_j$ |

$\mathbf{1}[\cdot]$ is 1 when the condition is true and 0 otherwise, so $\hat P_j$ is simply the share of draws in which the list runs past its end.

> **Worked example.** A three-case list on an 8-hour (480-minute) session. Suppose five draws give totals, changeovers included, of 431, 468, 495, 452 and 503 minutes. Two of the five exceed 480, so $\hat P = 2/5 = 40\%$. In practice $S = 500$ to $2{,}000$ draws are used. At a true risk of 5% with $S = 2{,}000$, the standard error is $\sqrt{0.05 \times 0.95 / 2000} = 0.005$, about half a percentage point.

**Common random numbers.** The draws are made once per patient and reused for every candidate list. Two candidate lists are therefore compared on the same simulated futures, and the difference between them reflects the lists rather than fresh simulation noise.

**Touch-time utilisation.** The national 85% theatre utilisation target counts *touch time*: the time a patient is in theatre, within the planned session, excluding changeovers and anything after the session end (GIRFT's capped touch-time utilisation; see the 2024 Cureus paper in the references). Cases run back to back from minute 0. Case $k$ on the list starts at $t_k$ and ends at $t_k + D_k$, with $t_1 = 0$ and $t_{k+1} = t_k + D_k + c$. Then

$$\text{touch utilisation} = \frac{1}{L}\sum_{k} \max\Big(0,\ \min(t_k + D_k,\ L) - t_k\Big)$$

> **Worked example.** Three 100-minute cases, 20-minute changeovers, a 300-minute session. They run 0–100, 120–220 and 240–340. Inside the session that is 100 + 100 + 60 = 260 minutes of touch time, so touch utilisation is 260/300 = 86.7%. Utilisation *counting* changeovers would be min(340, 300)/300 = 100%. The two measures differ by the changeover time, which is why lists of many short cases score badly on touch time.

---

## 5. Part 1: booking one four-week window

Code: [`schedule.py`](https://github.com/jackcormacwright1/theatre_scheduler/blob/main/src/theatre_sched/schedule.py), [`scripts/run_experiment.py`](https://github.com/jackcormacwright1/theatre_scheduler/blob/main/scripts/run_experiment.py)

**Setting.** There are 5 surgeons, each with 60 patients on their own waiting list and one all-day list a week for four weeks. Patients have a clinical priority (P2: 10%, P3: 40%, P4: 50%), assumed targets $g_i$ of 28, 90 and 180 days, and days already waited $a_i$, drawn from exponential distributions with means 14, 60 and 120 days.

**Value of booking a patient.**

$$u_i = \frac{a_i}{g_i}, \qquad v_i = 1 + \lambda\,u_i$$

$u_i$ is **urgency**, the fraction of the target already used, which exceeds 1 once the patient is past target. $v_i$ is the patient's **weight**: the 1 rewards booking anyone at all (throughput), and $\lambda u_i$ (with $\lambda = 1$) rewards booking those who have waited longest relative to their target.

**Must-book patients.** Every P2 patient, and anyone with $u_i \ge 1$, is placed first, most urgent first. The methods differ only in how they fill the remaining space.

> **Worked example.** Patient A has waited 45 days against a 90-day target, so $u = 0.5$ and $v = 1.5$, with an expected case time of 50 minutes. Patient B has waited 30 days against 90, so $u = 0.33$ and $v = 1.33$, with 120 minutes. Their weight per expected minute is 0.030 and 0.011. "Ratio order" therefore offers A's case first: more value per minute of theatre time.

**Book to average.** Assume each case takes its historical average $\bar m_i$, and fill each list while

$$\sum_{i \in j} \bar m_i + c\,(n_j - 1) \le f\,L_j$$

where $f$ is the **fill fraction**: 1.0 books to the end of the session, and 0.85 leaves a 15% buffer. Spare space is filled either in *queue order* (priority, then longest wait) or in *ratio order* (highest $v_i / \bar m_i$). Each patient goes on their surgeon's earliest list where they fit.

**Chance-constrained booking.** Choose which patients go on each list to make the total weight $\sum v_i$ as large as possible, subject to each list's estimated overrun risk being at most $\alpha$:

$$\hat P_j \le \alpha \quad \text{for every list } j$$

This is a chance constraint (Charnes and Cooper, 1959), estimated by the Monte Carlo method in section 4. The search is a heuristic:

1. **Insert.** Go through unbooked patients, must-book first and then the rest in ratio order, placing each on the earliest list where $\hat P_j \le \alpha$ still holds.
2. **Swap.** For each unbooked patient, try replacing a booked optional patient on one of the surgeon's lists. Accept the swap if the constraint still holds and it either brings in a must-book patient or raises the total weight.
3. **Repeat** until nothing changes. Each accepted move books more must-book patients or raises the total weight, both of which are bounded, so the search always ends.

**Evaluation.** There are 20 synthetic worlds. Planning uses $S = 2{,}000$ draws from the planner's model. Every schedule is then scored on $S = 20{,}000$ fresh draws from the **true** distributions, which no planner sees. Two summaries matter:

- **Matched-risk comparison.** Within each world, the book-to-average curve of patients booked against realised overrun rate (one point per fill fraction) is linearly interpolated at the overrun rate a chance-constrained schedule achieved. The difference is "extra patients at the same overrun rate", averaged across worlds with its standard error. Rates outside the baseline's range are not extrapolated.
- **Calibration.** For a target $\alpha$, compare the realised overrun probability of each list with $\alpha$, both on average and at the 90th percentile across lists (the riskiest tenth).

**Duration models.** The chance-constrained search is run with the plug-in model, the hierarchical model (section 3.1) and the true durations, on the same patients and lists.

**Sensitivity.** The comparison is rerun with a median of 3, 15, 50 and 200 past cases per pair, holding everything else in each world fixed (section 2.4; [`scripts/run_history_sweep.py`](https://github.com/jackcormacwright1/theatre_scheduler/blob/main/scripts/run_history_sweep.py)).

---

## 6. Part 2: a year of booking, week by week

Code: [`rolling.py`](https://github.com/jackcormacwright1/theatre_scheduler/blob/main/src/theatre_sched/rolling.py), [`scripts/run_capacity.py`](https://github.com/jackcormacwright1/theatre_scheduler/blob/main/scripts/run_capacity.py)

### 6.1 Time, surgeons and leave

Time runs in days. A **settled** simulation (used for all of Part 2) is 26 weeks of **warm-up**, discarded, followed by 52 **measured** weeks. A simulation **from today's waiting list** (section 6.8) has no warm-up and measures every week from the start. Each of the $N$ surgeons (6 by default) has $\ell$ all-day lists a week (1 by default) on a fixed weekday and their own waiting list.

**Leave.** Each surgeon operates $W$ weeks a year (42 by default). The remaining $52 - W$ weeks are taken as $(52 - W)/2$ non-overlapping two-week blocks at random points in each year, and no lists run while a surgeon is on leave.

### 6.2 Demand

Patients are added to the list by a **Poisson process**: in each week the number of new patients is random, with a fixed average rate and no memory of previous weeks. There are two ways of setting the rate.

- **Referrals** (the dashboard). $R$ patients a week for the whole specialty, shared equally between the surgeons, so each receives $R/N$ a week. Adding surgeons or lists adds capacity, not demand.
- **Load** (the Part 2 and Part 3 scripts). Each surgeon receives enough patients to fill a share $\rho$ (0.85 by default) of the session time they have available after leave, so demand scales with the number of surgeons. These scripts never vary the number of surgeons.

The two are linked by the share of available theatre time the demand needs:

$$\rho = \frac{R \times \bar m}{N \times \ell \times L \times W/52}$$

where $\bar m$ is the average theatre minutes per case, including a 15-minute changeover, across the surgeons' case mixes.

> **Worked example.** General surgery, 6 surgeons, one 480-minute list a week each, 42 operating weeks. Available time is $6 \times 480 \times 42/52 = 2{,}326$ minutes a week. At $R = 24$ referrals a week and about 82.6 minutes per case, demand needs $24 \times 82.6 = 1{,}982$ minutes, so $\rho = 1{,}982/2{,}326 = 85\%$. The dashboard shows this figure under the referrals setting.

Demand is fixed in **patients**, calculated at a reference 15-minute changeover, so a scenario with faster changeovers sees the same patients rather than more of them. Each new patient gets a procedure (by the referral shares for their surgeon's procedures) and an urgent (P2) flag with probability 10%. Each patient also gets one true duration, kept if they're cancelled and rebooked, and 500 planner draws. Every patient who will ever be on the list is generated up front from the seed. That way, every booking policy and every number of extra lists sees exactly the same patients.

**Starting list.** For a settled simulation, three weeks' worth of patients, with dates added spread evenly over the six weeks before the start. The warm-up then erases it. For a start from today's waiting list, see section 6.8.

### 6.3 Targets

The target is $g = 42$ days (6 weeks) from being added to the list to surgery, or 28 days for urgent patients. A patient is **due** at a booking round if, not booked now, they could pass their target before the next round:

$$a_i + 13 \ge g_i$$

13 days is the longest gap between this week's booking and a list next week.

### 6.4 Booking each week

At the start of each week, every patient added before that week can be booked. For each surgeon, a **booking order** ranks their eligible patients:

| Policy | Order |
|---|---|
| `in_turn` | Urgent patients first, then longest wait first. With the average rule, this is first come, first served. |
| `ratio_fill` | Due patients first, longest wait first. Then the rest by $v_i / \hat m_i$ with $v_i = 1 + a_i/g_i$, which favours short cases. |
| `touch_fill` | Due patients first, longest wait first. Then the rest by longest expected case, which maximises touch time per changeover. |

Each patient in that order goes on the earliest of their surgeon's lists this week where they fit under the **booking rule**:

- **To average case times** (`average`): $\sum \bar m_i + c\,(n_j - 1) \le f L$, using the historical averages from section 3. The default is $f = 1.0$.
- **To a Monte Carlo overrun limit** (`monte_carlo`): $\hat P_j \le \alpha$, using the planner's 500 draws per patient. The default is $\alpha = 0.10$.

The same expected durations (historical averages for the average rule, plug-in means for the Monte Carlo rule) are used for ordering and on the day.

**Extra lists.** $E$ extra all-day lists a week can be fractional. The number in week $t$ is $\lfloor E(t+1) \rfloor - \lfloor E t \rfloor$, where $\lfloor x \rfloor$ means rounding down, so $E = 0.25$ gives one list every fourth week. They run on Saturdays and go one at a time to the working surgeon with the most expected minutes of due patients. After each allocation, that surgeon's due minutes are reduced by 80% of a list.

### 6.5 On the day

Each list runs its cases longest-expected first, using **true** durations. Before each case starts, the team checks whether it is expected to finish more than 30 minutes after the session end (the **cancellation tolerance**). If so, or if the session has already ended, the case is cancelled and the patient returns to the waiting list with their clock still running, as under RTT rules. A list **overruns** if its last case finishes after the session end.

### 6.6 Measures

Three measures of the 6-week target:

| Measure | Definition | Can be flattered by |
|---|---|---|
| **Still waiting** (how the national RTT standard is measured) | At the end of each week, of the patients still waiting, the share who have waited no more than 42 days. Averaged over the 52 measured weeks. | Holding patients who are within target, because they count in its favour while they wait |
| **Treated** | Of the patients operated on in the measured weeks, the share who had waited no more than 42 days | Treating short waiters first |
| **Added to list** (cohort) | Of the patients added during the measured weeks, at least 42 days before the end, the share operated on within 42 days | Nothing short of treating people on time |

> **Worked example: why the still-waiting measure can mislead.** 1,000 patients are waiting and 920 are within target: 92.0%. Treating 50 who have waited 3 weeks leaves 870 within target out of 950, which is 91.6%. Treating 50 who have waited 30 weeks instead leaves 920 out of 950, which is 96.8%. The work is the same, but the measure moves in opposite directions, because a treated patient leaves both the top and the bottom of the fraction.

Other measures, all over the measured weeks:

- touch-time utilisation (section 4), averaged over every list held, including any with no patients booked;
- the share of lists that **ran short of patients**, meaning every eligible patient of that surgeon was booked, so the list couldn't be filled further from its own queue. Touch utilisation is also reported for the remaining **full** lists;
- the overrun rate, overtime minutes, and cancellations per 100 booked;
- patients treated, the median and 95th-percentile wait of those treated, waiting list size and its change over the year;
- annual figures: weekly averages × 52, for patients treated, lists run, session minutes (lists × $L$), touch minutes and overtime minutes.

### 6.7 Extra lists needed

For a scenario, the simulation is rerun at $E = 0, 0.25, 0.5, \dots, 3$ extra lists a week, on the same patients. For each seed and each measure, the fewest extra lists reaching the threshold (95%) is found by **linear interpolation** between the two grid points either side:

$$E^{\ast} = E_{k-1} + \frac{0.95 - y_{k-1}}{y_k - y_{k-1}}\,(E_k - E_{k-1})$$

where $y_k$ is the measure at grid point $E_k$, and $k$ is the first point at or above 95%.

> **Worked example.** Suppose the measure is 93.4% at 0.25 extra lists a week and 96.2% at 0.5. Then $E^{\ast} = 0.25 + \frac{0.950 - 0.934}{0.962 - 0.934} \times 0.25 = 0.25 + 0.571 \times 0.25 = 0.39$ lists a week, about 20 lists a year.

The reported figure is the mean across seeds. If any seed doesn't reach 95% within 3 extra lists a week, the scenario is reported as not reaching it. The Part 2 tables use the coarser rule of the first grid point whose mean across seeds reaches 95%.

**Scenarios run (Part 2).** For each of the four specialties: extra lists 0–2 in steps of 0.25, three booking policies, changeovers of 15 or 5 minutes, and overrun limits of 10% or 30%. That is 10 worlds each, with demand set by load $\rho = 0.85$.

### 6.8 Starting from today's waiting list

A settled simulation answers "where do these settings lead?", but deliberately forgets where the list starts. A service with a backlog needs the other question: **given the list we have now, what happens from here?** Setting a starting list answers it.

**The starting list** can be described in three ways, from least to most detail.

*Totals.* Three numbers: $S$, the patients waiting now (whole specialty); $q$, the share who have already waited longer than the target $T$; and $D$, the longest wait now. The $S$ patients are shared evenly between the surgeons (any remainder going to the first surgeons), and each gets a procedure and urgency flag as new referrals do. Their waits so far are drawn as follows:

- with probability $1 - q$, the wait is spread evenly between 1 day and $T$, as it would be for a list fed by steady referrals;
- with probability $q$, the wait is beyond the target, thinning out evenly to nobody at the longest wait: a **triangular** density on $(T, D]$, highest at $T$. It is sampled by inverting its cumulative distribution: with $u$ uniform on $(0, 1)$, $\text{wait} = T + (D - T)\,(1 - \sqrt{1 - u})$.

> **Worked example.** $T = 42$ days and $D = 210$ days (30 weeks). A patient over target with $u = 0.5$ has waited $42 + 168 \times (1 - \sqrt{0.5}) = 42 + 168 \times 0.293 = 91$ days, 13 weeks. Half of the over-target patients have waited less than that, because the density is highest just past the target. The mean over-target wait is a third of the way from $T$ to $D$: $42 + 168/3 = 98$ days.

*Surgeon by surgeon.* The same three numbers for each surgeon, $S_k$, $q_k$ and $D_k$, drawn the same way. Because each surgeon only treats their own list, this is what shows the effect of a backlog concentrated on one or two consultants.

*Counts by weeks waited*, as in a waiting-list extract: lines of `weeks, patients`, or `surgeon, weeks, patients`, pasted or loaded from a CSV ([`starting_list.py`](https://github.com/jackcormacwright1/theatre_scheduler/blob/main/src/theatre_sched/starting_list.py)). Weeks can be a band (`12-18`, meaning at least 12 and under 18), a whole number of weeks (`7`, meaning 7 to 8) or open-ended (`52+`, taken as 52 to 78 weeks). Within a band, waits are spread evenly. Surgeon names are matched, in the order they first appear, to the simulated surgeons S1, S2 and so on; any simulated surgeons left over start with an empty list. Without a surgeon column, each band is shared between the surgeons at random with equal chances. The share past the target, shown when the counts are read, counts a band that straddles the target in proportion: with $T$ = 6 weeks, half of a 4–8 week band counts as past it.

Whichever way it's described, the starting list uses its own random number stream, so changing it leaves the new referrals unchanged, and two scenarios with the same starting settings start from exactly the same patients.

**Measures from today.** As well as the measures in section 6.6 (now averaged over every week from today), the run reports:

- **when the target is first met**: the first week from which the share still waiting within $T$ is at or above the threshold (95%) for 4 weeks running. Requiring it to hold avoids counting a single lucky week. A run that never manages it within the simulated weeks has no value, and the dashboard says in how many runs it was met;
- **patients over target** today, after 13 and 26 weeks, and at the end;
- **the waiting list** today and at the end, set against the size a settled run with the same settings holds;
- **when everyone on today's list has been treated**, or the share treated by the end if some haven't;
- **where it ends up**: the share still waiting within target and the share treated within target over the last 13 weeks.

**Extra lists to meet the target by a chosen week.** The simulation is rerun at $E = 0, 0.25, \dots, 3$ extra lists a week on the same patients, recording the weeks until the target is met at each. For each run, the extra lists needed to meet it by week $W$ are found by linear interpolation in weeks-to-target between the first grid point that meets the deadline, $E_k$ with $t_k \le W$, and the one before, $E_{k-1}$ with $t_{k-1} > W$:

$$E^{\ast} = E_{k-1} + \frac{t_{k-1} - W}{t_{k-1} - t_k}\,(E_k - E_{k-1})$$

If the run never meets the target at $E_{k-1}$, $E^{\ast} = E_k$. The reported figure is the mean across runs, or "more than 3 a week" if any run can't meet the deadline within the sweep.

> **Worked example.** In one run, the target is met at week 35 with 0.25 extra lists a week and at week 24 with 0.5. To meet it by week 26: $E^{\ast} = 0.25 + \frac{35 - 26}{35 - 24} \times 0.25 = 0.25 + 0.818 \times 0.25 = 0.45$ extra lists a week.

**Findings this view brings out.**

- Booking in turn clears the longest waiters first, so everyone on today's list is treated well before the target is met: new referrals queue behind the backlog and pass the target while it clears. With the dashboard's defaults (general surgery, 6 surgeons, 24 referrals a week, 0.5 extra lists, booking to averages; 200 patients waiting, 35% over 6 weeks, the longest at 30), today's list is all treated by about week 10, but 95% within 6 weeks isn't held until about week 17, ranging from week 11 to 21 across five runs.
- Where the backlog sits matters. Putting 100 of the 200 patients on one surgeon (70% of them past 6 weeks) and 20 recent referrals on each of the others leaves about 31 patients over target after 13 weeks, against 13 when the same backlog is shared evenly. The target is met at about week 23 rather than 17, and today's list takes until about week 14 to clear rather than week 10 (five runs each).

---

## 7. Part 3: comparing scenarios

Code: [`scenarios.py`](https://github.com/jackcormacwright1/theatre_scheduler/blob/main/src/theatre_sched/scenarios.py), [`app.py`](https://github.com/jackcormacwright1/theatre_scheduler/blob/main/app.py), [`scripts/run_differentials.py`](https://github.com/jackcormacwright1/theatre_scheduler/blob/main/scripts/run_differentials.py)

**Pairing.** Run $r$ of a scenario uses world seed $1000 \times (\text{specialty index}) + r$, so run $r$ of A and run $r$ of B share the same surgeons, history and leave. Where the demand settings match, they also share the same patients. The comparison is then made **run by run**: for each measure, $d_r = B_r - A_r$.

**Uncertainty.** With $n$ runs, the mean difference is $\bar d = \frac{1}{n}\sum d_r$ and its standard error is $\text{sd}(d)/\sqrt{n}$. The dashboard reports $\bar d \pm 2$ standard errors, roughly a 95% interval, and marks a difference as **clear** when $|\bar d|$ exceeds two standard errors. Pairing removes the variation that A and B share, such as one world having slower surgeons, so it detects much smaller differences than comparing two unpaired averages would.

> **Worked example.** Five runs give differences in patients treated of 3, 1, 4, 0 and 2. The mean is 2. The deviations from it are 1, −1, 2, −2 and 0, so the sample standard deviation is $\sqrt{(1+1+4+4+0)/4} = 1.58$. The standard error is $1.58/\sqrt 5 = 0.71$, and two standard errors are 1.41. Since 2 > 1.41, the difference is clear: "2 more patients a year (± 1.4)".

**Extra lists needed** are compared in the same paired way, using each run's interpolated $E^{\ast}$ (section 6.7) multiplied by 52.

**Matching overrun rates.** A cautious scenario looks worse on waiting times simply because it books less, so the dashboard can retune B's caution to match A's overrun rate. It tries overrun limits $\alpha \in \{2, 5, 10, 15, 20, 25, 30, 40, 50\}\%$ (Monte Carlo rule) or fill fractions $f \in \{70, 75, \dots, 110\}\%$ (average rule), and keeps the setting whose mean overrun rate is closest to A's. If B cannot overrun as often as A even at the loosest setting, the closest is used and the remaining gap is shown.

**The written summary** reports patients treated, the share added to the list and treated within target, and the overrun rate. When both scenarios start from today's waiting list, it leads with when each meets the target and adds the patients over target after 26 weeks, and the table adds the measures in section 6.8. When requested, it adds extra lists needed. Two scenarios with different starting points (one settled, one from today's list) aren't compared. Any difference within two standard errors is reported as "no clear difference".

**Performance.** One simulated year takes about 0.1 seconds. A scenario with 5 runs takes under a second, and a full extra-lists sweep takes 5–10 seconds. Results are cached by their settings.

---

## 8. Checking the work

Code: [`tests/`](https://github.com/jackcormacwright1/theatre_scheduler/tree/main/tests) (42 automated tests, run with `pytest`)

- **The Monte Carlo against a known answer.** For a single lognormal case, the simulated overrun probability matches the exact tail probability within 4 Monte Carlo standard errors.
- **Arithmetic of lists.** Changeovers are counted between cases only. Fixed durations give certain outcomes. Touch time excludes changeovers and overrun (the worked example in section 4 is a test).
- **The duration fit.** With plenty of data, the plug-in estimates recover the true parameters. Sparse pairs borrow the pooled spread.
- **The hierarchical model.** With plenty of data it agrees with the plug-in estimates. A pair with one unusual case is pulled towards the prior, lands between its own average and the prior, and keeps a wide posterior. Draws for two patients of the same pair are correlated, and draws for different pairs aren't. A pair with no history gets the prior. Across synthetic worlds, $\hat\tau$ recovers the true surgeon–procedure deviation (0.08), and is larger without the surgeon effect, as it should be. Choosing a duration model leaves the patients and their true case times unchanged.
- **Starting from today's list.** The starting waits have the share over target asked for, none beyond the longest wait, and the triangular mean beyond the target. The week the target is met needs 4 weeks in a row (a single good week doesn't count). A run starts with exactly the list asked for, however it is described, and settled runs give exactly the same results as before the option was added. More extra lists leave fewer patients over target after 26 weeks, and a backlog concentrated on one surgeon clears more slowly than the same backlog shared. Pasted counts are read in the formats listed above, a band straddling the target counts in proportion, and an extract with more surgeons than the scenario is refused with a message saying how many are needed.
- **Schedulers.** No patient is booked twice, or on another surgeon's list. Every chance-constrained list meets its limit on the planning draws, and booking to averages respects the fill fraction.
- **The year-long simulation.**
  - No patient is operated on before being added.
  - Booked equals operated plus cancelled on every list.
  - No list runs while its surgeon is on leave, and leave is 10 weeks a year in two-week blocks.
  - The same patients are generated whatever the policy or number of lists.
  - More lists never make waiting times worse (to within noise).
  - Fixed referrals give the same demand whatever the number of surgeons.
- **Comparison.** A scenario compared with itself gives zero difference and "no clear difference". Matching moves B's overrun rate towards A's. The interpolation formula is checked directly.
- **Reproducibility.** A clean install reproduced the published Part 1 and Part 2 tables exactly, and the dashboard runs from `requirements.txt` alone.

Three investigations during development changed the model or its interpretation:

1. **Was the plug-in planner's poor calibration a bug, or small samples?** Rerunning Part 1 with 200 past cases per pair made the plug-in planner match the true-durations planner. That confirmed small samples.
2. **Abrupt waiting-time thresholds.** Without leave, waiting times jumped from failing to meeting every target within half a list a week, which is unrealistic. Adding two-week leave blocks (a 42-week operating year) gave the gradual curves reported.
3. **Unexpectedly low utilisation.** Rough arithmetic suggested higher touch utilisation than the simulation produced. Tracing it showed lists running short of patients, because each list can only be filled from its own surgeon's queue. The "short of patients" measure was added to report it.
4. **Are the hierarchical model's intervals honest?** Across 30 worlds, the 90% posterior intervals for the pair averages contained the true value 80% of the time with a median of 3 past cases per pair, 83% with 15, 87% with 50 and 89% with 200. They are somewhat too narrow when data are sparse, because the fit treats $\hat\sigma_p$ and $\hat\tau$ as known. This is one reason the hierarchical planner improves on the plug-in planner's calibration without reaching the true-durations planner's (see the README).

---

## 9. Assumptions and limitations

| Assumption | Why | Likely effect |
|---|---|---|
| All parameters are synthetic and illustrative | No real data is used | The results show mechanisms and directions, not magnitudes for any real service |
| Lognormal durations, independent between cases | A standard model for surgical times | Real lists have shared effects (a late start, a junior team), so overruns are probably more likely than modelled |
| Patients wait for their own consultant (no pooling) | A common arrangement | This is the main cause of lists running short of patients. Pooled lists would raise utilisation and cut the lists needed |
| Steady demand, with no seasonality or winter cancellations | Keeps the comparison clean | Real capacity needs are higher and lumpier |
| The planner's estimates are fixed at the start and never updated | Simplicity | Understates how much the planner would learn over a year |
| Case time is patient in theatre to patient out, with fixed changeovers | Simplicity | Touch utilisation is lower than many trusts report. The 5-minute changeover scenario approximates parallel anaesthetic rooms |
| 6 weeks from the decision to operate stands in for 18 weeks from referral | A working premise | Doesn't model outpatient or diagnostic waits |
| The cancellation rule uses the expected finish with a 30-minute tolerance | A plausible operational rule | Different tolerances trade cancellations against overtime |
| Booking happens weekly with patients eligible the week after being added | Represents pre-assessment and notice | Sets a floor of about 1–2 weeks on waits |
| Unless set surgeon by surgeon, or by surgeon in pasted counts, today's list is shared evenly between surgeons | Three numbers are often all that's to hand | A real backlog concentrated on one or two surgeons takes longer to clear: set it surgeon by surgeon to see how much |
| Within a band of pasted counts, and within the target in the totals description, waits are spread evenly | Extracts give counts by band, not exact dates | Small effect on when patients pass the target; narrower bands remove it |

---

## 10. Parameter reference

| Parameter | Default | Where |
|---|---|---|
| Surgeon speed spread (log scale) | 0.15 | World |
| Surgeon–procedure deviation (log scale) | 0.08 | World |
| Past cases per pair | median 15 (log-scale spread 1.0) | World |
| Procedures per surgeon | 4–6 of 6 | World |
| Session length $L$ | 480 minutes | All parts |
| Changeover $c$ | 15 minutes | All parts |
| Part 1 surgeons, patients each, weeks | 5, 60, 4 | Part 1 |
| Part 1 priorities and targets | P2/P3/P4: 10/40/50%, targets 28/90/180 days | Part 1 |
| Weight on urgency $\lambda$ | 1 | Part 1 |
| Fill fractions; overrun limits | 75–100%; 2, 5, 10, 20, 30% | Part 1 |
| Planning and evaluation draws | 2,000 and 20,000 | Part 1 |
| Surgeons $N$, lists per surgeon per week $\ell$ | 6, 1 | Parts 2–3 |
| Operating weeks a year $W$ | 42 (leave in two-week blocks) | Parts 2–3 |
| Demand | load 0.85 (scripts); 24 referrals a week (dashboard) | Parts 2–3 |
| Urgent share; targets | 10%; 28 days urgent, 42 days routine | Parts 2–3 |
| Booking rule; overrun limit $\alpha$; fill $f$ | Monte Carlo (Part 2 scripts). In the dashboard, scenario A defaults to average and B to Monte Carlo. 10%; 100% | Parts 2–3 |
| Cancellation tolerance | 30 minutes | Parts 2–3 |
| Planner draws per patient | 500 | Parts 2–3 |
| Hierarchical model | surgeon effect on; $\tau$ floored at 0.01; method of moments | All parts |
| Planner's duration model | plug-in (scripts and dashboard default); hierarchical and true durations as options | All parts |
| Warm-up, measured weeks | 26, 52 | Parts 2–3 |
| Runs (seeds) | 20 (Part 1), 10 (scripts), 5 (dashboard default) | All |

---

## 11. Next steps

1. **A fully Bayesian duration model.** The hierarchical model (section 3.1) treats $\hat\sigma_p$ and $\hat\tau$ as known, which makes its intervals a little too narrow when data are sparse. Fitting the same model by MCMC (in PyMC or Stan) would carry that uncertainty too. The estimates could also be updated as cases happen through the year rather than fixed at the start.
2. **Pooled lists.** Let any surgeon in the specialty take a patient, with that surgeon's durations.
3. **Pressure.** Add seasonal demand and winter list cancellations.
4. **An exact benchmark** for Part 1's heuristic: solve the same sampled problem as an integer programme.

---

## References

- Cardoen B, Demeulemeester E, Beliën J (2010). Operating room planning and scheduling: a literature review. *European Journal of Operational Research* 201(3): 921–932.
- Charnes A, Cooper WW (1959). Chance-constrained programming. *Management Science* 6(1): 73–79.
- Denton B, Viapiano J, Vogl A (2007). Optimization of surgery sequencing and scheduling decisions under uncertainty. *Health Care Management Science* 10(1): 13–24.
- Gelman A, Hill J (2007). *Data Analysis Using Regression and Multilevel/Hierarchical Models*. Cambridge University Press.
- Strum DP, May JH, Vargas LG (2000). Modeling the uncertainty of surgical procedure times: comparison of log-normal and normal models. *Anesthesiology* 92(4): 1160–1167.
- The importance of surgical cutting time as a key performance indicator alongside touchtime utilisation in operating theatre efficiency optimisation (2024). *Cureus*. Definition of capped touch-time utilisation and the GIRFT 85% target.
- NHS England. Referral to treatment consultant-led waiting times: rules suite. Clock rules, including that hospital cancellations do not stop the clock.
