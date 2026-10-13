# Literature map

Compiled 2026-10-01. Each entry was confirmed to exist via a web search hit on a publisher, arXiv, or author page. Summaries come from abstracts and search snippets, **not full-text reading**, unless stated. Items marked **[unverified detail]** exist but a specific claim about them is not checked. "Applies" lines are the author's judgement, not claims from the papers.

---

## 1. Restless bandits, Whittle index, switching costs, deteriorating arms

**1.1 Whittle (1988).** P. Whittle, "Restless bandits: activity allocation in a changing world," *J. Applied Probability* 25(A):287-298. https://www.semanticscholar.org/paper/Restless-bandits:-activity-allocation-in-a-changing-Whittle/45196e90c3b265cbcd008af6e1aac97128e525dc
Projects evolve whether or not they are served. Whittle relaxes "exactly m served" to "m served on average", and the Lagrange multiplier defines an index that reduces to the Gittins index when passive projects are frozen. The resulting index policy is a heuristic, defined only for "indexable" projects.
*Applies:* the natural framework for "N explorers decaying, one actuator"; the oracle/honest policies here are myopic (argmax of urgency), not Whittle.

**1.2 Weber & Weiss (1990).** R. R. Weber, G. Weiss, "On an index policy for restless bandits," *J. Applied Probability* 27(3):637-648. https://www.cambridge.org/core/journals/journal-of-applied-probability/article/abs/on-an-index-policy-for-restless-bandits/33B68AAB7D03097C5C17DA889D12B78A
As m, n -> infinity with m/n fixed, the Whittle index policy is asymptotically optimal if the fluid-limit ODE has a globally stable equilibrium (Whittle's conjecture, proved under that condition). An addendum corrects the original proof.
*Applies:* justifies an index policy as the benchmark at large N, but only for models with no switching cost.

**1.3 Niño-Mora (2001).** J. Niño-Mora, "Restless bandits, partial conservation laws and indexability," *Advances in Applied Probability* 33(1):76-98. (Cambridge Core, Advances in Applied Probability; no direct URL retrieved, found via search listing only)
Gives sufficient conditions for indexability (PCL-indexability) via partial conservation laws and an adaptive-greedy algorithm that computes the index. **[unverified detail: exact algorithm description]**
*Applies:* the tool for proving indexability of the health-decay arm if a restless reformulation is attempted.

**1.4 Niño-Mora (2026 review).** J. Niño-Mora, "Markovian restless bandits and index policies: a review." https://arxiv.org/pdf/2601.13045
Survey of the field. **[unverified detail: contents not read]**
*Applies:* entry point to find any newer deteriorating-arm results.

**1.5 Glazebrook, Ruiz-Hernandez & Kirkbride (2006).** K. D. Glazebrook, D. Ruiz-Hernandez, C. Kirkbride, "Some indexable families of restless bandit problems," *Adv. Appl. Probab.* 38(3):643-672. https://www.cambridge.org/core/journals/advances-in-applied-probability/article/some-indexable-families-of-restless-bandit-problems/74CE913F574094BB67AA2A8F1E913CAA
Proves indexability for several families of restless bandits motivated by machine maintenance and gives indices in explicit form for some of them. **[unverified detail: which families have closed forms]**
*Applies:* closest published indexability results for "machine degrades, maintenance resets it"; none of them include travel.

**1.6 Villar (2016).** S. S. Villar, "Indexability and optimal index policies for a class of reinitialising restless bandits," *Probab. Engrg. Inform. Sci.* 30(1):1-23, doi:10.1017/S026996481500025X. https://www.cambridge.org/core/journals/probability-in-the-engineering-and-informational-sciences/article/abs/indexability-and-optimal-index-policies-for-a-class-of-reinitialising-restless-bandits/895527D87CB65AFFD3A65F266BF4A9A9
Motivated by surveillance. Shows indexability for a class of bandits whose state re-initialises, derives the Whittle index in closed form, and shows it is optimal under the expected total criterion for stochastically heterogeneous arms. Compared with a myopic index and a 1-limited round robin. **[unverified detail: exact direction of the reset (on activation vs passivity) should be checked in the paper]**
*Applies:* the nearest published closed-form index for "reset on service" arms; it has zero travel/switch cost, which is exactly what this project adds.

**1.7 Restless bandits with switching costs / setup.** (a) A. Asawa and D. Teneketzis, "Multi-armed bandits with switching penalties," *IEEE Trans. Automatic Control* 41(3):328-348, 1996 **[citation from memory; existence confirmed only through secondary citations in 1.7b-d]**. (b) J. Niño-Mora, "Computing an index policy for bandits with switching penalties," VALUETOOLS 2007, doi:10.4108/smctools.2007.1994. https://eudl.eu/doi/10.4108/smctools.2007.1994 (c) J. Niño-Mora, "Fast two-stage computation of an index policy for multi-armed bandits with setup delays," *Mathematics* 9(1):52, 2021. https://doi.org/10.3390/math9010052 (d) J. Le Ny, M. Dahleh, E. Feron, "A linear programming relaxation and a heuristic for the restless bandit problem with general switching costs," arXiv:0805.1563. https://arxiv.org/abs/0805.1563 (e) Niño-Mora, "A faster index algorithm and a computational study for bandits with switching costs," arXiv:2304.01871. https://arxiv.org/abs/2304.01871
Search snippets say: the Asawa-Teneketzis index of a *classical* (non-restless) bandit with switching costs is its Whittle index in a restless reformulation, and Niño-Mora computes it for switching costs and delays. Le Ny et al. give an LP relaxation and one-step-lookahead heuristic for the restless case with *general* switching costs, supported only by numerical experiments and an ADP bound, with no optimality theory.
*Applies:* these are the nearest "index + switching cost" results; the arms there are non-restless (frozen when not served), so they do not cover decaying health.

**1.8 Mobile repairman / network maintenance.** D. Tian and R. Shone, "Dynamic repair and maintenance of heterogeneous machines dispersed on a network: a rollout method for online reinforcement learning," arXiv:2602.19277. https://arxiv.org/abs/2602.19277
One repairer on a network with travel time and random repair durations; MDP, an index heuristic claimed optimal "in certain special cases", improved by rollout. No closed-form Whittle index claimed in the abstract.
*Applies:* the closest published model to the actuator-plus-travel setting; a good comparator and likely reviewer pushback.

**Answer to the specific questions.**
- Whittle indexability for deteriorating-arm, reset-on-service models: **yes, without travel** (1.5, 1.6; plus the 2024 capacitated condition-based-maintenance paper, Flex. Serv. Manuf. J. 37:179-207, https://link.springer.com/article/10.1007/s10696-024-09544-y, which reports closed-form indices under threshold structure **[unverified detail]**).
- Closed form for *linear* deterioration: not found as a named result. Linear decay with reset is a degenerate deterministic case, and a closed form is plausible by hand but I did not find it published.
- Index policy for deteriorating arms **plus** travel/switching: **not found.** Switching-cost indices exist only for non-restless arms (1.7); mobile-repairman models with travel use heuristics or RL (1.8).

---

## 2. Dynamic traveling repairman, DVR, polling systems

**2.1 Bertsimas & van Ryzin (1991).** "A stochastic and dynamic vehicle routing problem in the Euclidean plane," *Operations Research* 39(4):601-615. (Already cited in README.)
Poisson demands in the plane with a single vehicle; derives light- and heavy-traffic lower bounds and optimal policies. Shows FCFS-type disciplines are unstable at rates where better policies are stable. **[unverified detail: specific policy names from memory of the README claim]**
*Applies:* the source of the "nearest-neighbour / partition+tour" ranking this repo reproduces.

**2.2 Bertsimas & van Ryzin (1993).** "Stochastic and dynamic vehicle routing with general interarrival and service time distributions," *Adv. Appl. Probab.* 25(4):947-978. https://www.mit.edu/~dbertsim/papers/Vehicle%20Routing/Stochastic%20and%20dynamic%20vehicle%20routing%20with%20general%20arrival%20and%20demand%20distributions.pdf
Extends the above to general arrival and service distributions, improves lower bounds, constructs policies within a constant factor of optimal, and shows optimal system time has a simple form.
*Applies:* the heavy-load benchmark for a tour-style (`none`) policy; they assume no private information.

**2.3 Bullo, Frazzoli, Pavone, Savla, Smith (2011).** "Dynamic vehicle routing for robotic systems," *Proc. IEEE* 99(9):1482-1504, doi:10.1109/JPROC.2011.2158181. http://dspace.mit.edu/handle/1721.1/81456
Survey of dynamic vehicle routing for robots: demands arrive at random locations and times, a vehicle travels to serve them to minimise expected wait; covers single- and multi-vehicle variants and distributed policies.
*Applies:* the robotics-community framing and policy menu (partitioning, TSP-based, nearest) for the travel side.

**2.4 Polling systems.** (a) O. J. Boxma, W. P. Groenendijk, "Pseudo-conservation laws in cyclic-service systems," *J. Applied Probability* 24:949-964, 1987 (the search snippet said 1986; check year). https://www.semanticscholar.org/paper/Pseudo-conservation-laws-in-cyclic-service-systems-Boxma-Groenendijk/ac4778aa39f5dacccdc54cbb46fff9f28515ff60 (b) H. Takagi, "Queuing analysis of polling models," *ACM Computing Surveys* 20(1):5-28, 1988. https://dl.acm.org/doi/10.1145/62058.62059 (c) H. Takagi, *Analysis of Polling Systems*, MIT Press, 1986.
With switchover times the server is not work-conserving, so the classical conservation law fails; Boxma and Groenendijk derive a *pseudo*-conservation law for a weighted sum of mean waits across queues in cyclic service (exhaustive, gated, limited). Takagi surveys the exhaustive/gated/limited disciplines.
*Applies:* the cyclic `none` policy is a polling system with switchover = travel; the pseudo-conservation law says its weighted mean wait is policy-insensitive within a class, which is the structural reason urgency-ordering buys little. **[unverified detail: applicability to the deterministic-decay model here is the author's inference]**

---

## 3. Allocation without money / artificial currencies

**3.1 Gorokh, Banerjee, Iyer (2017/2021).** "From monetary to non-monetary mechanism design via artificial currencies," EC'17 (https://dl.acm.org/doi/10.1145/3033274.3085140); *Math. Oper. Res.* 46(3):835-855, 2021 (https://pubsonline.informs.org/doi/10.1287/moor.2020.1098).
Two black-box constructions turn any one-shot monetary mechanism into a dynamic artificial-currency mechanism with vanishing gain from misreporting and vanishing efficiency loss over time; for two agents the price of anarchy vanishes.
*Applies:* the principled version of `priced`: replace the ad hoc lambda·d with a currency budget, but their agents have valuations, not a decaying state.

**3.2 Guo, Conitzer, Reeves (2009).** "Competitive repeated allocation without payments," WINE 2009, LNCS 5929. https://link.springer.com/chapter/10.1007/978-3-642-10841-9_23
Repeated allocation of one item with artificial payments built from a one-shot mechanism; 0.94-competitive (high/low values), 0.85 (general prior), at least 0.75 for any number of agents versus the optimum with payments.
*Applies:* efficiency-loss yardstick for non-monetary rationing of a single scarce resource.

**3.3 Karma.** E. Elokda, S. Bolognani, A. R. Censi, F. Dörfler, E. Frazzoli, "A self-contained karma economy for the dynamic allocation of common resources," *Dynamic Games and Applications* 14(3), 2024; arXiv:2207.00495. https://arxiv.org/abs/2207.00495 (the arXiv author order printed on the abstract page differs from the usual "Censi" listing; check authorship before citing). Related: "Dynamic resource allocation with karma: an experimental study," arXiv:2404.02687; "Fair money: public good value pricing with karma economies," arXiv:2407.05132.
Agents bid circulating karma tokens for a repeatedly allocated resource; with future-aware agents the equilibrium is efficient and ex-post fair for homogeneous agents.
*Applies:* the natural mechanism for "urgency as a bid"; the bid cost is exogenous to travel, so it would need a distance-dependent price.

**3.4 Balseiro, Gurkan, Sun (2019).** "Multiagent mechanism design without money," *Operations Research* 67(5):1417-1436.
Dynamic mechanism design for repeated allocation without transfers with private valuations. **[unverified detail: summary from title and citation context only; abstract not read]**
*Applies:* second theoretical baseline for repeated, no-money allocation.

**3.5 Prendergast (2022).** "The allocation of food to food banks," *J. Political Economy* 130(8). https://www.journals.uchicago.edu/doi/abs/10.1086/720332
Feeding America moved from a queue to an auction in a constructed currency ("shares"); reallocation raised value of food by about 21% (about $115M per year).
*Applies:* field evidence that an artificial currency beats queue/first-come allocation of a scarce good.

---

## 4. Audits and ex-post verification

**4.1 Townsend (1979).** "Optimal contracts and competitive markets with costly state verification," *J. Economic Theory* 21(2):265-293. https://ideas.repec.org/a/eee/jetheo/v21y1979i2p265-293.html
Agents privately know the state; the principal can learn it only at a cost. Optimal contracts verify only in some states and are simple (debt-like).
*Applies:* origin of "verify rarely, commit to it" logic for an auditor of explorer self-reports.

**4.2 Ben-Porath, Dekel, Lipman (2014).** "Optimal allocation with costly verification," *AER* 104(12):3779-3813. https://www.aeaweb.org/articles?id=10.1257%2Faer.104.12.3779
One indivisible good, no transfers, privately known value to the principal, verification at a cost. Optimal mechanisms are randomisations over "favored-agent" mechanisms: if all others report below a threshold the favoured agent gets the good unchecked, otherwise the highest report is checked and gets the good only if confirmed.
*Applies:* the closest template for "who gets the actuator" when claims can be checked; single-shot, no travel, no dynamics.

**4.3 Mylovanov & Zapechelnyuk (2017).** "Optimal allocation with ex post verification and limited penalties," *AER* 107(9):2666-2694. https://www.aeaweb.org/articles?id=10.1257%2Faer.20140494
Agents with private values compete for a prize; the winner's claim is verified ex post and a false claim brings a limited penalty. With many agents the optimum shortlists everyone above a threshold plus a fraction below it and randomises; with few agents it gives the prize to the highest claim but restricts the claim range.
*Applies:* the most relevant mechanism for **ex-post-verifiable** urgency (health becomes observable once the actuator arrives); "limited penalty" maps to a bounded cost such as reduced future priority or budget.

**4.4 Dai, Blanchard, Jaillet (COLT 2025).** "Non-monetary mechanism design without priors: achieving efficiency via adaptive costly audits," arXiv:2502.08412. https://arxiv.org/abs/2502.08412
Repeated allocation to strategic agents with no transfers and no prior; true utilities are revealed by audits after allocation (allocations are not revocable). Achieves T-independent O(K^2) welfare regret with O(K^3 log T) audits; Omega(K) regret and Omega(1) audit lower bounds; extension to imperfect audits.
*Applies:* the most direct modern analogue of repeated, no-money, audit-disciplined allocation; no state dynamics or travel.

**4.5 Survey.** "Evidence in games and mechanisms," *Annual Review of Economics* (2024/25). https://www.annualreviews.org/content/journals/10.1146/annurev-economics-051624-060215 **[unverified detail: authors and contents not read]**
*Applies:* index to the verifiable-evidence mechanism design literature.

**Reputation in repeated reporting:** not searched in depth. I do not cite a specific reputation paper; treat this as an open reading task.

---

## 5. Value of information; strategic requesting in queues

**5.1 Naor (1969).** P. Naor, "The regulation of queue size by levying tolls," *Econometrica* 37(1):15-24. https://www.econometricsociety.org/publications/econometrica/1969/01/01/regulation-queue-size-levying-tolls
M/M/1 queue, customers decide whether to join, balancing reward against their own waiting cost. Individually optimal joining exceeds the socially optimal threshold because each joiner ignores the delay imposed on others; a toll equal to that externality restores social optimality.
*Applies:* see the connection below.

**5.2 Hassin & Haviv (2003).** *To Queue or Not to Queue: Equilibrium Behavior in Queueing Systems*, Kluwer/Springer. https://link.springer.com/book/10.1007/978-1-4615-0359-0
Monograph on equilibrium behaviour in queues: observable/unobservable queues, priorities, schedules, retrials, tolls, service-rate decisions.
*Applies:* supplies price-of-anarchy and toll results for many variants; none with a mobile server or decaying health **[unverified: no chapter checked]**.

**Connection to the lambda·d request price (author's analysis, not from a cited paper).**
- In Naor, the optimal toll equals the *externality*: the expected added waiting a marginal joiner imposes on everyone behind it, valued at their waiting cost. Joiners decide privately on net benefit; the toll shifts them to the social optimum.
- Here a request imposes an externality through travel: serving a distant explorer burns actuator time that delays everyone else's restoration. An entry price increasing in distance, theta + lambda·d, is the same device: charge for the externality the request creates.
- So `priced` is a Naor-type toll. Naor-style tolls are *derived* from the externality in an exactly solved model. Here the externality is state-dependent (all health levels, actuator position, who else is waiting) and the actuator re-targets on the fly, so lambda* would have to come from the marginal value of actuator time, e.g. a relative value function or Whittle-style Lagrange multiplier. I found no published closed form for lambda* in this setting. Naor's closed form is for M/M/1 with identical customers, and heterogeneous-customer toll results (Hassin-Haviv; arXiv:1605.07107, "Revenue maximization in service systems with heterogeneous customers") do not cover travel costs. Expect lambda* to be computed or searched numerically, as done here, and to rise with load because the shadow price of actuator time rises with load.

---

## 6. Multi-robot task allocation

**6.1 Gerkey & Matarić (2004).** "A formal analysis and taxonomy of task allocation in multi-robot systems," *Int. J. Robotics Research* 23(9):939-954, doi:10.1177/0278364904045564. https://dblp.org/rec/journals/ijrr/GerkeyM04.html
Domain-independent taxonomy of MRTA (single- vs multi-task robots, single- vs multi-robot tasks, instantaneous vs time-extended assignment) and mappings to known optimisation problems (e.g. optimal assignment).
*Applies:* this project is single-robot-task, time-extended assignment with one super-robot: mostly outside their classification, since "tasks" arrive from agents that misreport.

**6.2 Dias, Zlot, Kalra, Stentz (2006).** "Market-based multirobot coordination: a survey and analysis," *Proc. IEEE* 94(7):1257-1270, doi:10.1109/JPROC.2006.876939. https://www.semanticscholar.org/paper/Market-Based-Multirobot-Coordination:-A-Survey-and-Dias-Zlot/08c0c3b42c337809e233371c730c23d08da442bf
Survey of auction/market approaches in which robots bid for tasks using cost or utility estimates.
*Applies:* these mechanisms assume honest cost bids from cooperative robots, which is the assumption `strategic` breaks.

---

## Positioning

- **Already known (travel side).** Serve-nearest and partition-plus-tour beat FCFS under load (Bertsimas-van Ryzin 1991/93; Bullo et al. 2011), and with switchover times work-conservation fails and policy-insensitivity results exist (pseudo-conservation laws). "The oracle does not beat ignoring urgency" is consistent with that; it is not new on its own, as the README already says.
- **Already known (index side).** Whittle-type indices with closed forms exist for reset-on-service maintenance/surveillance arms (Glazebrook et al. 2006; Villar 2016), and switching-cost indices exist for non-restless arms (Asawa-Teneketzis; Niño-Mora). Argmax-of-urgency is therefore not the right "smart" benchmark; a Whittle-style index (or Le Ny et al.'s lookahead heuristic) should be in the comparison set. Without it, "urgency has no value" is shown only against a myopic rule.
- **Not found: an index for deteriorating arms plus travel.** I found no published indexability result or closed-form index that has decaying health, reset by service, and a position-dependent travel/setup cost. This may be a gap worth a theorem on a ring (state includes actuator position, so the per-arm decomposition breaks) or an honest statement that it is open. A negative search is not proof of absence.
- **Already known (strategic side, no verification).** Repeated allocation without money via artificial currencies has vanishing-loss and competitive-ratio guarantees (Gorokh et al.; Guo et al.), practical success (Prendergast food banks) and an equilibrium theory in karma. `strategic` ("always claim the ceiling") is the cheap-talk babbling outcome those papers are designed to avoid.
- **Already known (verification).** With ex-post checkable claims, optimal no-transfer allocation is characterised in the static case (Ben-Porath-Dekel-Lipman; Mylovanov-Zapechelnyuk) and learning-based repeated audits exist (Dai-Blanchard-Jaillet 2025). Auditing is a known remedy for free, unverifiable claims; it should not be presented as new.
- **Pricing by externality is Naor's idea.** `priced` is a Naor toll in disguise. Calling it novel would be wrong; the defensible claim is empirical: lambda* rises with load and recovers a measurable share of loss in a dynamic, travel-coupled system.
- **Plausibly novel combination.** (i) private, noisy, *free-to-claim* urgency; (ii) a **physical server whose service cost is travel and whose queue position is geometry** (so the externality of a claim depends on distance, unlike a generic queue); (iii) urgency that is *ex-post verifiable on arrival* but *not* ex ante; (iv) a **decaying state** rather than i.i.d. valuations. I did not find any paper combining all four. Individual pairs are covered: (i)+(iv) by restless-bandit information models, (ii) by DVR, (i)+(iii) by the audit papers, (ii)+(i) partly by Naor-type queues.
- **Specifically open and testable.** (a) A distance-dependent Mylovanov-Zapechelnyuk-style mechanism: shortlist claimants within a distance band, randomise or pick nearest, and verify the winner's health on arrival with a bounded penalty (reduced priority). (b) An equilibrium (not hand-specified `strategic`) for requesters under `priced`, ideally a fixed point in lambda. (c) A Whittle-style index with travel as the benchmark replacing argmax-of-urgency. (d) A derivation or numerical bracketing of lambda* versus load, checked against the Naor externality interpretation above.
- **Caveats on the project's own claims.** The README's rank correlations are over six policies, so they are descriptive, and `strategic` is not an equilibrium. The literature above suggests those two gaps (no equilibrium, no index benchmark) are where a reviewer familiar with restless bandits and mechanism design would push first.
