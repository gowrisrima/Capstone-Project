# Overview

**Instamart Customer Intelligence & Support Copilot** — a four-part, fully offline, reproducible capstone using synthetic data only. Part A cleans and audits orders and builds SQLite analytics; Part B segments customers and predicts lapse; Part C redacts and routes support tickets using the supplied offline mock LLM; Part D answers policy questions with TF-IDF retrieval and executes a guarded, resumable refund agent. All prices and refunds are in INR. The Instamart scenario, policies, and customers are fictional.

# Reproduce

Run these **numbered commands from the repository root**, using Python 3.11+ and SQLite 3.25+. The tested environment used pandas 2.3.3, NumPy 2.3.4, scikit-learn 1.7.2 and joblib 1.5.2. Windows PowerShell:

1. `python -m venv .venv`
2. `.\.venv\Scripts\Activate.ps1`
3. `python -m pip install -r requirements.txt`
4. `python generate_data.py`
5. `python part_a/audit_data.py`
6. `python part_a/clean_data.py`
7. `python part_a/validate_data.py`
8. `python part_a/sql_analysis.py`
9. `python part_a/pandas_analysis.py`
10. `python part_b/cslp.py`
11. `python part_c/support_guardrails.py`
12. `python part_d/policy_refund_agent.py`

On macOS/Linux activate with `source .venv/bin/activate` instead of step 2. The supplied `generate_data.py` and `mock_llm.py` are used unmodified. Do not regenerate or edit those files manually. `requirements.txt` records the environment's package versions. The scripts should run in the above order; Parts B–D depend on the clean data or database created by Part A. The project writes data, reports, models, agent logs, and checkpoints to `data/`, `outputs/`, and `state/`. The final Part A validation message is `ALL CHECKPOINTS PASSED`. The commands were rerun in an existing Windows virtual environment; fresh-clone reproduction should also be checked before submission.

# Part A

## Raw audit

Each of the three raw CSV files is audited for shape, inferred data types, and missing count and percentage for every column. The full table is saved to `outputs/audit_report.txt`.

| Raw audit figure | Result |
|---|---:|
| Orders rows | 13,084 |
| Exact duplicate order rows | 119 |
| Pending order rows | 656 |
| Distinct order IDs | 12,309 |
| Non-numeric amounts | 120 |
| Negative delivery minutes | 106 |
| Blank delivery minutes | 2,541 |
| Blank customer ages | 61 |
| Ages outside 10–100 | 6 |
| Distinct raw city spellings | 24 |
| Distinct nonblank gender spellings | 10 |
| Blank genders | 30 |

## Cleaning rules and counts

- Customers: strip city whitespace and title-case to six standard cities. Map all male/M and female/F case variants to `male`/`female`; map 30 blank genders to `unknown`. Convert ages to numeric, mark six invalid ages missing, and impute all 67 invalid or blank ages using the median age of each city (`groupby(...).transform('median')`). Parse signup dates.

- Orders: parse timestamps; remove 119 exact duplicates; then retain the latest timestamp per order ID, eliminating 656 earlier partial duplicates. Convert amounts to numeric; reconstruct the 112 remaining missing final-order amounts using product price × quantity × (1 − discount/100), rounded to two decimals. Mark 105 remaining negative delivery durations missing (the raw audit has 106). Impute 774 missing completed-order durations with the category-specific median of completed orders. Leave all 1,199 cancelled-order durations missing. Export the three clean CSV files and three SQLite tables.

Clean customer cities: Pune 227; Delhi 210; Bengaluru 197; Chennai 191; Hyderabad 191; Mumbai 184. Clean genders: female 610; male 560; unknown 30. All ages are in 18–60 with none missing.

Clean orders: **12,309** unique orders, **11,110** completed, **1,199** cancelled, **0** pending, **0** missing amounts, **0** missing completed delivery durations, and **1,199** missing cancelled delivery durations. Completed revenue: **INR 2,998,668.80**. `validate_data.py` checks more than eight reference values and prints `ALL CHECKPOINTS PASSED`.

## SQL results

All seven queries live in `sql/queries.sql`. `sql_analysis.py` executes them using `pd.read_sql_query`, saves each result as `outputs/sql_q1.csv` through `outputs/sql_q7.csv`, and prints row counts.

| Query | Result |
|---|---|
| Q1 WHERE + HAVING | 30 qualifying customers |
| Q2 three-table JOIN | 30 city/category cells; INR 2,998,668.80 total; Bengaluru / Electronics highest at INR 233,909.45 |
| Q3 scalar subquery | 11 products above mean catalogue price |
| Q4 LEFT JOIN | 25 customers without completed orders |
| Q5 two CTEs | 550 above-average customers; average spend INR 2,552.06 |
| Q6 RANK / DENSE_RANK / ROW_NUMBER | Top spenders: Bengaluru C0082 INR 12,856.20; Chennai C0930 INR 10,151.40; Delhi C0291 INR 13,742.60; Hyderabad C0743 INR 12,730.05; Mumbai C0693 INR 14,766.55; Pune C0426 INR 10,377.10. Eight rows differ between RANK and ROW_NUMBER. |
| Q7 running total | 183 daily rows; last cumulative revenue INR 2,998,668.80 |

Ties cause `RANK()` to assign equal ranks (and skip subsequent rank numbers), while `ROW_NUMBER()` assigns a unique sequence number to every row.

## Pandas cross-check and EDA

`pandas_analysis.py` saves `outputs/pandas_report.txt`, including the 6×5 pivot table with margins, city groupby statistics, distribution metrics, and the correlation matrix. The pivot grand total equals the SQL total of INR 2,998,668.80.

| City | Completed spend (INR) | Mean (INR) | Completed orders |
|---|---:|---:|---:|
| Bengaluru | 523,643.40 | 276.47 | 1,894 |
| Chennai | 463,629.15 | 272.72 | 1,700 |
| Delhi | 502,829.80 | 243.62 | 2,064 |
| Hyderabad | 528,589.90 | 284.95 | 1,855 |
| Mumbai | 479,058.50 | 297.00 | 1,613 |
| Pune | 500,918.05 | 252.48 | 1,984 |

Completed order amounts: mean INR 269.91, median INR 135.00, skewness 4.6088, Q1 INR 64.80, Q3 INR 297.00, IQR INR 232.20, lower fence INR −283.50, upper fence INR 645.30, zero outliers below and 897 above. The strongest absolute Pearson correlation is amount versus discount percentage, approximately −0.095 (amount versus quantity is +0.087).

**EDA interpretation (four questions, under 200 words):**

1. The dataset contains synthetic grocery-delivery customer, product, and order records across six Indian cities, with order revenue and delivery outcomes.

2. Completed order amounts are strongly right-skewed: the mean of INR 269.91 exceeds the median of INR 135.00, and skewness is 4.61. Q1 and Q3 are INR 64.80 and INR 297.00.

3. The IQR upper fence is INR 645.30, with 897 high outliers and no low outliers. The raw exports also contained invalid ages, duplicate orders, unparseable amounts, and impossible negative delivery times, which were cleaned.

4. The strongest absolute correlation among amount, discount, quantity, and delivery time is amount with discount (about −0.095), slightly stronger than amount with quantity (+0.087). This is a weak association and does not establish causation.

# Part B

## RFM and segmentation

RFM uses **completed orders only** with snapshot **2025-10-01 00:00:00**: recency is whole days since last completed order, frequency counts distinct completed orders, and monetary is total completed spend. **1,175** customers have completed orders. Recency median/min/max: **19/0/180 days**; frequency mean/max **9.46/33**; monetary mean/max **INR 2,552.06 / INR 14,766.55**.

K-Means standardizes the three RFM features with `StandardScaler`, uses `random_state=42` and `n_init=10`, and evaluates K=2 through 8:

| K | Inertia | Silhouette |
|---:|---:|---:|
| 2 | 1820.761 | 0.465 |
| 3 | 1268.235 | 0.445 |
| 4 | 743.071 | **0.526** |
| 5 | 617.367 | 0.480 |
| 6 | 503.171 | 0.482 |
| 7 | 408.460 | 0.416 |
| 8 | 366.503 | 0.412 |

**K=4** has the highest tested silhouette. Inertia decreases as K increases because more clusters can fit the training points more closely; therefore K=8 having lower inertia does not by itself make it the best segmentation. Silhouette balances within-cluster cohesion against separation from other clusters, and K=4 has the best silhouette among K=2–8 (0.526). Names reflect behavioral profiles, not arbitrary numeric IDs:

| Segment ID | Name | Customers | Mean recency | Mean frequency | Mean monetary (INR) | Suggested action |
|---:|---|---:|---:|---:|---:|---|
| 0 | Occasional | 436 | 28.86 | 3.77 | 1,057.62 | Low-cost reorder reminders |
| 1 | Premium | 138 | 24.12 | 6.77 | 7,129.70 | Priority service and personalized retention |
| 2 | Frequent | 419 | 8.77 | 19.40 | 3,410.02 | Loyalty rewards and cross-sell |
| 3 | At Risk | 182 | 109.04 | 2.20 | 685.99 | Win-back campaign |

Premium is defined as the cluster with the **highest mean monetary value**. PCA on standardized RFM gives explained variance **0.6343** (PC1) and **0.2180** (PC2), totaling **0.8523** (85.23%). `outputs/customer_segments.csv` contains customer ID, RFM, segment ID/name, and Premium flag.

## Lapse prediction

The lapse cohort contains customers with at least one completed order **before 2025-09-01**. A customer is labeled lapsed if they have **no completed orders during September 2025**. All predictive features are computed from pre-cutoff data only; September data supplies labels, not input features. The model uses numerical and categorical customer/order features with preprocessing inside scikit-learn pipelines to fit transformations on training data only.

| Measure | Result |
|---|---:|
| Eligible cohort | 1,156 |
| Lapsed | 443 (38.32%) |
| Training / test | 924 / 232 |
| Majority-class baseline accuracy | 0.6164 |
| Majority-class lapsed F1 | 0.0000 |
| Training-only frequency mean | 8.1147 |
| Full-cohort frequency mean (comparison only) | 8.0285 |

Logistic Regression is tuned by five-fold CV across C values **0.01, 0.1, 1, 10, 100**; Random Forest uses 100 trees and searches depths **3, 5, 8** and minimum leaf sizes **5, 10, 20** with five-fold CV, both scored by lapsed-class F1.

| Model | Best parameters | CV F1 | Test precision | Test recall | Test F1 | Confusion matrix `[[TN,FP],[FN,TP]]` |
|---|---|---:|---:|---:|---:|---|
| Logistic Regression | C=100 | 0.6506 | 0.6444 | 0.6517 | **0.6480** | `[[111,32],[31,58]]` |
| Random Forest | max_depth=8, min_samples_leaf=10 | 0.6865 | 0.6404 | 0.6404 | **0.6404** | `[[111,32],[32,57]]` |
| Training-only oversampled Logistic Regression | Balanced training 570/570 | — | 0.5891 | 0.8539 | **0.6972** | `[[90,53],[13,76]]` |

Training-only oversampling increases recall but also increases false positives (53 versus 32). Missing a customer who is likely to lapse may mean losing future revenue, whereas incorrectly targeting a loyal customer costs outreach resources or discounts. For a retention campaign that prioritizes catching at-risk customers, the balanced logistic model offers higher recall (0.8539 versus 0.6517) and higher test F1 (0.6972 versus 0.6480), but at the cost of lower precision (0.5891 versus 0.6444). This trade-off should be evaluated against campaign costs before deployment. The unbalanced Logistic Regression is the persisted selected model based on the script's unbalanced-model selection comparison (test F1 0.6480 versus 0.6404 for Random Forest), not a comparison that includes the balanced model; `outputs/lapse_model.joblib` was reloaded and produced identical predictions. Random Forest's leading importance values: monetary **0.2778**, frequency **0.2768**, recency **0.1112**, average delivery **0.0616**, average discount **0.0532**. Logistic coefficient signs include recency **+0.0396**, frequency **−1.6982**, and monetary **+0.2846**; coefficients depend on preprocessing and feature scale and are not causal effects.

`outputs/retention_list.csv` contains **10** Premium test-set customers ranked for outreach; **3** actually lapsed. **Leakage caveat:** Premium segments are derived from the whole observation window including September, so this post-hoc list is illustrative and **not** a leakage-free prospective evaluation. The lapse predictor itself uses pre-cutoff features. No scaler, encoder, or oversampling procedure is fitted on test rows: preprocessing is fitted inside the training pipeline and oversampling is performed on training data only. Segment IDs, segment names, and the post-hoc Premium flag are not used as lapse-model predictive features.

# Part C

## Ticket classification, privacy and guardrails

The supplied `data/raw/tickets.csv` contains **164** tickets. Before LLM calls, the pipeline masks emails, Indian phone numbers, and order IDs, and handles empty or overlong messages. The offline `mock_llm.py` deliberately simulates transient failures, non-JSON responses, and invalid category values. The pipeline retries transient errors with short exponential backoff, validates JSON categories against `billing`, `delivery`, `technical`, `fraud`, uses a restricted keyword fallback when possible, and flags unresolved outputs for human review. The two prompts compare zero-shot with few-shot examples; the latter costs more tokens. In this mock benchmark, few-shot prompting is worthwhile when classification mistakes have a meaningful handling cost (especially fraud/security routing): it improves accuracy from 85.71% to 100% on the 147 classified tickets, while increasing token use from 8,932 to 13,768 (4,836 additional tokens, about 54% more). For low-risk, high-volume triage, zero-shot may be more cost-effective; these results are specific to the deterministic synthetic benchmark.

| Status | Tickets |
|---|---:|
| Valid JSON classification | 135 |
| Keyword fallback classification | 12 |
| Invalid output flagged | 9 |
| Overlong ticket flagged | 4 |
| Empty ticket skipped | 4 |

| Evaluation | Zero-shot | Few-shot |
|---|---:|---:|
| Classified | 147 | 147 |
| Correct | 126 | 147 |
| Accuracy on classified tickets | 85.71% | 100.00% |
| Total tokens | 8,932 | 13,768 |
| Tokens per classified ticket | 60.76 | 93.66 |

Zero-shot confusion matrix (true category rows; predicted columns in billing, delivery, fraud, technical order): billing **[38,0,0,0]**; delivery **[0,36,0,0]**; fraud **[11,0,23,0]**; technical **[10,0,0,29]**. Few-shot examples resolve the fraud/technical keyword ambiguities in this deterministic synthetic benchmark, but the 100% score is not evidence of production performance.

PII redaction substituted **55 emails**, **91 phone numbers**, and **132 order IDs** (**278 total**). The 6,000-token budget demonstration reported **6,023 tokens spent** and **51 budget-exhausted tickets**. **Limitation:** spending exceeded the nominal budget by 23 tokens; a strict pre-call reservation or post-call hard-cap policy would be needed to prevent overshoot. In the final few-shot routing run, **102** tickets went to `support_queue`, **34** to `security_team`, **17** to `human_review`, and **11** to `premium_desk`. JSON and fallback outputs were rule-validated; invalid/ambiguous and other unprocessable tickets were routed for human handling. The final outputs are `outputs/triage_results.csv` and `outputs/triage_budget_demo.csv`.

# Part D

## Policy retrieval and grounded answers

The five supplied policy text files are split into **40-word chunks with 10-word overlap**, retaining the source filename. This yields **15 chunks**: delivery 3, escalation 3, payment 3, privacy 2, refund 4. `TfidfVectorizer` fits on the chunk text; cosine similarity ranks the top three chunks. **TF** measures a word's frequency in a document/chunk, while **IDF** downweights words that occur across many chunks. TF-IDF depends on lexical overlap and may miss semantically equivalent wording.

For `refund policy late delivery`, all three top-ranked chunks come from `refund_policy.txt`. The policy assistant supplies the retrieved context to the unmodified mock LLM and instructs it not to invent unsupported policy facts.

| Policy evaluation | 10-word overlap | Zero overlap |
|---|---:|---:|
| Correct expected source in top 3 (12 answerable) | 12/12 | 12/12 |
| Answer contains expected fact (12 answerable) | 12/12 | 10/12 |
| Unanswerable questions with fallback | 2/2 | 2/2 |

A zero-overlap failure is **“What delivery time makes a completed order eligible for a full refund?”** (expected fact: **more than 45 minutes**). Without overlap, relevant wording can be split across chunk boundaries, so the best retrieved context may not preserve the complete condition.

## Guarded refund agent

The agent uses the offline model's tool-call interface and a `TOOLS_SPEC`/`TOOL_MAP` for **get_order**, **search_policy**, **calculate_refund**, **issue_refund**, and **escalate_to_human**. `get_order` reads the Part A SQLite database; `search_policy` reuses the TF-IDF retriever. Completed orders delivered in **more than 45 minutes** qualify for a full refund. Completed orders at **45 minutes or less** get no late-delivery refund; cancelled orders are ineligible. Refunds **above INR 1,000** are escalated, not automatically issued. The synthetic scenario requests are treated as within the seven-day policy window. Issued/escalated actions are appended to `outputs/refund_ledger.jsonl`.

| Scenario | Status | Refund (INR) | LLM calls | Tokens |
|---|---|---:|---:|---:|
| S1 | REFUND | 128.25 | 5 | 997 |
| S2 | REFUND | 135.00 | 5 | 997 |
| S3 | ESCALATE | 1,999.00 | 5 | 984 |
| S4 | DENIED | 0.00 | 4 | 685 |
| S5 | DENIED | 0.00 | 3 | 397 |
| S6 | NOT_FOUND | 0.00 | 2 | 112 |
| S7 | FALLBACK_MAX_ITERATIONS | 0.00 | 6 | 3,120 |

**Stop-safely demonstrations:** S7 stopped after six calls with `FALLBACK_MAX_ITERATIONS`. The S1 run with `token_limit=200` returned **FALLBACK_TOKEN_LIMIT**; a run with `time_limit_seconds=0` returned **FALLBACK_TIMEOUT** before calling the model. Fallbacks provide a human-handoff summary and retain completed steps.

**Crash-and-resume:** a simulated crash occurred immediately after the second tool's checkpoint was saved. `resume_agent` continued to **REFUND INR 128.25**. Tools executed, exactly once and in order: **get_order → search_policy → calculate_refund → issue_refund**. Checkpoints are stored in `state/`. Agent run logs are stored in `outputs/agent_runs.jsonl`.

## ReAct versus plan-and-execute

ReAct chooses the next action after each observation. It used **5 model calls for S1**, **3 for S5**, and **2 for S6**, consuming **997**, **397**, and **112** tokens respectively in these runs. A plan-and-execute approach could ask for an entire action sequence in one initial model call, potentially reducing calls and tokens on straightforward requests, but would need replanning when an order is missing, cancelled, or otherwise unexpected. ReAct can immediately adapt after `get_order` returns `NOT_FOUND` or a non-completed status; it avoids issuing a refund based on an outdated upfront plan. For this refund workflow, ReAct with tool validation, strict stop limits, audit logs, and human escalation is preferable because the monetary action depends on database observations. Its trade-off is extra calls and token use. A deterministic workflow could reduce costs further if business rules remain fixed.

# Limitations and Next Steps

All data and policies are synthetic; these metrics are not production performance estimates. Cleaning uses imputation and cannot reconstruct true missing values. Segmentation uses the whole observation window; its Premium retention list is **post-hoc**, not a leakage-free backtest. Model feature importance and coefficients do not imply causality. Few-shot ticket accuracy is measured on a small deterministic mock benchmark, not real customer language. The budget demonstration overshot by **23 tokens**; strengthen hard-budget enforcement before production. Policy retrieval is lexical and sensitive to chunk boundaries. The refund agent simulates ledger entries rather than making real payments, and requires robust authorization, idempotency, and operational monitoring for real deployment. Next steps: evaluate on held-out real-world-like data, implement strict token reservation, add authorization and deduplication to financial actions, and verify all outputs from a clean clone.
#   C a p s t o n e - P r o j e c t  
 