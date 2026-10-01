# OpenReview submission fields

Venue: https://openreview.net/group?id=NeurIPS.cc/2026/Workshop/PAIEC (Submission, due 31 Oct 2026 11:59 UTC)

**Title**
A Shrunk Logistic Predictor for Cold Start Benchmarks

**Authors**
Aghasalim Mustafazada (your OpenReview profile)

**Keywords**
item response theory, Bayesian shrinkage, cold start, predictive evaluation, Brier score

**TLDR**
A small logistic model with MAP online offsets beats the empirical mean baseline on held out benchmarks; most of the gain is shrinkage, and two text based ideas did not transfer.

**Abstract**
We predict whether an AI system answers an item correctly on a benchmark never seen in training, while labels are revealed a few at a time. The predictor is a logistic model with an ability prior for each system learned by model name, an online difficulty for the new benchmark, an offset for each system on that benchmark, and online offsets for items that share a tag or are the same question, all estimated as MAP under Gaussian priors. On a local copy of the scoring protocol that holds out one of the five binary measurement-db benchmarks at a time, it scores 0.2146 Brier ALC against 0.2625 for the empirical mean baseline and 0.2500 for a constant 0.5, and the best platform submission scored 0.179449. The budget breakdown shows that ability learned on other benchmarks barely helps before the first label; nearly all of the gain is shrinkage once labels arrive. Cheap text features and sentence embedding neighbours both failed to transfer across benchmarks, and we report both. Code: https://github.com/aghasalim/predictive-eval

**PDF**
report/REPORT.pdf

**Data release**
No training data beyond the public measurement-db release. Code: https://github.com/aghasalim/predictive-eval
