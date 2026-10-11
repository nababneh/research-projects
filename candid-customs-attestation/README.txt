CANDID, revision round 2: code and results
Manuscript information-4579471, main file ../candid_round_2.0.tex

Everything in this folder produced the numbers, tables and figures of the revised manuscript.
Scripts read and write results/ inside this folder, write figures to ../figures/R2/ and
tables to ../tables/.

  corpus.py             loads the substrate and draws the synthetic attestation layer
  estimators.py         scoring policies (raw, equal weight, static prior, Chair-Varshney oracle,
                        Beta reputation, truth discovery, learned likelihood ratio, mixture-corrected)
  experiments.py        declaration features (M0), computed causally
  evaluation.py         the single evaluation protocol, metrics and two-stage bootstrap
  run_main.py           E2 and E3: 20 replications calibrated, 10 uncalibrated
  run_sensitivity.py    one-factor-at-a-time sensitivity analysis, 5 replications per cell
  run_robustness.py     E4: audit latency and non-stationary collusion, 10 replications
  reference_impl.py     E1: reference node, stage costs, closed and open loop, durable storage,
                        footprint and environment record
  show_record.py        prints the record reproduced as Listing 1
  bootstrap_parallel.py intervals for Tables 7 and 8, budget curve and uncalibrated summary
  analyze.py            sensitivity and robustness summaries
  make_tables.py        writes ../tables/*.tex and fills every number of the manuscript text from
                        candid_round_2.0_template.tex
  figures.py, diagrams.py   all figures
  style_check.py        prose constraint verifier
  requirements.txt      pinned package versions

DATA
  Customs Import Declaration Dataset (Jeong, Kim, Park and Choi, doi:10.48550/arXiv.2208.02484),
  not redistributed here:
    git clone https://github.com/Seondong/Customs-Declaration-Datasets
  then copy its data/ folder to kcs_data/ inside this folder, or set KCS_DATA to it.

TO REPRODUCE (from this folder)
  python3 run_main.py 20 10
  python3 run_sensitivity.py 5
  python3 run_robustness.py 10
  python3 reference_impl.py
  python3 show_record.py > results/show_record.txt
  python3 bootstrap_parallel.py 400
  python3 analyze.py sensitivity robustness
  python3 diagrams.py && python3 figures.py
  python3 make_tables.py candid_round_2.0_template.tex ../candid_round_2.0.tex
  Performance figures depend on hardware. Ours are recorded in results/impl_environment.json.
