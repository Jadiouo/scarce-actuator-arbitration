PY ?= python3

.PHONY: test quick all figures
test:
	PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $(PY) -m pytest -q

quick:
	$(PY) -m arbitration.experiments --quick

all:
	$(PY) -m arbitration.experiments
	$(PY) -m arbitration.figures

figures:
	$(PY) -m arbitration.figures

# Replot ALL figures from existing JSON only (CPU, no experiments).
.PHONY: replot
replot:
	$(PY) -m arbitration.figures
	$(PY) scripts/run_c_pricing.py figs
	$(PY) scripts/run_d_scaling.py figs
	$(PY) scripts/run_dp_step.py figs
