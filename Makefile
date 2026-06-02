.PHONY: setup gen-data eval demo test lint serve lock clean
PY ?= python3

setup:        ## install runtime + dev deps and the package (editable)
	$(PY) -m pip install -e ".[dev]"

gen-data:     ## (re)generate the synthetic interaction history + gold queries
	$(PY) data/generate_sessions.py

eval:         ## memory-policy ablation (flat / +recency / +consolidation / +supersession)
	$(PY) eval/run_eval.py

demo:         ## recall a fact and show the audit trail
	$(PY) scripts/demo.py "Who is my current manager?"

test:         ## run the test suite
	$(PY) -m pytest -q

lint:         ## static checks
	$(PY) -m ruff check src eval serve bench data tests scripts

serve:        ## run the HTTP service locally on :8080 (Ctrl-C to stop)
	$(PY) -m uvicorn serve.app:app --host 0.0.0.0 --port 8080

lock:         ## regenerate the pinned, hash-locked requirements.txt (linux/py3.12 via docker)
	docker run --rm -v "$(CURDIR):/w" -w /w python:3.12-slim sh -c \
		"pip install -q pip-tools && pip-compile -q --generate-hashes --output-file requirements.txt requirements.in"

clean:
	rm -rf .cache .pytest_cache .ruff_cache **/__pycache__

locomo-data:  ## download the public LoCoMo-10 dataset (not redistributed in this repo)
	@mkdir -p data/locomo
	@test -f data/locomo/locomo10.json || ( \
		git clone --depth 1 https://github.com/snap-research/locomo /tmp/locomo_dl && \
		cp /tmp/locomo_dl/data/locomo10.json data/locomo/ && rm -rf /tmp/locomo_dl )
	@echo "LoCoMo-10 ready at data/locomo/locomo10.json"

locomo:       ## LoCoMo-10 retrieval recall of gold evidence (deterministic, no API key)
	$(PY) -m eval.locomo.run_locomo --mode retrieval --data data/locomo/locomo10.json

locomo-qa:    ## LoCoMo-10 end-to-end answer F1 (needs ANTHROPIC_API_KEY; add EXTRACT=1)
	$(PY) -m eval.locomo.run_locomo --mode qa --data data/locomo/locomo10.json $(if $(EXTRACT),--extract,)
