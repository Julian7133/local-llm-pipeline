MODEL ?= qwen-agentworld-35b
CFG := configs/$(MODEL).yaml
PY := python -m pipeline.cli

.PHONY: all fetch quantize fixups validate deploy test setup

all: ; $(PY) run $(CFG)
fetch: ; $(PY) fetch $(CFG)
quantize: ; $(PY) quantize $(CFG)
fixups: ; $(PY) fixups $(CFG)
validate: ; $(PY) validate $(CFG)
deploy: ; $(PY) deploy $(CFG)

test: ; pytest -q
setup: ; bash scripts/setup.sh
