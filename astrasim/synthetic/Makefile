# Run an end-to-end OCS experiment: generate ETs, simulate, sweep.
#
# Usage:
#   make                          # run all three steps with defaults
#   make et                       # only generate Chakra ETs
#   make sim                      # only run AstraSim (needs ETs)
#   make sweep                    # only run OCS C-sweep (needs trace)
#   MODEL=70b DP=8 make           # override parallelism / model
#   make help                     # show this help

DP       ?= 4
TP       ?= 8
PP       ?= 2
SP       ?= 1
MODEL    ?= 8b
NPUS     ?= $(shell expr $(DP) \* $(TP) \* $(PP) \* $(SP))
WORKLOAD ?= llama3_$(MODEL)_tp$(TP)_pp$(PP)_dp$(DP)

HERE       := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
PYTHON     ?= $(HERE)../../.venv/bin/python
# Must be the container-side path: run_astrasim_stage.sh passes this verbatim
# to AstraSim inside Docker, where the host tree is mounted at /app/llm-parallelism.
SYSTEM_CFG ?= /app/llm-parallelism/astrasim/synthetic/stage_configs/system_trace.json

.PHONY: all et sim sweep help

all: et sim sweep

et:
	DP=$(DP) TP=$(TP) PP=$(PP) SP=$(SP) MODEL=$(MODEL) bash $(HERE)generate_stage_et.sh

sim:
	SYSTEM_CFG=$(SYSTEM_CFG) bash $(HERE)run_astrasim_stage.sh $(WORKLOAD) $(NPUS)

sweep:
	$(PYTHON) $(HERE)run_ocs_sweep.py $(WORKLOAD) $(NPUS) --dp-group-size $(DP)

help:
	@printf "Targets:\n"
	@printf "  all     Run et + sim + sweep (default)\n"
	@printf "  et      Generate Chakra ETs via STAGE (CPU, ~seconds)\n"
	@printf "  sim     Run AstraSim with trace logging (~3s for 64 ranks)\n"
	@printf "  sweep   Run OCS circuit-capacity C-sweep\n"
	@printf "\nVariables (current):\n"
	@printf "  DP=$(DP)  TP=$(TP)  PP=$(PP)  SP=$(SP)  MODEL=$(MODEL)\n"
	@printf "  NPUS=$(NPUS)  WORKLOAD=$(WORKLOAD)\n"
	@printf "\nExamples:\n"
	@printf "  make\n"
	@printf "  MODEL=70b DP=8 make\n"
	@printf "  make sweep WORKLOAD=llama3_8b_tp8_pp2_dp4 NPUS=64\n"
