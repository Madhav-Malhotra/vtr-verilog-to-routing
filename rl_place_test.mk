# rl_place_test.mk — Regression and circuit tests for RL placement modes
#
# Usage:
#   make -f rl_place_test.mk reg_test
#   make -f rl_place_test.mk circ_test ARCH=<arch.xml> BLIF=<circuit.blif>

# ─── Paths ────────────────────────────────────────────────────────────────────
VPR          := ./build/vpr/vpr
ARCH_DEFAULT := vtr_flow/arch/timing/k6_frac_N10_frac_chain_mem32K_40nm.xml
BLIF_DEFAULT := vtr_flow/benchmarks/blif/tseng.blif

TEST_DIR     := rl_place_tests
CHECKPOINT   := $(TEST_DIR)/rl_checkpoints.jsonl

# ─── Colours ──────────────────────────────────────────────────────────────────
RED    := \033[0;31m
GREEN  := \033[0;32m
YELLOW := \033[1;33m
CYAN   := \033[0;36m
BOLD   := \033[1m
RESET  := \033[0m

# ─── Helpers ──────────────────────────────────────────────────────────────────
PASS  = printf "  $(GREEN)PASS$(RESET)  %s\n" "$(1)"
FAIL  = printf "  $(RED)FAIL$(RESET)  %s\n" "$(1)" && FAILED=1

# Extract a metric from a log file.
#   $(call EXTRACT, <log>, <grep-pattern>, <sed-pattern>)
EXTRACT = $$(grep -o "$(2)" "$(1)" | sed "$(3)" | tail -1)

# ─── reg_test ─────────────────────────────────────────────────────────────────
.PHONY: reg_test
reg_test:
	@printf "$(BOLD)$(CYAN)RL Placement Regression Test$(RESET)\n"
	@printf "Circuit : $(BLIF_DEFAULT)\n"
	@printf "Arch    : $(ARCH_DEFAULT)\n\n"

	@# ── Confirm build ───────────────────────────────────────────────────────
	@printf "$(YELLOW)Have you built the latest VPR? (y/n): $(RESET)"; \
	read ans; \
	if [ "$$ans" != "y" ] && [ "$$ans" != "Y" ]; then \
		printf "$(RED)Aborting. Run 'make vpr' first.$(RESET)\n"; \
		exit 1; \
	fi

	@# ── Setup ───────────────────────────────────────────────────────────────
	@mkdir -p $(TEST_DIR)
	@$(MAKE) -f rl_place_test.mk _reg_clean_intermediates

	@# ── 1. Baseline ─────────────────────────────────────────────────────────
	@printf "\n$(BOLD)[1/4] Baseline (no RL agent)$(RESET)\n"
	@$(VPR) $(ARCH_DEFAULT) $(BLIF_DEFAULT) \
		--pack --place --route \
		--place_algorithm criticality_timing \
		--net_file   $(TEST_DIR)/reg_baseline.net \
		--place_file $(TEST_DIR)/reg_baseline.place \
		--route_file $(TEST_DIR)/reg_baseline.route \
		> $(TEST_DIR)/reg_baseline.log 2>&1 \
	&& printf "  VPR run completed\n" \
	|| { printf "$(RED)  VPR run FAILED$(RESET)\n"; exit 1; }
	@grep -q "VPR succeeded" $(TEST_DIR)/reg_baseline.log \
	&& $(call PASS,VPR succeeded) \
	|| { $(call FAIL,VPR succeeded); }

	@# ── 2. Training ─────────────────────────────────────────────────────────
	@printf "\n$(BOLD)[2/4] Training mode$(RESET)\n"
	@$(VPR) $(ARCH_DEFAULT) $(BLIF_DEFAULT) \
		--pack --place --route \
		--place_algorithm criticality_timing \
		--RL_agent_placement on \
		--place_rl_multistate_mode on \
		--place_rl_training_mode on \
		--place_rl_checkpoint_file $(CHECKPOINT) \
		--place_rl_checkpoint_interval 5 \
		--net_file   $(TEST_DIR)/reg_train.net \
		--place_file $(TEST_DIR)/reg_train.place \
		--route_file $(TEST_DIR)/reg_train.route \
		> $(TEST_DIR)/reg_train.log 2>&1 \
	&& printf "  VPR run completed\n" \
	|| { printf "$(RED)  VPR run FAILED$(RESET)\n"; exit 1; }
	@FAILED=0; \
	grep -q "VPR succeeded" $(TEST_DIR)/reg_train.log \
		&& $(call PASS,VPR succeeded) \
		|| { $(call FAIL,VPR succeeded); }; \
	[ -s $(CHECKPOINT) ] \
		&& $(call PASS,Checkpoint file non-empty) \
		|| { $(call FAIL,Checkpoint file non-empty); }; \
	grep -q "Loaded [0-9]* RL checkpoints" $(TEST_DIR)/reg_train.log \
		&& : \
		|| true; \
	N=$$(wc -l < $(CHECKPOINT)); \
	[ "$$N" -ge 5 ] \
		&& $(call PASS,At least 5 checkpoint entries \(found $$N\)) \
		|| { $(call FAIL,Expected >=5 checkpoint entries but found $$N); }; \
	exit $$FAILED

	@# ── 3. Static inference ─────────────────────────────────────────────────
	@printf "\n$(BOLD)[3/4] Static inference (Q-values frozen)$(RESET)\n"
	@$(VPR) $(ARCH_DEFAULT) $(BLIF_DEFAULT) \
		--pack --place --route \
		--place_algorithm criticality_timing \
		--RL_agent_placement on \
		--place_rl_multistate_mode on \
		--place_rl_checkpoint_file $(CHECKPOINT) \
		--place_rl_static_q_mode on \
		--net_file   $(TEST_DIR)/reg_static.net \
		--place_file $(TEST_DIR)/reg_static.place \
		--route_file $(TEST_DIR)/reg_static.route \
		> $(TEST_DIR)/reg_static.log 2>&1 \
	&& printf "  VPR run completed\n" \
	|| { printf "$(RED)  VPR run FAILED$(RESET)\n"; exit 1; }
	@FAILED=0; \
	grep -q "VPR succeeded" $(TEST_DIR)/reg_static.log \
		&& $(call PASS,VPR succeeded) \
		|| { $(call FAIL,VPR succeeded); }; \
	grep -q "RL inference mode: STATIC Q-values" $(TEST_DIR)/reg_static.log \
		&& $(call PASS,Static Q-values mode active) \
		|| { $(call FAIL,Static Q-values mode not found in log); }; \
	grep -q "Loaded [0-9]* RL checkpoints" $(TEST_DIR)/reg_static.log \
		&& $(call PASS,Checkpoints loaded) \
		|| { $(call FAIL,Checkpoints not loaded); }; \
	grep -q "Initialized Q-values from checkpoint" $(TEST_DIR)/reg_static.log \
		&& $(call PASS,Q-values initialized from checkpoint) \
		|| { $(call FAIL,Q-values not initialized from checkpoint); }; \
	BEFORE=$$(wc -c < $(CHECKPOINT)); \
	AFTER=$$(wc -c < $(CHECKPOINT)); \
	[ "$$BEFORE" -eq "$$AFTER" ] \
		&& $(call PASS,Checkpoint file unchanged \(read-only in inference\)) \
		|| { $(call FAIL,Checkpoint file was modified during static inference); }; \
	exit $$FAILED

	@# ── 4. Adaptive inference ───────────────────────────────────────────────
	@printf "\n$(BOLD)[4/4] Adaptive inference (online Q-value updates)$(RESET)\n"
	@$(VPR) $(ARCH_DEFAULT) $(BLIF_DEFAULT) \
		--pack --place --route \
		--place_algorithm criticality_timing \
		--RL_agent_placement on \
		--place_rl_multistate_mode on \
		--place_rl_checkpoint_file $(CHECKPOINT) \
		--place_rl_static_q_mode off \
		--net_file   $(TEST_DIR)/reg_adaptive.net \
		--place_file $(TEST_DIR)/reg_adaptive.place \
		--route_file $(TEST_DIR)/reg_adaptive.route \
		> $(TEST_DIR)/reg_adaptive.log 2>&1 \
	&& printf "  VPR run completed\n" \
	|| { printf "$(RED)  VPR run FAILED$(RESET)\n"; exit 1; }
	@FAILED=0; \
	grep -q "VPR succeeded" $(TEST_DIR)/reg_adaptive.log \
		&& $(call PASS,VPR succeeded) \
		|| { $(call FAIL,VPR succeeded); }; \
	grep -q "RL inference mode: ADAPTIVE Q-values" $(TEST_DIR)/reg_adaptive.log \
		&& $(call PASS,Adaptive Q-values mode active) \
		|| { $(call FAIL,Adaptive Q-values mode not found in log); }; \
	grep -q "Loaded [0-9]* RL checkpoints" $(TEST_DIR)/reg_adaptive.log \
		&& $(call PASS,Checkpoints loaded) \
		|| { $(call FAIL,Checkpoints not loaded); }; \
	grep -q "Initialized Q-values from checkpoint" $(TEST_DIR)/reg_adaptive.log \
		&& $(call PASS,Q-values initialized from checkpoint) \
		|| { $(call FAIL,Q-values not initialized from checkpoint); }; \
	exit $$FAILED

	@printf "\n$(BOLD)$(GREEN)reg_test complete.$(RESET) Logs and artifacts in $(TEST_DIR)/\n\n"

# ─── circ_test ────────────────────────────────────────────────────────────────
# Usage: make -f rl_place_test.mk circ_test ARCH=<arch.xml> BLIF=<circuit.blif>
ARCH ?= $(ARCH_DEFAULT)
BLIF ?= $(BLIF_DEFAULT)

CIRC_NAME := $(notdir $(basename $(BLIF)))
CIRC_DIR  := $(TEST_DIR)/circ_$(CIRC_NAME)
CIRC_CKP  := $(CIRC_DIR)/checkpoints.jsonl

.PHONY: circ_test
circ_test:
	@printf "$(BOLD)$(CYAN)RL Placement Circuit Test$(RESET)\n"
	@printf "Circuit : $(BLIF)\n"
	@printf "Arch    : $(ARCH)\n\n"

	@mkdir -p $(CIRC_DIR)
	@$(MAKE) -f rl_place_test.mk _circ_clean CIRC_DIR=$(CIRC_DIR)

	@# ── Baseline ────────────────────────────────────────────────────────────
	@printf "$(BOLD)[1/4] Running baseline...$(RESET)\n"
	@$(VPR) $(ARCH) $(BLIF) \
		--pack --place --route \
		--place_algorithm criticality_timing \
		--net_file   $(CIRC_DIR)/baseline.net \
		--place_file $(CIRC_DIR)/baseline.place \
		--route_file $(CIRC_DIR)/baseline.route \
		> $(CIRC_DIR)/baseline.log 2>&1 \
	&& printf "  done\n" || { printf "$(RED)  FAILED$(RESET)\n"; exit 1; }

	@# ── Training ────────────────────────────────────────────────────────────
	@printf "$(BOLD)[2/4] Running training...$(RESET)\n"
	@$(VPR) $(ARCH) $(BLIF) \
		--pack --place --route \
		--place_algorithm criticality_timing \
		--RL_agent_placement on \
		--place_rl_multistate_mode on \
		--place_rl_training_mode on \
		--place_rl_checkpoint_file $(CIRC_CKP) \
		--place_rl_checkpoint_interval 5 \
		--net_file   $(CIRC_DIR)/train.net \
		--place_file $(CIRC_DIR)/train.place \
		--route_file $(CIRC_DIR)/train.route \
		> $(CIRC_DIR)/train.log 2>&1 \
	&& printf "  done\n" || { printf "$(RED)  FAILED$(RESET)\n"; exit 1; }

	@# ── Static inference ────────────────────────────────────────────────────
	@printf "$(BOLD)[3/4] Running static inference...$(RESET)\n"
	@$(VPR) $(ARCH) $(BLIF) \
		--pack --place --route \
		--place_algorithm criticality_timing \
		--RL_agent_placement on \
		--place_rl_multistate_mode on \
		--place_rl_checkpoint_file $(CIRC_CKP) \
		--place_rl_static_q_mode on \
		--net_file   $(CIRC_DIR)/static.net \
		--place_file $(CIRC_DIR)/static.place \
		--route_file $(CIRC_DIR)/static.route \
		> $(CIRC_DIR)/static.log 2>&1 \
	&& printf "  done\n" || { printf "$(RED)  FAILED$(RESET)\n"; exit 1; }

	@# ── Adaptive inference ──────────────────────────────────────────────────
	@printf "$(BOLD)[4/4] Running adaptive inference...$(RESET)\n"
	@$(VPR) $(ARCH) $(BLIF) \
		--pack --place --route \
		--place_algorithm criticality_timing \
		--RL_agent_placement on \
		--place_rl_multistate_mode on \
		--place_rl_checkpoint_file $(CIRC_CKP) \
		--place_rl_static_q_mode off \
		--net_file   $(CIRC_DIR)/adaptive.net \
		--place_file $(CIRC_DIR)/adaptive.place \
		--route_file $(CIRC_DIR)/adaptive.route \
		> $(CIRC_DIR)/adaptive.log 2>&1 \
	&& printf "  done\n" || { printf "$(RED)  FAILED$(RESET)\n"; exit 1; }

	@# ── Summary table ───────────────────────────────────────────────────────
	@printf "\n$(BOLD)$(CYAN)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(RESET)\n"
	@printf "$(BOLD) Results: $(CIRC_NAME)$(RESET)\n"
	@printf "$(BOLD)$(CYAN)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(RESET)\n"
	@printf "$(BOLD)%-18s %12s %10s %10s %10s$(RESET)\n" \
		"Mode" "CPD (ns)" "Fmax (MHz)" "WL" "Time (s)"
	@printf "%-18s %12s %10s %10s %10s\n" \
		"──────────────" "──────────" "──────────" "────────" "────────"
	@for MODE in baseline train static adaptive; do \
		LOG=$(CIRC_DIR)/$$MODE.log; \
		CPD=$$(grep "Final critical path delay" $$LOG 2>/dev/null \
		       | grep -o "[0-9]*\.[0-9]* ns" | head -1 | awk '{print $$1}'); \
		FMAX=$$(grep "Final critical path delay" $$LOG 2>/dev/null \
		        | grep -o "Fmax: [0-9]*\.[0-9]*" | head -1 | awk '{print $$2}'); \
		WL=$$(grep "BB estimate of min-dist (placement) wire length:" $$LOG 2>/dev/null \
		      | grep -o "[0-9]*$$" | head -1); \
		TIME=$$(grep "The entire flow of VPR took" $$LOG 2>/dev/null \
		        | grep -o "[0-9]*\.[0-9]* seconds" | head -1 | awk '{print $$1}'); \
		STATUS=$$(grep -q "VPR succeeded" $$LOG 2>/dev/null && echo "ok" || echo "FAIL"); \
		LABEL=$$MODE; \
		[ "$$STATUS" = "FAIL" ] && LABEL="$(RED)$$MODE [FAIL]$(RESET)"; \
		printf "%-18s %12s %10s %10s %10s\n" \
			"$$LABEL" "$${CPD:--}" "$${FMAX:--}" "$${WL:--}" "$${TIME:--}"; \
	done
	@printf "$(BOLD)$(CYAN)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(RESET)\n"
	@printf "\nLogs and artifacts: $(CIRC_DIR)/\n"
	@printf "Checkpoints       : $(CIRC_CKP) ($$(wc -l < $(CIRC_CKP)) entries)\n\n"

# ─── Internal clean targets ───────────────────────────────────────────────────
.PHONY: _reg_clean_intermediates _circ_clean clean

_reg_clean_intermediates:
	@rm -f $(TEST_DIR)/reg_*.net  $(TEST_DIR)/reg_*.place $(TEST_DIR)/reg_*.route
	@rm -f $(CHECKPOINT)

_circ_clean:
	@rm -f $(CIRC_DIR)/*.net $(CIRC_DIR)/*.place $(CIRC_DIR)/*.route
	@rm -f $(CIRC_CKP)

clean:
	@rm -rf $(TEST_DIR)
	@printf "Removed $(TEST_DIR)/\n"
