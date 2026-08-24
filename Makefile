.PHONY: lock setup setup-tabular setup-generative setup-llm setup-reinforcement setup-reasoning setup-evaluation test test-generative test-llm test-reinforcement test-reasoning test-evaluation benchmark benchmark-tabular benchmark-generative visual-generative benchmark-reinforcement benchmark-evaluation demo-reinforcement audit-ppo audit-rlvr download-llm download-llm-data download-reasoning download-reasoning-data download-evaluation-data prepare-llm prepare-llm-data smoke-llm smoke-reasoning pretrain-llm pretrain-llm-multi-gpu sft-reasoning evaluate-reasoning train-math-capstone data-reinforcement quick quick-tabular quick-generative quick-visual-generative quick-reinforcement report-llm report-reasoning report-reinforcement report-evaluation notes-tabular notes-generative notes-llm notes-reasoning notes-reinforcement notes-evaluation homework-tabular homework-generative homework-llm homework-reasoning homework-reinforcement homework-evaluation validate-homework-tabular validate-homework-generative validate-homework-llm validate-homework-reasoning validate-homework-reinforcement validate-homework-evaluation verify-slides-tabular verify-slides-generative verify-slides-llm verify-slides-reasoning verify-slides-reinforcement verify-slides-evaluation clean-latex
.PHONY: setup-gnn test-gnn benchmark-gnn data-gnn quick-gnn report-gnn notes-gnn homework-gnn validate-homework-gnn verify-slides-gnn

REPO_LIBOMP := $(CURDIR)/.local/libomp/18.1.5/lib
ARM_BREW_LIBOMP := /opt/homebrew/opt/libomp/lib
INTEL_BREW_LIBOMP := /usr/local/opt/libomp/lib
TABULAR_LIBOMP_DIR ?= $(shell if [ -f "$(REPO_LIBOMP)/libomp.dylib" ]; then printf '%s' "$(REPO_LIBOMP)"; elif [ -f "$(ARM_BREW_LIBOMP)/libomp.dylib" ]; then printf '%s' "$(ARM_BREW_LIBOMP)"; else printf '%s' "$(INTEL_BREW_LIBOMP)"; fi)
TABULAR_RUNTIME := DYLD_LIBRARY_PATH="$(TABULAR_LIBOMP_DIR)"

UV_TABULAR := $(TABULAR_RUNTIME) uv run --extra tabular
UV_GENERATIVE := uv run --extra generative
UV_GNN := uv run --extra gnn
UV_LLM := uv run --extra llm
UV_REINFORCEMENT := uv run --extra rl
UV_REASONING := uv run --extra reasoning
UV_EVALUATION := uv run --extra evaluation
UV_ALL := $(TABULAR_RUNTIME) uv run --extra tabular --extra generative --extra gnn --extra llm --extra rl --extra reasoning --extra evaluation
NUMPY_THREADS := OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
LLM_BOOK_LIMIT ?= 16
LLM_REQUEST_DELAY ?= 2
LLM_SELECTION_SEED ?=
LLM_BOOK_URL_TEMPLATE ?=
LLM_NPROC_PER_NODE ?= 2
LLM_MASTER_PORT ?= 29500
LLM_VOCAB_SIZE ?= 512
LLM_TOKENIZER_CHARACTERS ?= 500000
LLM_SELECTION_SEED_ARG := $(if $(strip $(LLM_SELECTION_SEED)),--selection-seed $(LLM_SELECTION_SEED),)
LLM_BOOK_URL_TEMPLATE_ARG := $(if $(strip $(LLM_BOOK_URL_TEMPLATE)),--book-url-template '$(LLM_BOOK_URL_TEMPLATE)',)

lock:
	uv lock

setup:
	uv sync --locked --extra tabular --extra generative --extra gnn --extra llm --extra rl --extra reasoning --extra evaluation

setup-tabular:
	uv sync --locked --extra tabular

setup-generative:
	uv sync --locked --extra generative

setup-gnn:
	uv sync --locked --extra gnn

setup-llm:
	uv sync --locked --extra llm

setup-reinforcement:
	uv sync --locked --extra rl

setup-reasoning:
	uv sync --locked --extra reasoning

setup-evaluation:
	uv sync --locked --extra evaluation

test:
	$(NUMPY_THREADS) MPLCONFIGDIR=.cache/matplotlib $(UV_ALL) python -m unittest discover -s tests -v

test-generative:
	$(NUMPY_THREADS) MPLCONFIGDIR=.cache/matplotlib $(UV_GENERATIVE) python -m unittest discover -s tests/generative -v

test-gnn:
	$(NUMPY_THREADS) MPLCONFIGDIR=.cache/matplotlib $(UV_GNN) python -m unittest discover -s tests/gnn -v

test-llm:
	$(NUMPY_THREADS) MPLCONFIGDIR=.cache/matplotlib $(UV_LLM) python -m unittest discover -s tests/llm -v

test-reinforcement:
	$(NUMPY_THREADS) MPLCONFIGDIR=.cache/matplotlib $(UV_REINFORCEMENT) python -m unittest discover -s tests/reinforcement -v

test-reasoning:
	$(NUMPY_THREADS) MPLCONFIGDIR=.cache/matplotlib $(UV_REASONING) python -m unittest discover -s tests/reasoning -v

test-evaluation:
	$(NUMPY_THREADS) MPLCONFIGDIR=.cache/matplotlib $(UV_EVALUATION) python -m unittest discover -s tests/evaluation -v

benchmark: benchmark-tabular benchmark-generative benchmark-gnn benchmark-reinforcement

benchmark-tabular:
	$(NUMPY_THREADS) MPLCONFIGDIR=.cache/matplotlib $(UV_TABULAR) python -m experiments.tabular.run_benchmark

visual-generative:
	$(NUMPY_THREADS) MPLCONFIGDIR=.cache/matplotlib $(UV_GENERATIVE) python -m experiments.generative.run_visual_examples --device auto

benchmark-generative: visual-generative
	$(NUMPY_THREADS) MPLCONFIGDIR=.cache/matplotlib $(UV_GENERATIVE) python -m experiments.generative.run_benchmark --bootstrap 2000

benchmark-gnn:
	$(NUMPY_THREADS) MPLCONFIGDIR=.cache/matplotlib $(UV_GNN) python -m experiments.gnn.run_benchmark --device auto

benchmark-reinforcement:
	$(NUMPY_THREADS) MPLCONFIGDIR=.cache/matplotlib $(UV_REINFORCEMENT) python -m experiments.reinforcement.run_benchmark --device auto

benchmark-evaluation:
	$(NUMPY_THREADS) MPLCONFIGDIR=.cache/matplotlib $(UV_EVALUATION) python -m experiments.evaluation.run_benchmark

demo-reinforcement:
	$(NUMPY_THREADS) $(UV_REINFORCEMENT) python -m experiments.reinforcement.play_policy --algorithm dqn --device auto

audit-ppo:
	$(NUMPY_THREADS) MPLCONFIGDIR=.cache/matplotlib $(UV_REINFORCEMENT) python -m experiments.reinforcement.ppo_token_reweighting

audit-rlvr:
	$(NUMPY_THREADS) MPLCONFIGDIR=.cache/matplotlib $(UV_REASONING) python -m experiments.reasoning.compare_rlvr_objectives

download-llm:
	$(UV_LLM) python -m experiments.llm.download_data --dataset all \
		--book-limit $(LLM_BOOK_LIMIT) --request-delay $(LLM_REQUEST_DELAY) \
		$(LLM_SELECTION_SEED_ARG) $(LLM_BOOK_URL_TEMPLATE_ARG)

download-llm-data: download-llm

download-reasoning:
	$(UV_REASONING) python -m experiments.reasoning.download_data

download-reasoning-data: download-reasoning

download-evaluation-data:
	$(UV_EVALUATION) python -m experiments.evaluation.download_data

prepare-llm:
	$(UV_LLM) python -m experiments.llm.prepare_data --tokenizer bpe \
		--vocab-size $(LLM_VOCAB_SIZE) \
		--tokenizer-characters $(LLM_TOKENIZER_CHARACTERS)

prepare-llm-data: prepare-llm

smoke-llm:
	$(NUMPY_THREADS) $(UV_LLM) python -m experiments.llm.run_smoke

smoke-reasoning:
	$(NUMPY_THREADS) $(UV_REASONING) python -m experiments.reasoning.run_smoke

sft-reasoning:
	$(UV_REASONING) python -m experiments.reasoning.sft --preset mps --device auto

train-math-capstone:
	$(UV_EVALUATION) python -m experiments.evaluation.train_math_capstone --preset mps --device auto

evaluate-reasoning:
	$(UV_REASONING) python -m experiments.reasoning.evaluate --dataset gsm8k --device auto --limit 128 --samples 1 --output-name base_gsm8k_128

pretrain-llm:
	$(UV_LLM) python -m experiments.llm.pretrain --preset mps --device auto

pretrain-llm-multi-gpu:
	$(UV_LLM) torchrun --nnodes=1 --nproc-per-node=$(LLM_NPROC_PER_NODE) \
		--master-addr=127.0.0.1 --master-port=$(LLM_MASTER_PORT) \
		-m experiments.llm.pretrain_distributed --preset colab --device cuda

data-reinforcement:
	$(UV_REINFORCEMENT) python -m experiments.reinforcement.collect_replay

data-gnn:
	$(UV_GNN) python -m experiments.gnn.download_data
	$(UV_GNN) python -m experiments.gnn.prepare_cora_homework

quick: quick-tabular quick-generative quick-gnn quick-reinforcement

quick-tabular:
	$(NUMPY_THREADS) MPLCONFIGDIR=.cache/matplotlib $(UV_TABULAR) python -m experiments.tabular.run_benchmark --quick --bootstrap 200

quick-visual-generative:
	$(NUMPY_THREADS) MPLCONFIGDIR=.cache/matplotlib $(UV_GENERATIVE) python -m experiments.generative.run_visual_examples --quick --device auto

quick-generative: quick-visual-generative
	$(NUMPY_THREADS) MPLCONFIGDIR=.cache/matplotlib $(UV_GENERATIVE) python -m experiments.generative.run_benchmark --quick --bootstrap 200

quick-gnn:
	$(NUMPY_THREADS) MPLCONFIGDIR=.cache/matplotlib $(UV_GNN) python -m experiments.gnn.run_benchmark --quick --device auto

quick-reinforcement:
	$(NUMPY_THREADS) MPLCONFIGDIR=.cache/matplotlib $(UV_REINFORCEMENT) python -m experiments.reinforcement.run_benchmark --quick --device auto

report-reinforcement:
	LC_ALL=C LANG=C latexmk -pdf -interaction=nonstopmode -halt-on-error -cd output/pdf/reinforcement_learning_report.tex

report-llm:
	LC_ALL=C LANG=C latexmk -pdf -interaction=nonstopmode -halt-on-error -cd output/pdf/llm_from_scratch_report.tex

report-gnn:
	LC_ALL=C LANG=C latexmk -pdf -interaction=nonstopmode -halt-on-error -cd output/pdf/gnn_geometric_learning_report.tex

report-reasoning:
	LC_ALL=C LANG=C latexmk -pdf -interaction=nonstopmode -halt-on-error -cd output/pdf/reasoning_models_report.tex

report-evaluation:
	LC_ALL=C LANG=C latexmk -pdf -interaction=nonstopmode -halt-on-error -cd output/pdf/ai_evaluation_benchmarking_report.tex

notes-tabular:
	LC_ALL=C LANG=C latexmk -pdf -interaction=nonstopmode -halt-on-error -cd notes/tabular/lecture_notes.tex

notes-generative:
	LC_ALL=C LANG=C latexmk -pdf -interaction=nonstopmode -halt-on-error -cd notes/generative/lecture_notes.tex

notes-gnn:
	LC_ALL=C LANG=C latexmk -pdf -interaction=nonstopmode -halt-on-error -cd notes/gnn/lecture_notes.tex

notes-llm:
	LC_ALL=C LANG=C latexmk -pdf -interaction=nonstopmode -halt-on-error -cd notes/llm/lecture_notes.tex

notes-reinforcement:
	LC_ALL=C LANG=C latexmk -pdf -interaction=nonstopmode -halt-on-error -cd notes/reinforcement/lecture_notes.tex

notes-reasoning:
	LC_ALL=C LANG=C latexmk -pdf -interaction=nonstopmode -halt-on-error -cd notes/reasoning/lecture_notes.tex

notes-evaluation:
	LC_ALL=C LANG=C latexmk -pdf -interaction=nonstopmode -halt-on-error -cd notes/evaluation/lecture_notes.tex

homework-tabular:
	LC_ALL=C LANG=C latexmk -pdf -interaction=nonstopmode -halt-on-error -cd homework/tabular/homework.tex
	LC_ALL=C LANG=C latexmk -pdf -interaction=nonstopmode -halt-on-error -cd homework/tabular/solutions.tex

homework-generative:
	LC_ALL=C LANG=C latexmk -pdf -interaction=nonstopmode -halt-on-error -cd homework/generative/homework.tex
	LC_ALL=C LANG=C latexmk -pdf -interaction=nonstopmode -halt-on-error -cd homework/generative/solutions.tex

homework-gnn:
	LC_ALL=C LANG=C latexmk -pdf -interaction=nonstopmode -halt-on-error -cd homework/gnn/homework.tex
	LC_ALL=C LANG=C latexmk -pdf -interaction=nonstopmode -halt-on-error -cd homework/gnn/solutions.tex

homework-llm:
	LC_ALL=C LANG=C latexmk -pdf -interaction=nonstopmode -halt-on-error -cd homework/llm/homework.tex
	LC_ALL=C LANG=C latexmk -pdf -interaction=nonstopmode -halt-on-error -cd homework/llm/solutions.tex

homework-reinforcement:
	LC_ALL=C LANG=C latexmk -pdf -interaction=nonstopmode -halt-on-error -cd homework/reinforcement/homework.tex
	LC_ALL=C LANG=C latexmk -pdf -interaction=nonstopmode -halt-on-error -cd homework/reinforcement/solutions.tex

homework-reasoning:
	LC_ALL=C LANG=C latexmk -pdf -interaction=nonstopmode -halt-on-error -cd homework/reasoning/homework.tex
	LC_ALL=C LANG=C latexmk -pdf -interaction=nonstopmode -halt-on-error -cd homework/reasoning/solutions.tex

homework-evaluation:
	LC_ALL=C LANG=C latexmk -pdf -interaction=nonstopmode -halt-on-error -cd homework/evaluation/homework.tex
	LC_ALL=C LANG=C latexmk -pdf -interaction=nonstopmode -halt-on-error -cd homework/evaluation/solutions.tex

validate-homework-tabular:
	$(NUMPY_THREADS) MPLCONFIGDIR=.cache/matplotlib $(UV_TABULAR) python homework/tabular/solutions/validate_solutions.py

validate-homework-generative:
	$(NUMPY_THREADS) MPLCONFIGDIR=.cache/matplotlib $(UV_GENERATIVE) python homework/generative/solutions/validate_solutions.py

validate-homework-gnn:
	$(NUMPY_THREADS) $(UV_GNN) python homework/gnn/solutions/validate_solutions.py

validate-homework-llm:
	$(NUMPY_THREADS) $(UV_LLM) python homework/llm/solutions/validate_solutions.py

validate-homework-reinforcement:
	$(NUMPY_THREADS) $(UV_REINFORCEMENT) python homework/reinforcement/solutions/validate_solutions.py

validate-homework-reasoning:
	$(NUMPY_THREADS) $(UV_REASONING) python homework/reasoning/solutions/validate_solutions.py

validate-homework-evaluation:
	$(NUMPY_THREADS) MPLCONFIGDIR=.cache/matplotlib $(UV_EVALUATION) python homework/evaluation/solutions/validate_solutions.py

verify-slides-tabular:
	uv run python tools/verify_slide_artifacts.py \
		--key slides/tabular/tabular_models_teaching.key \
		--pdf slides/tabular/tabular_models_teaching.pdf \
		--expected-pages 15

verify-slides-generative:
	uv run python tools/verify_slide_artifacts.py \
		--key slides/generative/generative_models_teaching.key \
		--pdf slides/generative/generative_models_teaching.pdf \
		--expected-pages 48 \
		--expected-equations 55

verify-slides-gnn:
	uv run python tools/verify_slide_artifacts.py \
		--pptx slides/gnn/gnn_geometric_learning_teaching.pptx \
		--pdf slides/gnn/gnn_geometric_learning_teaching.pdf \
		--expected-pages 20 \
		--expected-equations 8

verify-slides-llm:
	uv run python tools/verify_slide_artifacts.py \
		--pptx slides/llm/source/llm_from_scratch_teaching.pptx \
		--pdf slides/llm/llm_from_scratch_teaching.pdf \
		--expected-pages 28 \
		--expected-equations 5

verify-slides-reinforcement:
	uv run python tools/verify_slide_artifacts.py \
		--key slides/reinforcement/reinforcement_learning_teaching.key \
		--pdf slides/reinforcement/reinforcement_learning_teaching.pdf \
		--expected-pages 39 \
		--expected-equations 14

verify-slides-reasoning:
	uv run python tools/verify_slide_artifacts.py \
		--key slides/reasoning/reasoning_models_teaching.key \
		--pdf slides/reasoning/reasoning_models_teaching.pdf \
		--expected-pages 32 \
		--expected-equations 15

verify-slides-evaluation:
	uv run python tools/verify_slide_artifacts.py \
		--pptx slides/evaluation/source/ai_evaluation_benchmarking_teaching.pptx \
		--pdf slides/evaluation/ai_evaluation_benchmarking_teaching.pdf \
		--expected-pages 26

clean-latex:
	LC_ALL=C LANG=C latexmk -C -cd output/pdf/tabular_models_report.tex
	LC_ALL=C LANG=C latexmk -C -cd output/pdf/generative_models_report.tex
	LC_ALL=C LANG=C latexmk -C -cd output/pdf/gnn_geometric_learning_report.tex
	LC_ALL=C LANG=C latexmk -C -cd notes/tabular/lecture_notes.tex
	LC_ALL=C LANG=C latexmk -C -cd notes/generative/lecture_notes.tex
	LC_ALL=C LANG=C latexmk -C -cd notes/gnn/lecture_notes.tex
	LC_ALL=C LANG=C latexmk -C -cd homework/tabular/homework.tex
	LC_ALL=C LANG=C latexmk -C -cd homework/tabular/solutions.tex
	LC_ALL=C LANG=C latexmk -C -cd homework/generative/homework.tex
	LC_ALL=C LANG=C latexmk -C -cd homework/generative/solutions.tex
	LC_ALL=C LANG=C latexmk -C -cd homework/gnn/homework.tex
	LC_ALL=C LANG=C latexmk -C -cd homework/gnn/solutions.tex
	LC_ALL=C LANG=C latexmk -C -cd output/pdf/llm_from_scratch_report.tex
	LC_ALL=C LANG=C latexmk -C -cd notes/llm/lecture_notes.tex
	LC_ALL=C LANG=C latexmk -C -cd homework/llm/homework.tex
	LC_ALL=C LANG=C latexmk -C -cd homework/llm/solutions.tex
	LC_ALL=C LANG=C latexmk -C -cd output/pdf/reinforcement_learning_report.tex
	LC_ALL=C LANG=C latexmk -C -cd output/pdf/reasoning_models_report.tex
	LC_ALL=C LANG=C latexmk -C -cd notes/reinforcement/lecture_notes.tex
	LC_ALL=C LANG=C latexmk -C -cd notes/reasoning/lecture_notes.tex
	LC_ALL=C LANG=C latexmk -C -cd homework/reinforcement/homework.tex
	LC_ALL=C LANG=C latexmk -C -cd homework/reinforcement/solutions.tex
	LC_ALL=C LANG=C latexmk -C -cd homework/reasoning/homework.tex
	LC_ALL=C LANG=C latexmk -C -cd homework/reasoning/solutions.tex
	LC_ALL=C LANG=C latexmk -C -cd output/pdf/ai_evaluation_benchmarking_report.tex
	LC_ALL=C LANG=C latexmk -C -cd notes/evaluation/lecture_notes.tex
	LC_ALL=C LANG=C latexmk -C -cd homework/evaluation/homework.tex
	LC_ALL=C LANG=C latexmk -C -cd homework/evaluation/solutions.tex
