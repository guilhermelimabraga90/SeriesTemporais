.PHONY: install
install: ## Install Python requirements.
	python -m pip install --upgrade pip setuptools wheel poetry
	poetry lock
	poetry install --no-root
	poetry run pre-commit install

.PHONY: runRNN
runRNN: ## Run the project.
	poetry run python -m src.app.Rnn

.PHONY: runLSTM
runLSTM: ## Run the project.
	poetry run python -m src.app.LSTM

.PHONY: runGRU
runGRU: ## Run the project.
	poetry run python -m src.app.GRU

.PHONY: format
format: ## Format code.
	poetry run autoflake --remove-all-unused-imports --recursive --remove-unused-variables --in-place src
	poetry run isort src
	poetry run black src