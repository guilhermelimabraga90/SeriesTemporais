.PHONY: install
install: ## Install Python requirements.
	pip install tensorflow
	pip install sklearn
	pip install matplotlib

.PHONY: runRNN
runRNN: ## Run the project.
	python3 ./src/app/Rnn.py

.PHONY: runLSTM
runLSTM: ## Run the project.
	python3 ./src/app/LSTM.py

.PHONY: runGRU
runGRU: ## Run the project.
	python3 ./src/app/GRU.py
