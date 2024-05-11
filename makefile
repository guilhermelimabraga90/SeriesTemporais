.PHONY: install2
install: ## Install Python requirements.
	pip install tensorflow
	pip install sklearn
	pip install matplotlib



.PHONY: runRNN
run: ## Run the project.
	 python3 ./src/app/Rnn.py

.PHONY: runLSTM
run: ## Run the project.
	python3 ./src/app/LSTM.py

.PHONY: runGRU
run: ## Run the project.
 	python3 ./src/app/GRU.py