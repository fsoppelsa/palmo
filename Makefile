.PHONY: deps tests flask train-bohemia train-sherlock train-malpelo monitor-bohemia monitor-sherlock monitor-malpelo cleanup

deps:
	pip install -r requirements.txt

tests:
	PYTHONPATH=. pytest -s tests/

flask:
	python -m src.app

# Pretraining targets
train-bohemia:
	./pretrain/run_pretraining_detached.sh bohemia 10

train-sherlock:
	./pretrain/run_pretraining_detached.sh sherlock 20

# Monitoring targets
monitor-bohemia:
	./pretrain/monitor_pretraining.sh bohemia

monitor-sherlock:
	./pretrain/monitor_pretraining.sh sherlock

# Cleanup
cleanup:
	@echo "Removing all checkpoint and log files..."
	rm -f checkpoints/*.pt
	rm -f checkpoints/*.log
	rm -f checkpoints/*.pid
	@echo "Cleanup complete!"

# Show help
help:
	@echo "Palmo Makefile Commands:"
	@echo ""
	@echo "  make deps              - Install dependencies"
	@echo "  make tests             - Run tests"
	@echo "  make flask             - Start Flask app"
	@echo ""
	@echo "  make train-bohemia     - Train on Bohemia corpus (detached)"
	@echo "  make train-sherlock    - Train on Sherlock corpus (detached)"
	@echo ""
	@echo "  make monitor-bohemia   - Monitor Bohemia training"
	@echo "  make monitor-sherlock  - Monitor Sherlock training"
	@echo ""
	@echo "  make cleanup           - Remove all checkpoint .pt files"
	@echo ""
