.PHONY: test-torch test-tf test-torch-fast test-tf-fast test-data

# Run inside the matching virtualenv (see README "Setup").
test-torch:
	python -m pytest tests torch_impl

test-tf:
	python -m pytest tests tf_impl

# Skip the Day 13 overfit test (trains for ~600 steps) for a quick iteration loop.
test-torch-fast:
	python -m pytest tests torch_impl -m "not slow"

test-tf-fast:
	python -m pytest tests tf_impl -m "not slow"

# Framework-free tests (tokenizer + data pipeline): need neither torch nor tensorflow.
test-data:
	python -m pytest tests/test_bpe_tokenizer.py tests/test_parallel_data.py
