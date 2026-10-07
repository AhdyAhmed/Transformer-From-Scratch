.PHONY: test-torch test-tf test-torch-fast test-tf-fast

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
