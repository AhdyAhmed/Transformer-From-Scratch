.PHONY: test-torch test-tf

# Run inside the matching virtualenv (see README "Setup").
test-torch:
	python -m pytest tests torch_impl

test-tf:
	python -m pytest tests tf_impl
