deploy:
	git ls-files | rsync -azP --files-from=- . csft:kasl
	ssh csft "cd kasl && PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring /home/diego/.local/bin/poetry install"