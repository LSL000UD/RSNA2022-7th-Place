import os


def print_log_to_text_file(output_log_path, text, mode, terminal_only=False):
    if not terminal_only:
        if not os.path.exists(output_log_path):
            mode = 'w'
        with open(output_log_path, mode) as f:
            f.write(text)

    # Also display log in the terminal
    print(text, end='', flush=True)


