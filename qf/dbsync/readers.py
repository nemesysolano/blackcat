
def read_quote_names(file_path):
    with open(file_path, 'r') as f:
        quote_names = [line.strip() for line in f]
    return quote_names