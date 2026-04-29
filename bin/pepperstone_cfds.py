import sys
import os
import re


extract_symbol = lambda x: x[1:x.index('.')  ]
    
if __name__ == "__main__":
    source_html_file = sys.argv[1]
    target_csv_file = sys.argv[2]
    cfds = []

    with open(source_html_file, 'r') as source_file:
        with open(target_csv_file, 'w') as target_file:
            for line in source_file:
                matches = re.findall(r'\(\w+\.\w\)', line)
                if matches:
                    cfds.extend(matches)
            tickers = list(map(extract_symbol, cfds))
            for ticker in tickers:
                print(ticker, file=target_file)
    

            