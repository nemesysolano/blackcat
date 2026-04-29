import os
import sys

if __name__ == "__main__":
    split_files_dir = sys.argv[1]
    prefix = sys.argv[2]
    suffixes = sys.argv[3:]
    
    output_file_name = os.path.join(split_files_dir, f"{prefix}.csv")    
    if os.path.exists(output_file_name):
        os.remove(output_file_name)

    with open(output_file_name, 'w') as output_file:
        existing = set()
        for suffix in suffixes:
            input_file_name = os.path.join(split_files_dir, f"{prefix}-{suffix}.csv")            
            with open (input_file_name, 'r') as input_file:
                for line in input_file:                    
                    line = line.strip()                                        
                    ticker = line[0:line.index(',')]
                    if not (ticker in existing):                        
                        print(line, file=output_file)
                        existing.add(ticker)
                
        

        