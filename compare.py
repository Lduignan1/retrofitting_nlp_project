import sys

if len(sys.argv) != 3:
    sys.exit("Usage: python compare.py file1 file2")

with open(sys.argv[1], 'r') as file1, open(sys.argv[2], 'r') as file2:
    same = True
    num_dif = 0
    large_val = 0
    large_vec = 0
    large_v = False
    for line_number, (line1, line2) in enumerate(zip(file1, file2), start=1):
        if line1.strip() != line2.strip():
            same = False
            num_dif +=1
            # print the first line that includes a difference
            print(f"Difference found at line {line_number}:\n")
            for i, (val1, val2) in enumerate(zip(line1.rstrip().split(), line2.rstrip().split())):
                if val1 != val2:
                    if float(val2) > 1:
                        large_val += 1
                        large_v = True
                    print(f"index: {i}")
                    print(f"File 1: {val1}")
                    print(f"File 2: {val2}\n")
            if large_v:
                large_vec += 1
            large_v = False
    print(num_dif)
    print(large_val)
    print(large_vec)
    if same:
        print("\nThe contents of the two files are the same\n")
