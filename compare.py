import sys

if len(sys.argv) != 3:
    sys.exit("Usage: python compare.py file1 file2")

with open(sys.argv[1], 'r') as file1, open(sys.argv[2], 'r') as file2:
    same = True
    num_dif = 0
    off_words = {}

    for line_number, (line1, line2) in enumerate(zip(file1, file2), start=1):
        if line1.strip() != line2.strip():
            same = False
            num_dif +=1
            # print the first line that includes a difference
            print(f"Difference found at line {line_number}:\n")
            for i, (val1, val2) in enumerate(zip(line1.rstrip().split(), line2.rstrip().split())):
                if val1 != val2:
                    print(f"index: {i}")
                    print(f"File 1: {val1}")
                    print(f"File 2: {val2}\n")

                    # trying to see which words are not being retrofitted well
                    if abs(float(val1) - float(val2)) > 0.0002:
                        if line_number not in off_words.keys():
                            off_words[line_number] = line1.split()[0]
    
    print(f"Number of different components: {num_dif}")
    print(f"Number of lines with significant differences: {len(off_words)}")
    print(f"Words creating the significant differences:\n{off_words.values()}")

    # TODO: compare vectors between the pairs of odd words to see if the difference is a bad thing or an improvement

    if same:
        print("\nThe contents of the two files are the same\n")
