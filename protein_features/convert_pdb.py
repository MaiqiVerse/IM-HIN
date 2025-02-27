import os
file_list = os.listdir("./pdb_files/")

os.makedirs("./coordinates/", exist_ok=True)
for file in file_list:
    file_tmp = []
    with open(os.path.join("./pdb_files/", file), 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith('ATOM'):
                file_tmp.append(line)
        new_file_tmp = []
        for line in file_tmp:
            x = line[30:38]
            y = line[38:46]
            z = line[46:54]
            temp = line[60:66]
            new_file_tmp.append([float(x), float(y), float(z), float(temp)])
                
        file_tmp = new_file_tmp
        # write to new txt file
        with open(os.path.join("./coordinates/", file.replace('.pdb', '.txt')), 'w') as f:
            for line in file_tmp:
                # f.write(' '.join(line) + '\n')
                f.write(','.join(map(str, line)) + '\n')