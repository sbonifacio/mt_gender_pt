"""
Usage: separate translation files into pro-stereotypical and anti-stereotypical translations 
"""
import pdb
def process_file(trans_full, trans_pro, trans_anti):
    with open("tmp_anti.txt", 'r') as en_anti:
            en_anti_lines = en_anti.readlines()

    with open("tmp_pro.txt", 'r') as en_pro:
            en_pro_lines = en_pro.readlines()

    # Lists to hold lines for each output file
    pro_lines = []
    anti_lines = []

    # Classify lines
    with open(trans_full, 'r') as file:
        for line in file:
            line_en = line.split(" ||| ")[0]  
            line_en += "\n"
            #pdb.set_trace()
            if line_en in en_anti_lines:
                anti_lines.append(line.strip()) 
            elif line_en in en_pro_lines:
                pro_lines.append(line.strip())   

    # Write pro lines
    with open(trans_pro, 'w') as pro_file:
        for line in pro_lines:
            pro_file.write(line + '\n')

    # Write anti lines
    with open(trans_anti, 'w') as anti_file:
        for line in anti_lines:
            anti_file.write(line + '\n')



input_file = '../translations/google/en-es.txt'           # Path to all translations
output_pro = '../translations/google/en-es_pro.txt'  # Path to file with pro-stereotypical translations
output_anti = '../translations/google/en-es_anti.txt'       # Path to file with anti-stereotypical translations

process_file(input_file, output_pro, output_anti)