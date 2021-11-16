def parse_file(file_path):
    lines = []
    # sentence = "<start> "
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip('\n')
            if len(line) == 0:
                continue
            # index = line.find('.')
            # while index != -1:
            #     sentence += line[: index] + " <end>"
            #     lines.append((sentence.split(), sentence.split()))
            #     sentence = "<start> "
            #     if(index+1 < len(line)):
            #         line = line[index+1:]
            #     else:
            #         line = ""
            #         break
            #     index = line.find('.')
            # sentence += line
            sentence = "<start> " + line + " <end>"
            lines.append((sentence.split(), sentence.split()))
    
    return lines

# before_parsed_data = parse_file("./day_3/test_file.txt")
# print(before_parsed_data[0])