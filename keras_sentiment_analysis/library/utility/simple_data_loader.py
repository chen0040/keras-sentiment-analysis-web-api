
def load_text_label_pairs(data_file_path):
    file = open(data_file_path, mode='rt', encoding='utf8')
    result = []
    for line in file:
        label, sentence = line.strip().split('\t')
        result.append((sentence, label))
    return result
