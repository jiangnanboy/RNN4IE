import os
import csv

def process_val(dev_path, source_path, target_path):
    with open(dev_path, 'r', encoding='utf-8') as dev_read, open(source_path, 'w', encoding='utf-8') as source_write, open(target_path, 'w', encoding='utf-8') as target_write:
        source_line = ''
        target_line = ''
        for token in dev_read:
            if len(token.strip()) == 0:
                source_write.write(source_line.strip())
                source_write.write('\n')
                target_write.write(target_line.strip())
                target_write.write('\n')
                source_line = ''
                target_line = ''
            else:
                tokens = token.split()
                source_word = tokens[0].strip()
                source_word += ' '
                target_word = tokens[1].strip()
                target_word += ' '
                source_line += source_word
                target_line += target_word
        if source_line != '' and target_line != '':
            source_write.write(source_line.strip())
            source_write.write('\n')
            target_write.write(target_line.strip())
            target_write.write('\n')
            source_line = ''
            target_line = ''
        print('process dev done!')



def val2csv(train_source, train_target, save_path):
    with open(train_source, 'r', encoding='utf-8') as source_read, open(train_target, 'r', encoding='utf-8') as target_read:
        with open(save_path, 'w', encoding='utf-8', newline='') as csv_write:
            header = ['label', 'source', 'target']
            csv_writer = csv.writer(csv_write)
            csv_writer.writerow(header)
            for source, target in zip(source_read, target_read):
                csv_writer.writerow([None, source, target])
    print("done!")

if __name__ == '__main__':

    source_path = os.path.join(os.getcwd(), 'dev_source.txt')
    target_path = os.path.join(os.getcwd(), 'dev_target.txt')
    save_path = os.path.join(os.getcwd(), 'validation.csv')
    val2csv(source_path, target_path, save_path)


    '''
    source_path = os.path.join(os.getcwd(), 'dev_source.txt')
    target_path = os.path.join(os.getcwd(), 'dev_target.txt')
    dev_path = os.path.join(os.getcwd(), 'dev.conll')
    process_val(dev_path, source_path, target_path)
    '''