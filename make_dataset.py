import numpy as np
import pandas as pd

import os
import xml.etree.ElementTree as ET
import codecs

def load_file(path):
    with codecs.open(path, 'r', 'utf-8', 'ignore') as f:
        lines = f.readlines()
        lines = ''.join(lines).replace('\n', ' ')
    return lines

def in_sentence(element):
    for sub in element:
        if sub.text and sub.text != '\n' and sub.text != '':
            yield sub.text
        if sub.tail and sub.tail != '\n' and sub.tail != '':
            yield sub.tail
    for sub in element:
        for text in in_sentence(sub):
            yield text

def extract_sentence(element):
    for sub in element:
        sentence = ''
        if sub.tag == 'sentence':
            if sub.text:
                sentence += sub.text
            for text in in_sentence(sub):
                sentence += text
            sentence += sub.tail
            yield sentence
        else:
            for sentence in extract_sentence(sub):
                yield sentence


def extract_text(element):
    for sub in element:
        if sub.text and sub.text != '\n' and sub.text != '':
            yield sub.text
        if sub.tail and sub.tail != '\n' and sub.tail != '':
            yield sub.tail
    for sub in element:
        for text in extract_text(sub):
            yield text


def load(path='./data'):
    dataset = []
    for directory in os.listdir(path):
        label = directory
        for file in os.listdir(f'{path}/{directory}'):
            text = load_file(f'{path}/{directory}/{file}')
            dataset.append([text, label, file])
    return dataset

def load_c_xml(path='./C-XML/VARIABLE'):
    dataset_document = []
    dataset_sentence = []
    domain = ['OC', 'OY', 'OW', 'PB', 'PM', 'PN']
    for directory in domain:
        label = directory
        for file in os.listdir(f'{path}/{directory}/{directory}'):
            lines = load_file(f'{path}/{directory}/{directory}/{file}')
            root = ET.fromstringlist(lines)
            texts = ''
            for text in extract_text(root):
                texts += text
            texts = texts.replace('\n', '')
            dataset_document.append([texts, label, file])
            for sentence in extract_sentence(root):
                sentence = sentence.replace('\n', '')
                dataset_sentence.append([sentence, label, file])
    return dataset_document, dataset_sentence

def dump(dataset, file_name):
    df = pd.DataFrame(dataset, columns=['text', 'label', 'file'])
    # df = df.reindex(np.random.permutation(df.index)).reset_index(drop=True)
    df.to_csv(file_name, sep='\t', index=False)

def main():
    dataset_document, dataset_sentence = load_c_xml()
    dump(dataset_document, file_name='document.tsv')
    dump(dataset_sentence, file_name='sentence.tsv')


if __name__ == '__main__':
    main()