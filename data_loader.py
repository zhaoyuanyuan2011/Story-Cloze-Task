import csv
import pandas as pd

def fetch_data(path):
    data = []
    with open(path) as csv_file:
        csv_file.readline()
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            sentence = ' '.join(row[1:7])
            data.append((sentence, int(row[7])-1))
    return data

def fetch_data2(path, start=1):
    data = []
    with open(path) as csv_file:
        csv_file.readline()
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            sentence = row[start:7]
            data.append((sentence, int(row[7])-1))
    return data

def fetch_test(path):
    data = []
    with open(path) as csv_file:
        csv_file.readline()
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            sentence = row[4:7]
            data.append(sentence)
    return data

def get_id(path):
    ids = []
    with open(path) as csv_file:
        csv_file.readline()
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            ids.append(row[0])
    return ids