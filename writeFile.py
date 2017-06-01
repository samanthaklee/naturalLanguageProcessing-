import main
import csv
import io
import sys
import string

"""
with open("~/Desktop/naturalLanguageProcessing/factCheck.csv", 'wb') as outcsv:   
    #configure writer to write standard csv file
    writer = csv.writer(outcsv, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
    writer.writerow(['Word', 'Token'])
    for item in list:
        #Write item to outcsv
        writer.writerow([item[0], item[1]])
"""

with open('output.csv', 'w') as csvFile:
	fieldnames  = ['Words', 'Token']
	writer = csv.DictWriter(csvFile, fieldnames = fieldnames)
	writer.writeheader()

	i = 0
	for i in range(len(word_final)):
		writer.writerow({'Words': word_final[i][0], 'Token': word_final[i][1]})