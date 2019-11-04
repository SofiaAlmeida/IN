# -*- coding: utf-8 -*-

# importing 
import csv
import sys

# csv file name
filename = sys.argv[1]

# initializing the titles and rows list 
fields = [] 
rows = [] 
  
# reading csv file 
with open(filename, 'r') as csvfile: 
    # creating a csv reader object 
    csvreader = csv.reader(csvfile) 
      
    # extracting field names through first row 
    fields = csvreader.next() 
  
    # extracting each data row one by one 
    for row in csvreader: 
        rows.append(row) 
  
    # get total number of rows 
    # print("Total no. of rows: %d"%(csvreader.line_num)) 
  
# printing the field names 
print('\\begin{table}[H]\n\centering\n\caption{- Matriz de confusión}\n\label{tab:CM}')

# printing the structure of the table

print('\\begin{tabular}{lrrr}')
print('\\toprule')
print(' & '.join(field for field in fields) + '\\\\ \midrule')

#  printing rows 
for row in rows: 
    # parsing each column of a row
    s = row[0]
    for col in row[1:]:
        s += ' & ' + col
    s += "\\\\"
    print s

# printing end of table
print('\\bottomrule\n\end{tabular}\n\end{table}')

