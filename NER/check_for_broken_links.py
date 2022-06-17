# This script checks a list of links for broken links
import csv 
from bs4 import BeautifulSoup, SoupStrainer
import requests
import pandas as pd 

# convert the csv to a list
with open('ER_pages.csv', 'r') as f:
    pages = [row[1] for row in csv.reader(f)]

# This is to remove the first two useless rows
n = 2
pages = pages[n:]
# pages = pages[2400:2420]
# Putting into the correct format
pages = ["https://www.gov.uk" + s for s in pages]

# Checking the links
response_codes = []

for page in range(len(pages)):
    length = len(pages)
    print(page, "/", length)
    url = requests.get(pages[page])
    response_code = str(url.status_code)
    response_codes.append(response_code)


# removing the bad links and putting into new .csv
dict = {"url": pages, "response_codes": response_codes}
df = pd.DataFrame(dict)
# Get names of indexes for which column `response_codes` has the correct value
indexNames = df[df['response_codes'] == "200"].index
# print(indexNames)
# Delete these row indexes from dataFrame
df = df.loc[indexNames]
df.to_csv("ER_pages_cleaned.csv")