#!/bin/bash

# One dir to keep papers that I have already processed
# and another dir to store newly added papers
existing_file="./data/ingested.txt"
output_dir="./data/new"
temp_dir="/home/wouter/Documents/LangChain_Projects/Nexis/data/temp"

counter=0

total=$(find /home/wouter/Documents/LangChain_Projects/Nexis/data/pdf -type f -name "*.pdf" | wc -l)

find /home/wouter/Documents/LangChain_Projects/Nexis/data/pdf -type f -name "*.pdf" | while read -r file
do
    base_name=$(basename "$file" .pdf)

    if grep -Fxq "$base_name.txt" "$existing_file"; then
	echo -ne "Text file for $file already exists, skipping.\r"
    else 
	pdftotext -enc UTF-8 "$file" "$output_dir/$base_name.txt"
	
    fi
    counter=$((counter + 1))
    echo -ne "Processed $counter out of $total PDFs.\r"
    
done
