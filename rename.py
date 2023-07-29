import bibtexparser

bibtex_file_path = '/home/wouter/Tools/Zotero/bibtex/library.bib'
output_txt_file = './renamed.txt'

def extract_citation_keys(bibtex_file, output_file):
    with open(bibtex_file, 'r') as bibfile:
        bib_database = bibtexparser.load(bibfile)

    citation_keys = [entry['ID'] for entry in bib_database.entries]

    with open(output_file, 'w') as f:
        f.write("\n".join(citation_keys).join('.txt')

if __name__ == "__main__":
    input_bibtex_file = bibtex_file_path

    extract_citation_keys(input_bibtex_file, output_txt_file)
    print("Citation keys extracted and saved to 'renamed.txt'.")
