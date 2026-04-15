# This is the prototype - not the final!

from pypdf import PdfReader
import sys
import re


def convert_pdf_to_txt(pdf_path, output_txt_path):
    text = ""

    if pdf_path.endswith(".pdf"):
        with open(pdf_path, 'rb') as file_obj:
            reader = PdfReader(file_obj)
            for page in reader.pages:
                text += page.extract_text() or ""
    else:
        with open(pdf_path, 'r', encoding='utf-8') as file:
            text = file.read()

    lower = text.lower()

    sentences = split_into_sentences(lower)

    with open(output_txt_path, 'w', encoding='utf-8') as output_file:
        for sentence in sentences:
            output_file.write(sentence + "\n")

    print(f"Text successfully saved to '{output_txt_path}'")

def split_into_sentences(text):
    # Common abbreviations you don’t want to split on
    abbreviations = [
        "mr.", "mrs.", "ms.", "dr.", "sen.", "rep.", "u.s.", "u.k.",
        "jan.", "feb.", "mar.", "apr.", "jun.", "jul.", "aug.",
        "sep.", "sept.", "oct.", "nov.", "dec."
    ]
    
    # Protect abbreviations by temporarily removing periods
    for abbr in abbreviations:
        text = text.replace(abbr, abbr.replace(".", "<PERIOD>"))

    # Replace newlines with space (except double newline which may be title breaks)
    text = re.sub(r'\n+', ' ', text)

    # Split on sentence-ending punctuation
    sentences = re.split(r'(?<=[.!?])\s+', text)

    # Restore periods in abbreviations
    sentences = [s.replace("<PERIOD>", ".") for s in sentences]

    # Strip whitespace and remove empty
    sentences = [s.strip() for s in sentences if s.strip()]

    return sentences

def build_issues(issues_file)->dict:

    issues = dict()
    with open(issues_file, 'r') as file:
        for line in file:
            if line != '\n':
                text = line.lower()
                title = text.split(":")[0]
                term_line = text.split(":")[1]
                terms = [item.strip() for item in term_line.split(',')]
                issues[title] = terms

    return issues

def find_sentences_with_term(filename, term):
    found_sentences = []
    # Use a with statement to ensure the file is closed automatically
    with open(filename, 'r') as f:
        for line in f:
            # Check if the search string is in the current line
            if term in line:
                # Append the line, stripping leading/trailing whitespace
                found_sentences.append(line.strip())
    return found_sentences

def print_relevant_sentences(issues, sentences1, sentences2, candidate1_name="Candidate 1", candidate2_name="Candidate 2"):
    for issue in issues.keys():
        print(f"\n=== {issue.upper()} ===\n")

        # Candidate 1 sentences
        print(f"{candidate1_name}:\n")
        if issue in sentences1 and sentences1[issue]:
            for s in sentences1[issue]:
                print(f"- {s}")
        else:
            print("- No sentences found.")

        print("\n" + "-"*40 + "\n")

        # Candidate 2 sentences
        print(f"{candidate2_name}:\n")
        if issue in sentences2 and sentences2[issue]:
            for s in sentences2[issue]:
                print(f"- {s}")
        else:
            print("- No sentences found.")

        print("\n" + "="*60 + "\n")

def main():
    file1_name = "file1.txt"
    file2_name = "file2.txt"

    file1 = sys.argv[1]
    convert_pdf_to_txt(file1, file1_name)
    file2 = sys.argv[2]
    convert_pdf_to_txt(file2, file2_name)

    issues = build_issues("issues.txt")
    print(f"Loaded issues: {list(issues.keys())}")

    relevant_sentences_1 = dict()
    relevant_sentences_2 = dict()


    for issue, terms in issues.items():
        for term in terms:
            sentences1 = find_sentences_with_term(file1_name, term)
            sentences2 = find_sentences_with_term(file2_name, term)

            relevant_sentences_1.setdefault(issue, []).extend(sentences1)
            relevant_sentences_2.setdefault(issue, []).extend(sentences2)

    print_relevant_sentences(issues, relevant_sentences_1, relevant_sentences_2,
                         candidate1_name="Trump", candidate2_name="Kamala")
    
if __name__ == "__main__":
    main()