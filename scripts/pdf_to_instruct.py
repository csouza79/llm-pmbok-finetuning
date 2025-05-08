import fitz  # PyMuPDF
import json
import os

def extract_text_from_pdf(file_path):
    doc = fitz.open(file_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    return full_text

def split_into_paragraphs(text, min_length=100):
    paragraphs = [p.strip() for p in text.split('\n') if len(p.strip()) > min_length]
    return paragraphs

def generate_instruction(paragraph):
    first_words = paragraph.split(' ')[:6]
    topic = ' '.join(first_words).rstrip('.:')
    return f"Explique sobre: {topic}..."

def create_instruct_dataset(paragraphs, max_samples=100):
    dataset = []
    for para in paragraphs[:max_samples]:
        item = {
            "instruction": generate_instruction(para),
            "input": "",
            "output": para
        }
        dataset.append(item)
    return dataset

def main():
    input_pdf = "../PMBOK.pdf"  # ajuste se necessário
    output_path = "../data/pmbok_instruct_dataset.jsonl"

    text = extract_text_from_pdf(input_pdf)
    paragraphs = split_into_paragraphs(text)
    dataset = create_instruct_dataset(paragraphs)

    os.makedirs("../data", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"✅ Dataset salvo em: {output_path}")

if __name__ == "__main__":
    main()
