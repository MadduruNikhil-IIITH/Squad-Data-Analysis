import json
import csv
import os
from collections import defaultdict, Counter
import sys
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import pos_tag, ne_chunk
from nltk.tree import Tree

# Download necessary NLTK resources
import nltk
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('maxent_ne_chunker', quiet=True)
nltk.download('words', quiet=True)

# Function to get sentence boundaries
def get_sentence_boundaries(context):
    """Split context into sentences and compute character boundaries."""
    sentences = sent_tokenize(context)
    boundaries = []
    start = 0
    for sent in sentences:
        start = context.find(sent, start)
        if start == -1:
            start = boundaries[-1][1] if boundaries else 0
        end = start + len(sent)
        boundaries.append((start, end, sent))
        start = end
    return boundaries

# Function to extract POS and NER from text
def extract_pos_ner(text):
    """Extract POS and NER tags from text using NLTK."""
    tokens = word_tokenize(text)
    pos_tags = pos_tag(tokens)
    ner_tree = ne_chunk(pos_tags)
    
    pos_list = [tag for word, tag in pos_tags]
    ner_list = []
    for subtree in ner_tree:
        if isinstance(subtree, Tree):
            ner_list.append(subtree.label())
    return pos_list, ner_list

# Main function to process JSON and generate annotated dataset
def annotate_squad_sentences(json_data, filename):
    """Annotate sentences for salience with deduplicated answer spans."""
    data = json_data['data']
    context_mapping = []  # For contexts.json
    all_annotated = []  # For all_annotated_sentences.csv
    all_pos = set()  # Unique POS tags
    all_ner = set()  # Unique NER tags
    unique_answer_texts = set()  # Unique answer texts for batch processing

    # Collect unique answer texts
    for entry in data:
        for paragraph in entry['paragraphs']:
            questions = paragraph['qas']
            for qa in questions:
                answers = qa['answers'] if not qa['is_impossible'] and 'answers' in qa else (qa['plausible_answers'] if 'plausible_answers' in qa else [])
                for ans in answers:
                    unique_answer_texts.add(ans['text'])
            break  # Process only the first paragraph per entry for simplicity
        break

    # Batch process POS and NER for unique texts
    pos_results = {}
    ner_results = {}
    print(f"Processing {len(unique_answer_texts)} unique answer texts for POS and NER...")
    for text in unique_answer_texts:
        pos, ner = extract_pos_ner(text)
        pos_results[text] = pos
        ner_results[text] = ner
        all_pos.update(pos)
        all_ner.update(ner)

    # # Assign IDs to POS and NER
    pos_mapping = {tag: i+1 for i, tag in enumerate(sorted(all_pos))}
    ner_mapping = {tag: i+1 for i, tag in enumerate(sorted(all_ner))}

    print(f"Unique POS tags: {len(pos_mapping)}, Unique NER tags: {len(ner_mapping)}")

    # Create dynamic folder name based on input filename
    base_filename = os.path.splitext(os.path.basename(filename))[0]
    output_folder = f'processed_salience_data/per_context_csvs_{base_filename}'
    output_id_mappings_folder = f'id_mappings/{base_filename}'
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(output_id_mappings_folder, exist_ok=True)

    # Output POS and NER mappings to CSV
    with open(f'{output_id_mappings_folder}/pos_tags.csv', 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['PosID', 'PosTag'])
        for tag, id in sorted(pos_mapping.items(), key=lambda x: x[1]):
            writer.writerow([id, tag])

    with open(f'{output_id_mappings_folder}/ner_tags.csv', 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['NERID', 'NERTag'])
        for tag, id in sorted(ner_mapping.items(), key=lambda x: x[1]):
            writer.writerow([id, tag])

    # Process contexts
    context_id = 1
    for entry in data:
        title = entry['title']
        for paragraph in entry['paragraphs']:
            context = paragraph['context']
            context_mapping.append({
                'contextID': context_id,
                'title': title,
                'context': context
            })

            print(f"Processing Context ID {context_id}: {title}")

            # Get sentence boundaries
            boundaries = get_sentence_boundaries(context)

            print(f"Total sentences in context: {len(boundaries)}")

            # Collect annotations for this context with deduplication
            annotations = []
            seen_spans = defaultdict(set)  # sentence_num -> set of (answer, span_in_context)
            questions = paragraph['qas']
            for qa in questions:
                print(qa)
                answers = qa['answers'] if not qa['is_impossible'] and 'answers' in qa else (qa['plausible_answers'] if 'plausible_answers' in qa else [])
                for ans in answers:
                    print(ans)
                    start = ans['answer_start']
                    text = ans['text']
                    end = start + len(text)

                    # Find sentence
                    sent_num = None
                    sent_start = None
                    sent_text = None
                    for idx, (s_start, s_end, s_text) in enumerate(boundaries, 1):
                        if s_start <= start < s_end:
                            sent_num = idx
                            sent_start = s_start
                            sent_text = s_text
                            break

                    print(f"Answer: '{text}' at {start}-{end} in sentence {sent_num}")
                    
                    if sent_num:
                        # Check for duplicates
                        span_key = (text, f"{start}-{end}")
                        if span_key not in seen_spans[sent_num]:
                            seen_spans[sent_num].add(span_key)
                            # Span in sentence
                            span_in_sent_start = start - sent_start
                            span_in_sent_end = end - sent_start

                            # POS and NER IDs
                            pos_ids = ','.join(str(pos_mapping.get(p, 0)) for p in pos_results.get(text, []))
                            ner_ids = ','.join(str(ner_mapping.get(n, 0)) for n in ner_results.get(text, []))

                            # Annotation
                            ann = {
                                'contextID': context_id,
                                'sentence': sent_text,
                                'sentenceNumInContext': sent_num,
                                'answer': text,
                                'answerSpanInSentence': f"{span_in_sent_start}-{span_in_sent_end}",
                                'answerSpanInContext': f"{start}-{end}",
                                'PosID': pos_ids,
                                'NERID': ner_ids
                            }
                            print(ann)  # Debug print
                            annotations.append(ann)
                            all_annotated.append(ann)

            # Write per context CSV
            with open(f'{output_folder}/context_{context_id}.csv', 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=['contextID', 'sentence', 'sentenceNumInContext', 'answer', 'answerSpanInSentence', 'answerSpanInContext', 'PosID', 'NERID'])
                writer.writeheader()
                writer.writerows(annotations)

            context_id += 1
            break  # Process only the first paragraph per entry for simplicity

    # Output context mapping to JSON
    with open(f'{output_id_mappings_folder}/contexts.json', 'w', encoding='utf-8') as f:
        json.dump(context_mapping, f, indent=4)

    # Output across contexts CSV
    with open(f'processed_salience_data/all_annotated_sentences_{base_filename}.csv', 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['contextID', 'sentence', 'sentenceNumInContext', 'answer', 'answerSpanInSentence', 'answerSpanInContext', 'PosID', 'NERID'])
        writer.writeheader()
        writer.writerows(all_annotated)

    print(f"Generated files: pos_tags.csv, ner_tags.csv, contexts.json, all_annotated_sentences.csv, and per-context CSVs in {output_folder} folder (context_1.csv to context_{context_id-1}.csv)")

# Load JSON and run annotation
def main(filename):
    """Load JSON file and generate annotated dataset."""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        annotate_squad_sentences(json_data, filename)
    
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in '{filename}'.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python squad_sentence_annotation_dedup.py <json_filename>")
        sys.exit(1)
    main(sys.argv[1])