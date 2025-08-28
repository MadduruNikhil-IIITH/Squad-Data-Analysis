import json
import sys
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from nltk.tokenize import sent_tokenize
import spacy

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")

# Function to get sentence boundaries using NLTK
def get_sentence_boundaries(context):
    sentences = sent_tokenize(context)
    boundaries = []
    start = 0
    for sent in sentences:
        start = context.find(sent, start)
        if start == -1:  # Fallback if sentence not found
            start = boundaries[-1][1] if boundaries else 0
        end = start + len(sent)
        boundaries.append((start, end, sent))
        start = end
    return boundaries

# Function to extract POS and NER from text using SpaCy
def extract_pos_ner(texts):
    pos_results = {}
    ner_results = {}
    docs = list(nlp.pipe(texts))  # Batch process texts
    for text, doc in zip(texts, docs):
        pos_list = [token.pos_ for token in doc]  # SpaCy POS tags
        ner_list = [ent.label_ for ent in doc.ents]  # SpaCy NER tags
        pos_results[text] = pos_list
        ner_results[text] = ner_list
    return pos_results, ner_results

# Main analysis function
def analyze_squad(json_data):
    data = json_data['data']
    results = []
    total_sentences = 0
    aggregate_sentences = []  # To track cumulative sentences over contexts
    total_contexts = 0
    total_questions = 0
    total_definite_spans = 0
    total_plausible_spans = 0
    definite_span_distribution = defaultdict(int)  # sentence index -> definite span count
    plausible_span_distribution = defaultdict(int)  # sentence index -> plausible span count
    sentence_counts_per_context = []  # For plotting
    all_pos_tags = Counter()  # Aggregate POS tags from all answers
    all_ner_tags = Counter()  # Aggregate NER tags from all answers
    unique_answer_texts = set()  # Collect unique answer texts for batch processing

    # Collect unique answer texts
    for entry in data:
        for paragraph in entry['paragraphs']:
            questions = paragraph['qas']
            for qa in questions:
                if not qa['is_impossible'] and 'answers' in qa and qa['answers']:
                    for ans in qa['answers']:
                        unique_answer_texts.add(ans['text'])
                if qa['is_impossible'] and 'plausible_answers' in qa:
                    for pans in qa['plausible_answers']:
                        unique_answer_texts.add(pans['text'])

    # Batch process POS and NER for unique texts
    pos_results, ner_results = extract_pos_ner(list(unique_answer_texts))

    # Process contexts
    for entry in data:
        title = entry['title']
        for paragraph in entry['paragraphs']:
            context = paragraph['context']
            sentences = sent_tokenize(context)
            num_sentences = len(sentences)
            total_sentences += num_sentences
            aggregate_sentences.append(total_sentences)  # Cumulative sum
            sentence_counts_per_context.append(num_sentences)
            total_contexts += 1

            boundaries = get_sentence_boundaries(context)

            questions = paragraph['qas']
            num_questions = len(questions)
            total_questions += num_questions

            definite_spans = 0
            plausible_spans = 0
            context_definite_spans_per_sentence = defaultdict(list)  # sentence index -> list of definite answer texts
            context_plausible_spans_per_sentence = defaultdict(list)  # sentence index -> list of plausible answer texts
            for qa in questions:
                # Handle definite answers (is_impossible: false)
                if not qa['is_impossible'] and 'answers' in qa and qa['answers']:
                    for ans in qa['answers']:
                        start = ans['answer_start']
                        text = ans['text']
                        definite_spans += 1
                        matched = False
                        for idx, (s_start, s_end, _) in enumerate(boundaries):
                            if s_start <= start < s_end:
                                definite_span_distribution[idx + 1] += 1
                                context_definite_spans_per_sentence[idx + 1].append(text)
                                # Update POS and NER counts from precomputed results
                                all_pos_tags.update(pos_results.get(text, []))
                                all_ner_tags.update(ner_results.get(text, []))
                                matched = True
                                break
                        if not matched:
                            definite_span_distribution['unmatched'] += 1
                # Handle plausible answers (is_impossible: true)
                if qa['is_impossible'] and 'plausible_answers' in qa:
                    for pans in qa['plausible_answers']:
                        start = pans['answer_start']
                        text = pans['text']
                        plausible_spans += 1
                        matched = False
                        for idx, (s_start, s_end, _) in enumerate(boundaries):
                            if s_start <= start < s_end:
                                plausible_span_distribution[idx + 1] += 1
                                context_plausible_spans_per_sentence[idx + 1].append(text)
                                # Update POS and NER counts from precomputed results
                                all_pos_tags.update(pos_results.get(text, []))
                                all_ner_tags.update(ner_results.get(text, []))
                                matched = True
                                break
                        if not matched:
                            plausible_span_distribution['unmatched'] += 1

            total_definite_spans += definite_spans
            total_plausible_spans += plausible_spans
            results.append({
                'context': context[:100] + '...' if len(context) > 100 else context,
                'num_sentences': num_sentences,
                'num_questions': num_questions,
                'num_definite_spans': definite_spans,
                'num_plausible_spans': plausible_spans,
                'title': title,
                'definite_spans_per_sentence': context_definite_spans_per_sentence,
                'plausible_spans_per_sentence': context_plausible_spans_per_sentence
            })

    avg_sentences = total_sentences / total_contexts if total_contexts > 0 else 0

    # # Print aggregate sentences count over contexts
    # print("Aggregate Sentences Count Over Contexts:")
    # for i, cum_sentences in enumerate(aggregate_sentences, 1):
    #     print(f"After Context {i}: {cum_sentences} sentences")

    # Plot statistics
    plot_statistics(aggregate_sentences, sentence_counts_per_context, definite_span_distribution, plausible_span_distribution, all_pos_tags, all_ner_tags)

    # Prepare aggregated stats summary
    aggregated_summary = f"## Aggregated Stats\n\n"
    aggregated_summary += f"**Total Contexts**: {total_contexts}\n\n"
    aggregated_summary += f"**Total Questions**: {total_questions}\n\n"
    aggregated_summary += f"**Total Definite Answer Spans**: {total_definite_spans}\n\n"
    aggregated_summary += f"**Total Plausible Answer Spans**: {total_plausible_spans}\n\n"
    aggregated_summary += f"**Total Sentences**: {total_sentences}\n\n"
    aggregated_summary += f"**Average Sentences per Context**: {avg_sentences:.2f}\n\n"
    
    aggregated_summary += "### Answer Span Distribution by Sentence Index\n\n"
    aggregated_summary += "#### Definite Answer Spans\n"
    # Sort by sentence number (numeric) and span count (descending)
    sorted_definite = sorted(
        definite_span_distribution.items(),
        key=lambda x: (x[0] if isinstance(x[0], str) else int(x[0]), -x[1])  # Numeric first, then count
    )
    for sent_idx, count in sorted_definite:
        aggregated_summary += f"- Sentence {sent_idx}: {count} spans\n"
    aggregated_summary += "\n#### Plausible Answer Spans\n"
    # Sort by sentence number (numeric) and span count (descending)
    sorted_plausible = sorted(
        plausible_span_distribution.items(),
        key=lambda x: (x[0] if isinstance(x[0], str) else int(x[0]), -x[1])  # Numeric first, then count
    )
    for sent_idx, count in sorted_plausible:
        aggregated_summary += f"- Sentence {sent_idx}: {count} spans\n"

    aggregated_summary += "\n### Most Common POS Tags in Answers\n"
    for pos, count in all_pos_tags.most_common(5):  # Top 5
        aggregated_summary += f"- {pos}: {count}\n"

    aggregated_summary += "\n### Most Common NER Tags in Answers\n"
    for ner, count in all_ner_tags.most_common(5):  # Top 5
        aggregated_summary += f"- {ner}: {count}\n"

    # Prepare per context stats summary
    per_context_summary = "\n## Per Context Stats\n\n"
    for i, res in enumerate(results, 1):
        per_context_summary += f"#### Context {i} (Title: {res['title']})\n"
        per_context_summary += f"- **Sentences**: {res['num_sentences']}\n"
        per_context_summary += f"- **Questions**: {res['num_questions']}\n"
        per_context_summary += f"- **Definite Answer Spans**: {res['num_definite_spans']}\n"
        per_context_summary += f"- **Plausible Answer Spans**: {res['num_plausible_spans']}\n"
        per_context_summary += f"- **Preview**: {res['context']}\n\n"
        per_context_summary += "##### Definite Answer Spans Per Sentence\n"
        for sent_idx in sorted(res['definite_spans_per_sentence'].keys(), key=int):
            spans = res['definite_spans_per_sentence'][sent_idx]
            per_context_summary += f"- Sentence {sent_idx}: {len(spans)} spans - {', '.join(spans)}\n"
        per_context_summary += "\n##### Plausible Answer Spans Per Sentence\n"
        for sent_idx in sorted(res['plausible_spans_per_sentence'].keys(), key=int):
            spans = res['plausible_spans_per_sentence'][sent_idx]
            per_context_summary += f"- Sentence {sent_idx}: {len(spans)} spans - {', '.join(spans)}\n\n"

    return aggregated_summary, per_context_summary

# Function to plot statistics
def plot_statistics(aggregate_sentences, sentence_counts_per_context, definite_span_distribution, plausible_span_distribution, all_pos_tags, all_ner_tags):
    # Plot 1: Aggregate sentences over contexts
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(aggregate_sentences) + 1), aggregate_sentences, marker='o', linestyle='-', color='b')
    plt.title('Aggregate Sentences Count Over Contexts')
    plt.xlabel('Context Number')
    plt.ylabel('Cumulative Sentences')
    plt.grid(True)
    plt.savefig('plots/aggregate_sentences.png')
    plt.close()

    # Plot 2: Sentences per context
    plt.figure(figsize=(10, 5))
    plt.bar(range(1, len(sentence_counts_per_context) + 1), sentence_counts_per_context, color='g')
    plt.title('Sentences Per Context')
    plt.xlabel('Context Number')
    plt.ylabel('Number of Sentences')
    plt.grid(True)
    plt.savefig('plots/sentences_per_context.png')
    plt.close()

    # Plot 3: Definite and Plausible Span Distributions
    sent_indices = sorted(set(list(definite_span_distribution.keys()) + list(plausible_span_distribution.keys())), key=lambda x: (x if isinstance(x, str) else int(x)))
    definite_counts = [definite_span_distribution[idx] for idx in sent_indices]
    plausible_counts = [plausible_span_distribution[idx] for idx in sent_indices]

    plt.figure(figsize=(12, 6))
    bar_width = 0.35
    x = range(len(sent_indices))
    plt.bar(x, definite_counts, width=bar_width, label='Definite Spans', color='blue')
    plt.bar([p + bar_width for p in x], plausible_counts, width=bar_width, label='Plausible Spans', color='red')
    plt.title('Definite vs Plausible Answer Spans by Sentence Index')
    plt.xlabel('Sentence Index')
    plt.ylabel('Number of Spans')
    plt.xticks([p + bar_width / 2 for p in x], sent_indices)
    plt.legend()
    plt.grid(True)
    plt.savefig('plots/span_distribution.png')
    plt.close()

    # Plot 4: POS Tag Distribution
    top_pos_tags = all_pos_tags.most_common(5)  # Top 5 POS tags
    pos_labels = [pos for pos, _ in top_pos_tags]
    pos_counts = [count for _, count in top_pos_tags]
    plt.figure(figsize=(10, 5))
    plt.bar(pos_labels, pos_counts, color='purple')
    plt.title('Top 5 POS Tags in Answer Spans')
    plt.xlabel('POS Tag')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig('plots/pos_distribution.png')
    plt.close()

    # Plot 5: NER Tag Distribution (Improved)
    ner_labels = list(all_ner_tags.keys())
    ner_counts = [all_ner_tags[ner] for ner in ner_labels]
    plt.figure(figsize=(12, 5))
    plt.bar(ner_labels, ner_counts, color='orange')
    plt.title('NER Tags in Answer Spans')
    plt.xlabel('NER Tag')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45, ha='right', fontsize=10)  # Rotate labels to prevent overlap
    plt.grid(True)
    plt.tight_layout()  # Adjust layout to prevent clipping
    plt.savefig('plots/ner_distribution.png')
    plt.close()

    print("Plots saved: aggregate_sentences.png, sentences_per_context.png, span_distribution.png, pos_distribution.png, ner_distribution.png")

# Load JSON from file and run analysis
def main(filename):
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        aggregated_summary, per_context_summary = analyze_squad(json_data)
        
        # Output to separate Markdown files
        with open('aggregated_stats.md', 'w', encoding='utf-8') as doc:
            doc.write(aggregated_summary)
        
        with open('per_context_stats.md', 'w', encoding='utf-8') as doc:
            doc.write(per_context_summary)
        
        print("Aggregated stats documented in aggregated_stats.md")
        print("Per context stats documented in per_context_stats.md")
    
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
    main("data.json")