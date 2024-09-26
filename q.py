import math
import os
from collections import defaultdict
import re

# Dictionary and postings list
dictionary = defaultdict(list)
doc_lengths = {}
doc_names = {}

# Function to tokenize text
def tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())

# Function to build the index using the specified corpus directory
def build_index(corpus_dir):
    doc_id = 0
    total_docs = 0
    
    for filename in os.listdir(corpus_dir):
        file_path = os.path.join(corpus_dir, filename)
        
        if os.path.isfile(file_path):
            total_docs += 1
            doc_id += 1
            doc_names[doc_id] = filename
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                tokens = tokenize(content)
                term_freqs = defaultdict(int)
                
                for term in tokens:
                    term_freqs[term] += 1

                length = 0
                for term, freq in term_freqs.items():
                    log_tf = 1 + math.log10(freq)
                    length += log_tf ** 2
                    dictionary[term].append((doc_id, freq))
                
                doc_lengths[doc_id] = math.sqrt(length)

    return total_docs

def process_query(query, total_docs):
    query_tokens = tokenize(query)
    query_freqs = defaultdict(int)

    for term in query_tokens:
        query_freqs[term] += 1

    scores = []
    for doc_id in range(1, total_docs + 1):
        score = 0
        for term, freq in query_freqs.items():
            if term in dictionary:
                doc_term_freq = next((tf for d_id, tf in dictionary[term] if d_id == doc_id), 0)
                log_tf = 1 + math.log10(doc_term_freq) if doc_term_freq > 0 else 0
                log_query_tf = 1 + math.log10(freq)
                score += log_tf * log_query_tf
        
        if doc_id in doc_lengths:
            score /= doc_lengths[doc_id]

        if score > 0:
            scores.append((doc_names[doc_id], score))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:10]

# Test the code with your corpus
# Test the code with your corpus
if __name__ == "__main__":
    corpus_dir = 'Corpus'  # your path
    
    # Build the index from the corpus
    total_docs = build_index(corpus_dir)
    
    # Test queries
    query1 = """The company also makes the market-leading Kindle e-book readers. Its promotion of these devices has led to dramatic growth in e-book publishing and turned Amazon.com into a major disruptive force in the book-publishing market. In 1994 Jeff Bezos, a former Wall Street hedge fund executive, incorporated Amazon.com, choosing the name primarily because it began with the first letter of the alphabet and because of its association with the vast South American river. On the basis of research he had conducted, Bezos concluded that books would be the most logical product initially to sell online. Amazon.com was not the first company to do so; Computer Literacy, a Silicon Valley bookstore, began selling books from its inventory to its technically astute customers in 1991. However, the promise of Amazon.com was to deliver any book to any reader anywhere. While Amazon.com famously started as a bookseller, Bezos contended from its start that the site was not merely a retailer of consumer products. He argued that Amazon.cm 
was a technology company whose business was simplifying online transactions for consumers."""
    query2 = "Warwickshire, came from an ancient family and was the heiress to some land"

    # Process and print results for Query 1
    print(f"Q1: '{query1}'")
    results1 = process_query(query1, total_docs)
    for idx, (doc_name, score) in enumerate(results1, start=1):
        print(f"{idx}. {doc_name} ({score:.10f})")  # Same format as Q2

    # Process and print results for Query 2
    print(f"\nQ2: {query2}")
    results2 = process_query(query2, total_docs)
    for idx, (doc_name, score) in enumerate(results2, start=1):
        print(f"{idx}. {doc_name} ({score:.10f})")

