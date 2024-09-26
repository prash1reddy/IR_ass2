import math
import os
from collections import defaultdict


class VSM:
    def __init__(self):
        self.dictionary = {}
        self.postings = defaultdict(list)
        self.doc_lengths = {}
        self.N = 0  # Total number of documents
        self.doc_ids = {}  # Mapping of filenames to doc_ids

    def load_corpus(self, directory):
        for filename in os.listdir(directory):
            if filename.endswith(".txt"):
                filepath = os.path.join(directory, filename)
                with open(filepath, 'r', encoding='utf-8') as file:
                    content = file.read()
                self.N += 1
                self.doc_ids[self.N] = filename
                self.index_document(self.N, content)

    def index_document(self, doc_id, content):
        terms = self.tokenize(content)
        term_freq = defaultdict(int)
        for term in terms:
            term_freq[term] += 1

        doc_length = 0
        for term, tf in term_freq.items():
            if term not in self.dictionary:
                self.dictionary[term] = 1
            else:
                self.dictionary[term] += 1
            log_tf = 1 + math.log10(tf) if tf > 0 else 0
            self.postings[term].append((doc_id, log_tf))
            doc_length += log_tf ** 2

        self.doc_lengths[doc_id] = math.sqrt(doc_length)

    def tokenize(self, text):
        # Basic tokenization - split on whitespace and convert to lowercase
        return text.lower().split()

    def search(self, query):
        query_terms = self.tokenize(query)
        query_weights = defaultdict(float)
        query_length = 0

        for term in query_terms:
            tf = 1 + math.log10(query_terms.count(term))
            if term in self.dictionary:
                df = self.dictionary[term]
                idf = math.log10(self.N / df)
                query_weights[term] = tf * idf
                query_length += query_weights[term] ** 2

        query_length = math.sqrt(query_length)

        scores = defaultdict(float)
        for term, weight in query_weights.items():
            normalized_query_weight = weight / query_length
            for doc_id, log_tf in self.postings[term]:
                tfidf = log_tf * normalized_query_weight
                scores[doc_id] += tfidf

        # Normalize document scores
        for doc_id in scores:
            scores[doc_id] /= self.doc_lengths[doc_id]

        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:10]

# Example usage
vsm = VSM()

# Load corpus from a directory
corpus_directory = "Corpus"
vsm.load_corpus(corpus_directory)


query1 = """Developing your Zomato business account and profile is a great way to boost your 
restaurants online reputation"""
query2 = """Warwickshire, came from an ancient family and was the heiress to 
some land"""

result1 = vsm.search(query1)
result2 = vsm.search(query2)

print("Search Results for test-query 1:")
for doc_id, score in result1:
    print(f"Document {vsm.doc_ids[doc_id]}: {score}")

print("---------------------------------------\n\n")
print("Search Results for test-query 2:")
for doc_id, score in result2:
    print(f"Document {vsm.doc_ids[doc_id]}: {score}")

print("\n")
# Search query
query = input("Enter the query you want to test: ")
results = vsm.search(query)

print("Search Results:")
for doc_id, score in results:
    print(f"Document {vsm.doc_ids[doc_id]}: {score}")