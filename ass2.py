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
            self.postings[term].append((doc_id, tf))
            doc_length += (1 + math.log10(tf)) ** 2

        self.doc_lengths[doc_id] = math.sqrt(doc_length)

    def tokenize(self, text):
        # Basic tokenization - split on whitespace and convert to lowercase
        # You may want to implement more sophisticated tokenization
        return text.lower().split()

    def search(self, query):
        query_terms = self.tokenize(query)
        query_weights = {}
        for term in query_terms:
            if term in self.dictionary:
                df = self.dictionary[term]
                idf = math.log10(self.N / df)
                query_weights[term] = (1 + math.log10(1)) * idf

        scores = defaultdict(float)
        for term, weight in query_weights.items():
            for doc_id, tf in self.postings[term]:
                tfidf = (1 + math.log10(tf)) * weight
                scores[doc_id] += tfidf

        for doc_id in scores:
            scores[doc_id] /= self.doc_lengths[doc_id]

        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:10]

# Example usage
vsm = VSM()

# Load corpus from a directory
corpus_directory = "Corpus"
vsm.load_corpus(corpus_directory)

# Search query
query = """The company also makes the market-leading Kindle e-book readers. Its promotion of these devices has led to dramatic growth in e-book publishing and turned Amazon.com into a major disruptive force in the book-publishing market. In 1994 Jeff Bezos, a former Wall Street hedge fund executive, incorporated Amazon.com, choosing the name primarily because it began with the first letter of the alphabet and because of its association with the vast South American river. On the basis of research he had conducted, Bezos concluded that books would be the most logical product initially to sell online. Amazon.com was not the first company to do so; Computer Literacy, a Silicon Valley bookstore, began selling books from its inventory to its technically astute customers in 1991. However, the promise of Amazon.com was to deliver any book to any reader anywhere. While Amazon.com famously started as a bookseller, Bezos contended from its start that the site was not merely a retailer of consumer products. He argued that Amazon.cm 
was a technology company whose business was simplifying online transactions for consumers."""
results = vsm.search(query)

print("Search Results:")
for doc_id, score in results:
    print(f"Document {vsm.doc_ids[doc_id]}: {score}")