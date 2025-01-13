import sys
from collections import defaultdict
import itertools 

# Function to load data from training file
def load_data(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.strip():
                data.append(tuple(map(int, line.strip().split())))
    return data

# Function to calculate conditional probability tables (CPTs)
def compute_cpts(data):
    cpt = defaultdict(dict)
    # Total counts
    total = len(data)
    
    # P(B)
    b_counts = {0: 0, 1: 0}
    for row in data:
        b_counts[row[0]] += 1
    cpt['B'][1] = b_counts[1] / total
    cpt['B'][0] = b_counts[0] / total
    
    # P(G | B)
    gb_counts = defaultdict(lambda: {0: 0, 1: 0})
    for row in data:
        gb_counts[row[0]][row[1]] += 1
    for b_val in [0, 1]:
        total_b = sum(gb_counts[b_val].values())
        if total_b > 0:
            cpt['G|B'][b_val] = {g_val: gb_counts[b_val][g_val] / total_b for g_val in [0, 1]}
    
    # P(C)
    c_counts = {0: 0, 1: 0}
    for row in data:
        c_counts[row[2]] += 1
    cpt['C'][1] = c_counts[1] / total
    cpt['C'][0] = c_counts[0] / total
    
    # P(F | G, C)
    fgc_counts = defaultdict(lambda: {0: {0: 0, 1: 0}, 1: {0: 0, 1: 0}})
    for row in data:
        fgc_counts[row[1]][row[2]][row[3]] += 1
    for g_val, c_val in itertools.product([0, 1], repeat=2):
        total_gc = sum(fgc_counts[g_val][c_val].values())
        if total_gc > 0:
            cpt['F|G,C'][(g_val, c_val)] = {f_val: fgc_counts[g_val][c_val][f_val] / total_gc for f_val in [0, 1]}
    
    return cpt

# Function to compute Joint Probability Distribution (JPD)
def compute_jpd(cpt):
    jpd = {}
    for b, g, c, f in itertools.product([0, 1], repeat=4):
        prob = (cpt['B'][b] * cpt['G|B'][b][g] * cpt['C'][c] * cpt['F|G,C'][(g, c)][f])
        jpd[(b, g, c, f)] = prob
    return jpd

# Function to perform inference by enumeration
def inference(query_vars, evidence_vars, jpd):
    hidden_vars = {'B', 'G', 'C', 'F'}.difference(query_vars.keys()).difference(evidence_vars.keys())
    
    def get_prob(assignments):
        return jpd[tuple(assignments[var] for var in ['B', 'G', 'C', 'F'])]

    # Calculate numerator P(Query, Evidence)
    numerator = 0
    for values in itertools.product([0, 1], repeat=len(hidden_vars)):
        assignment = {**query_vars, **evidence_vars, **dict(zip(hidden_vars, values))}
        numerator += get_prob(assignment)
    
    # Calculate denominator P(Evidence)
    denominator = 0
    for values in itertools.product([0, 1], repeat=len(hidden_vars.union(query_vars))):
        assignment = {**evidence_vars, **dict(zip(hidden_vars.union(query_vars), values))}
        denominator += get_prob(assignment)
    
    return numerator / denominator

# Function to parse user query
def parse_query(query):
    parts = query.split(' given ')
    query_vars = {var[0].upper(): int(var[1] == 't') for var in parts[0].strip().split()}
    evidence_vars = {}
    if len(parts) > 1:
        evidence_vars = {var[0].upper(): int(var[1] == 't') for var in parts[1].strip().split()}
    return query_vars, evidence_vars

# Interactive query interface
def interactive_query_prompt(jpd):
    while True:
        query = input("Query: ")
        if query.strip().lower() == "none":
            break
        try:
            query_vars, evidence_vars = parse_query(query)
            result = inference(query_vars, evidence_vars, jpd)
            print(f"Probability: {result:.4f}")
        except Exception as e:
            print(f"Error: {e}")

# Main program
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python bnet.py <training_data_file>")
        sys.exit(1)
    
    training_file = sys.argv[1]
    try:
        data = load_data(training_file)
        cpts = compute_cpts(data)
        jpd = compute_jpd(cpts)
        interactive_query_prompt(jpd)
    except Exception as e:
        print(f"An error occurred: {e}")
