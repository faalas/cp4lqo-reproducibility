import pandas as pd

def calculate_error(first, last):
    # first = pd.read_csv(prevItr)
    # last = pd.read_csv(currentItr)

    initial_query_latency = dict(zip(first.iloc[:, 0], first.iloc[:, 1]))
    best_query_latency = dict(zip(last.iloc[:, 0], last.iloc[:, 1]))

    validation = list(initial_query_latency.keys())[:80]

    combine = {query: [initial_query_latency[query], best_query_latency[query]] for query in validation if query in best_query_latency}

    errors = {query: abs(combine[query][0] - combine[query][1]) for query in combine}

    quant90val = sorted(errors.values())[int(len(errors) * 0.9)]

    testSet = []
    for query in best_query_latency:
        if query not in validation and query != 'all':
            testSet.append(query)

    rangePrediction = {query: [max(0, best_query_latency[query] - quant90val), best_query_latency[query] + quant90val] for query in testSet}

    correct = 0
    wrong = 0
    for i in testSet:
        actual = initial_query_latency[i]
        if(actual >= rangePrediction[i][0] and actual <= rangePrediction[i][1]):
            correct += 1
        else:
            wrong += 1

    accuracy = 1 - ((len(testSet) - correct) / correct)

    return quant90val, accuracy, best_query_latency["all"]