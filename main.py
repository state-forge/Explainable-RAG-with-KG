from retriever import ret
from nlp_processor import nlp_graph_builder
from answer_generator import ans_gen

def main():
    query, results = ret()
    if query is None or results is None:
        return
    else:
        ans_gen(query, results)
        nlp_graph_builder(query, results)
        return
if __name__ == "__main__":
    main()