from langchain_community.utilities import DuckDuckGoSearchAPIWrapper


class WebSearch:

    def __init__(self, max_results):
        self.wrapper = DuckDuckGoSearchAPIWrapper(max_results=max_results)

    def search(self, query):
        return self.wrapper.results(query,max_results=self.wrapper.max_results)