from ray.tune.search import Searcher


class SearchAgent(Searcher):

    def __init__(self, optimizer: object, metric: str, mode: str, **kwargs) -> None:
        super(SearchAgent, self).__init__(metric=metric, mode=mode, **kwargs)

        self.optimizer = optimizer
        self.configurations = {}

    # queries the algorithm to retrieve the next set of parameters
    def suggest(self, trial_id: int) -> dict:
        configuration = self.optimizer.query()
        self.configurations[trial_id] = configuration
        
        return self.configurations[trial_id]

    # notification for the completion of trial
    def on_trial_complete(self, trial_id: int, result: dict, **kwargs) -> None:
        configuration = self.configurations[trial_id]
        self.optimizer.update(configuration, result)
    