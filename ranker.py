import json

class Ranker:
    def rank_candidates(self, population):
        """
        Input: List of dictionaries [{'sequence': ..., 'score': ..., 'mutations': ...}]
        Output: Top 10 sorted candidates.
        """
        # Sort by 'score' in descending order (highest log-likelihood is best)
        sorted_pop = sorted(population, key=lambda x: x['score'], reverse=True)
        return sorted_pop[:10]