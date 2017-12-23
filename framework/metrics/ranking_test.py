import unittest
import ranking

class TestRankingMetrics(unittest.TestCase):
    """ Unit test class to test the WordEmbedding class
    """

    def test_ndcg(self):
        """
        Test DCG and NDCG metrics
        @return:
        """
        qids =  [1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5]
        rel =   [0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3]
        model = [.1,.5,.3,.4,.1,.2,.3,.4,.1,.2,.3,.4,.1,.2,.3,.4,.1,.2,.3,.4]
        [ndcg, qndcg] = ranking.ndcg_from_scores(qids, model, rel, 4)
        print(ndcg)
        print(qndcg)

    def test_ndcg_shuffled(self):
        qids =  [1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5]
        rel =   [0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3]
        model = [.1,.5,.3,.4,.1,.2,.3,.4,.1,.2,.3,.4,.1,.2,.3,.4,.1,.2,.3,.4]

        [ndcg, qndcg] = ranking.ndcg_from_scores(qids, model, rel, 4)
        print(ndcg)
        print(qndcg)

if __name__ == '__main__':
    unittest.main()
