import numpy as np
import torch
from unittest import TestCase, main

from mdn_metric.layers.distribution import DiracDistribution, GMMDistribution
from mdn_metric.layers.scorer import NegativeL2Scorer
from mdn_metric.metrics.nearest import NearestNeighboursMetrics, MAPR, GAPR, GroupedNearestNeighboursMetrics
from mdn_metric.torch import tmp_seed


class TestUtils(TestCase):
    def test_knn(self):
        for backend in ["faiss", "numpy", "torch"]:
            d = DiracDistribution(config={"dim": 2, "spherical": False})
            s = NegativeL2Scorer(d)
            metric = NearestNeighboursMetrics(d, s, config={"backend": backend})

            # Test unimodal.
            x = torch.tensor([
                [2, 0],
                [2.1, 0],
                [1.1, 0],
                [0, 1],
                [0, 0]
            ]).float().reshape((-1, 1, 2))

            indices = metric._multimodal_knn(x, 2)
            indices_gt = np.array([
                [0, 1],
                [1, 0],
                [2, 0],
                [3, 4],
                [4, 3]
            ]).reshape((-1, 1, 2))
            self.assertTrue(np.allclose(indices, indices_gt))

            # Test multimodal.
            d = GMMDistribution(config={"dim": 2, "components": 2})
            s = NegativeL2Scorer(d)
            metric = NearestNeighboursMetrics(d, s, config={"backend": backend})

            _, x = d.modes(d.join_parameters(
                log_probs=torch.randn(4, 2),
                means=torch.tensor([
                    [[0, 0], [0, 1]],
                    [[1, 0], [1, 1]],
                    [[-0.5, 0], [-0.5, 1]],
                    [[0.1, 0.1], [0.99, 0.99]]
                ]).float(),
                hidden_vars=torch.randn(4, 2, 1)
            ))  # (4, 2, 2).

            indices = metric._multimodal_knn(x, 1)
            indices_gt = np.array([
                [0, 0],
                [1, 1],
                [2, 2],
                [3, 3]
            ]).reshape((4, 2, 1))
            self.assertTrue(np.allclose(indices, indices_gt))

    def test_gather(self):
        x = torch.tensor([
            [1, 2],
            [3, 4],
            [5, 6]
        ])  # (3, 2).
        index = torch.tensor([
            [0, 2],
            [2, 1]
        ])  # (2, 2).
        result = NearestNeighboursMetrics._gather_broadcast(x[None], 1, index[..., None])  # (2, 2, 2).
        result_gt = [
            [[1, 2], [5, 6]],
            [[5, 6], [3, 4]]
        ]
        self.assertTrue(np.allclose(result, result_gt))

    def test_remove_duplicates(self):
        # K unique values are available.
        x = torch.tensor([
            [5, 3, 2, 5, 1, 1, 5],
            [5, 4, 3, 2, 1, 1, 1],
            [5, 4, 2, 2, 2, 2, 4],
        ])
        result = NearestNeighboursMetrics._remove_duplicates(x, 3)
        result_gt = [
            [5, 3, 2],
            [5, 4, 3],
            [5, 4, 2]
        ]
        self.assertTrue(np.allclose(result, result_gt))

        # The number of unique values is less than K.
        result = NearestNeighboursMetrics._remove_duplicates(x, 6)
        result_gt = [
            [5, 3, 2, 1, 1, 5],
            [5, 4, 3, 2, 1, 1],
            [5, 4, 2, 2, 2, 4]
        ]
        self.assertTrue(np.allclose(result, result_gt))

    def test_get_positives(self):
        d = DiracDistribution(config={"dim": 1, "spherical": False})
        s = NegativeL2Scorer(d)
        metric = NearestNeighboursMetrics(d, s)
        labels = torch.tensor([1, 0, 1, 2, 0])
        parameters = torch.tensor([
            [0],
            [0],
            [1],
            [3],
            [5]
        ]).float()
        scores, counts, same_mask = metric._get_positives(parameters, labels)
        scores_gt = torch.tensor([
            [0, -1],
            [0, -25],
            [0, -1],
            [0, -26],  # Second is dummy score with minimum value.
            [0, -25]
        ])
        counts_gt = torch.tensor([2, 2, 2, 1, 2])
        same_mask_gt = torch.tensor([
            [1, 0],
            [1, 0],
            [1, 0],
            [1, 0],
            [1, 0]
        ])
        self.assertTrue((scores == scores_gt).all())
        self.assertTrue((counts == counts_gt).all())
        self.assertTrue((same_mask == same_mask_gt).all())


class TestMAPR(TestCase):
    def test_simple(self):
        d = DiracDistribution(config={"dim": 1, "spherical": False})
        s = NegativeL2Scorer(d)

        m = NearestNeighboursMetrics(d, s, config={"metrics": ["mapr-ms"], "prefetch_factor": 1})
        labels = torch.tensor([1, 1, 0, 1, 2, 2, 0, 0, 1])  # (B).
        parameters = torch.arange(len(labels))[:, None] ** 1.01  # (B, 1).

        # N  R   Nearest      Same      P            MAP@R
        # 0  4   0, 1, 2, 3   1 1 0 1   1, 1, 3/4    11/16
        # 1  4   1, 0, 2, 3   1 1 0 1   1, 1, 3/4    11/16
        # 2  3   2, 1, 3      1 0 0     1            1/3
        # 3  4   3, 2, 4, 1   1 0 0 1   1, 1/2       3/8
        # 4  2   4, 3         1 0                    1/2
        # 5  2   5, 4         1 1       1, 1         1
        # 6  3   6, 5, 7      1 0 1     1, 2/3       5/9
        # 7  3   7, 6, 8      1 1 0     1, 1         2/3
        # 8  4   8, 7, 6, 5   1 0 0 0   1            1/4

        result = m(parameters, labels)["mapr-ms"].item()
        result_gt = np.mean([11 / 16, 11 / 16, 1 / 3, 3 / 8, 1 / 2, 1, 5 / 9, 2 / 3, 1 / 4])
        self.assertAlmostEqual(result, result_gt)

        m = NearestNeighboursMetrics(d, s, config={"metrics": ["mapr"], "prefetch_factor": 1})
        labels = torch.tensor([1, 1, 0, 1, 2, 2, 0, 0, 1])  # (B).
        parameters = torch.arange(len(labels))[:, None] ** 1.01  # (B, 1).

        # N  R   Nearest      Same      P       MAP@R
        # 0  4   1, 2, 3      1 0 1   1, 2/3    5/9
        # 1  4   0, 2, 3      1 0 1   1, 2/3    5/9
        # 2  3   1, 3         0 0               0
        # 3  4   2, 4, 1      0 0 1   1/3       1/9
        # 4  2   3            0                 0
        # 5  2   4            1       1         1
        # 6  3   5, 7         0 1     1/2       1/4
        # 7  3   6, 8         1 0     1         1/2
        # 8  4   7, 6, 5      0 0 0             0

        result = m(parameters, labels)["mapr"].item()
        result_gt = np.mean([5 / 9, 5 / 9, 0, 1 / 9, 0, 1, 1 / 4, 1 / 2, 0])
        self.assertAlmostEqual(result, result_gt)

    def test_toy(self):
        """Eval MAP@R on toy examples from original paper."""
        sample_size = 1000
        mapr_gt = {
            (0, 1): 0.779,
            (0, None, 1): 0.998,
            (0, None, 0, 1, None, 1): 0.714
        }
        d = DiracDistribution(config={"dim": 2, "spherical": False})
        s = NegativeL2Scorer(d)
        m = NearestNeighboursMetrics(d, s, config={"metrics": ["mapr-ms"], "prefetch_factor": 1})
        for pattern, gt in mapr_gt.items():
            embeddings1 = torch.rand(sample_size, 2)
            embeddings2 = torch.rand(sample_size, 2)
            self._apply_pattern_inplace(embeddings1, pattern, 0)
            self._apply_pattern_inplace(embeddings2, pattern, 1)
            embeddings = torch.cat((embeddings1, embeddings2))
            labels = torch.cat((torch.zeros(sample_size).long(), torch.ones(sample_size).long()))
            result = m(embeddings, labels)["mapr-ms"].item()
            self.assertTrue(abs(result - gt) < 0.05)

    def _apply_pattern_inplace(self, sample, pattern, label):
        """Apply pattern to uniform distribution."""
        pattern = tuple(p == label for p in pattern)
        num_bins = sum(pattern)
        sample[:, 0] *= num_bins
        for i, j in reversed(list(enumerate(np.nonzero(pattern)[0]))):
            mask = (sample[:, 0] >= i) & (sample[:, 0] <= i + 1)
            sample[mask, 0] += j - i
        sample[:, 0] *= 2 / len(pattern)


class TestGAPR(TestCase):
    def test_cat_variable_length(self):
        negative_scores = torch.tensor([
            [1, 2, 0, 0],
            [3, 4, 5, 0],
            [6, 0, 0, 0]
        ])
        num_negatives = torch.tensor([2, 3, 1])
        positive_scores = torch.tensor([
            [7, 0, 0, 0],
            [8, 9, 10, 11],
            [12, 0, 0, 0]
        ])
        num_positives = torch.tensor([1, 4, 1])
        scores, num_scores = GAPR._cat_variable_length(negative_scores, num_negatives, positive_scores, num_positives)
        scores_gt = torch.tensor([
            [1, 2, 7, 0, 0, 0, 0, 0],
            [3, 4, 5, 8, 9, 10, 11, 0],
            [6, 12, 0, 0, 0, 0, 0, 0]
        ])
        valid_mask = scores_gt > 0
        self.assertTrue((scores[valid_mask] == scores_gt[valid_mask]).all())
        num_scores_gt = torch.tensor([3, 7, 2])
        self.assertTrue((num_scores == num_scores_gt).all())

    def test_get_negative_scores(self):
        scores = torch.tensor([
            [1, 2, 3, 4],
            [5, 4, 7, 8],
            [9, 10, 11, 12]
        ])
        same_mask = torch.tensor([
            [1, 0, 0, 0],
            [0, 1, 0, 1],
            [0, 0, 0, 0]
        ])
        negative_scores, num_negatives = GAPR._get_negative_scores(same_mask, scores)
        scores_gt = torch.tensor([
            [2, 3, 4, 0],
            [5, 7, 0, 0],
            [9, 10, 11, 12]
        ])
        num_negatives_gt = (scores_gt > 0).sum(1)
        valid_mask = scores_gt > 0
        self.assertEqual(negative_scores.shape, scores_gt.shape)
        self.assertTrue((negative_scores[valid_mask] == scores_gt[valid_mask]).all())
        self.assertTrue((num_negatives == num_negatives_gt).all())

    def test_simple(self):
        d = DiracDistribution(config={"dim": 1, "spherical": False})
        s = NegativeL2Scorer(d)
        labels = torch.tensor([1, 1, 0, 2, 0, 1])  # (B).
        parameters = torch.tensor([0, 0.5, 1.5, 3.5, 4, 5])[:, None]  # (B, 1).

        config = {"metrics": ["gapr", "gapr-ms", "gapr-balanced", "gapr-balanced-ms"], "prefetch_factor": 1}
        m = NearestNeighboursMetrics(d, s, config=config)(parameters, labels)
        # N  2R  Nearest  Scores                 Same
        # 0  6   012345   0 0.5 1.5 3.5 4 5      110001
        # 1  6   102345   0 0.5 1 3 3.5 4.5      110001
        # 2  4   2103     0 1 1.5 2.5            1001
        # 3  2   34       0 0.5                  10
        # 4  4   4352     0 0.5 1 2.5            1001
        # 5  6   543210   0 1 1.5 3.5 4.5 5      100011
        #
        # Total weight: 12 = 2 * 6
        #
        # MERGED
        # Label:       1   1   1   1   1   1   1   1   0   0   0   0   0   0   0   0   0   1   1   0   0   0   0   0   1   1   1   1
        # Score:      0.0 0.0 0.0 0.0 0.0 0.0 0.5 0.5 0.5 0.5 1.0 1.0 1.0 1.0 1.5 1.5 1.5 2.5 2.5 3.0 3.5 3.5 3.5 4.0 4.5 4.5 5.0 5.0
        # Weight / 6:  2   2   3   6   3   2   2   2   6   3   2   3   3   2   2   3   2   3   3   2   2   2   2   2   2   2   2   2
        #
        # Score   T   Precision     Recall (/ 36)
        # -inf    0      1             0
        # 0.0     18     1             18
        # 0.5     31   22/31           22
        # 1.0     41   22/41           22
        # 1.5     48   22/48           22
        # 2.5     54   28/54           28
        # 3.0     56   28/56           28
        # 3.5     62   28/62           28
        # 4.0     64   28/64           28
        # 4.5     68   32/68           32
        # 5.0     72   36/72           36
        # inf     72     0             36

        result = m["gapr-balanced-ms"].item()
        result_gt = (18 * 1 + 4 * 22 / 31 + 6 * 28 / 54 + 4 * 32 / 68 + 4 * 36 / 72) / 36
        self.assertAlmostEqual(result, result_gt)

        # N  2R  Nearest  Scores                 Same
        # 0  4   1235     0.5 1.5 3.5 5          1001
        # 1  4   0235     0.5 1 3 4.5            1001
        # 2  2   14       1 2.5                  01
        # 3  0
        # 4  2   32       0.5 2.5                01
        # 5  4   4310     1 1.5 4.5 5            0011
        #
        # Total weight: 10 = 2 * 5
        #
        # MERGED
        # Label:       1   1   0   0   0   0   0   0   1   1   0   0   1   1   1   1
        # Score:      0.5 0.5 0.5 1.0 1.0 1.0 1.5 1.5 2.5 2.5 3.0 3.5 4.5 4.5 5.0 5.0
        # Weight / 2:  1   1   2   1   2   1   1   1   2   2   1   1   1   1   1   1
        #
        # Score   T   Precision     Recall (/ 10)
        # -inf    0      1             0
        # 0.5     4     2/4            2
        # 1.0     8                    2
        # 1.5     10                   2
        # 2.5     14    6/14           6
        # 3.0     15                   6
        # 3.5     16                   6
        # 4.5     18    8/18           8
        # 5.0     20    10/20          10
        # inf     20                   10
        result = m["gapr-balanced"].item()
        result_gt = (2 * 2 / 4 + 4 * 6 / 14 + 2 * 8 / 18 + 2 * 10 / 20) / 10
        self.assertAlmostEqual(result, result_gt)

        #labels = torch.tensor([1, 1, 0, 2, 0, 1])  # (B).
        config["metrics"] = ["gapr"]
        m = NearestNeighboursMetrics(d, s, config=config)(parameters, labels)
        # N  2R  Nearest  Scores                 Same
        # 0  4   1234     0.5 1.5 3.5 4          1000
        # 1  4   0234     0.5 1 3 3.5            1000
        # 2  2   10       1 1.5                  00
        # 3  0
        # 4  2   35       0.5 1                  00
        # 5  4   4321     1 1.5 3.5 4.5          0001
        #
        # Total weight: 10 = 2 * 5
        #
        # MERGED
        # Label:       1   1   0   0   0   0   0   0   0   0   0   0   0   0   0   1
        # Score:      0.5 0.5 0.5 1.0 1.0 1.0 1.0 1.5 1.5 1.5 3.0 3.5 3.5 3.5 4.0 4.5
        # Weight / 2:  1   1   2   1   2   2   1   1   2   1   1   1   1   1   1   1
        #
        # Score   T   Precision     Recall (/ 10)
        # -inf    0      1             0
        # 0.5     4     2/4            2
        # 1.0     10                   2
        # 1.5     14                   2
        # 3.0     15                   2
        # 3.5     18                   2
        # 4.0     19                   2
        # 4.5     20    3/20           3
        # inf     20                   3
        result = m["gapr"].item()
        result_gt = (2 * 2 / 4 + 1 * 3 / 20) / 10
        self.assertAlmostEqual(result, result_gt)

    def test_border(self):
        d = DiracDistribution(config={"dim": 1, "spherical": False})
        s = NegativeL2Scorer(d)
        m = NearestNeighboursMetrics(d, s, config={"metrics": ["gapr-balanced-ms"], "prefetch_factor": 1})

        labels = torch.tensor([1, 1, 1, 0])
        parameters = torch.arange(len(labels))[:, None].float()  # (B, 1).
        # N  2R  Nearest  Scores    Same
        # 0  2   03       03        10
        # 1  2   13       02        10
        # 2  2   23       01        10
        # 3  2   32       01        10
        #
        # MERGED
        # Label:       1 1 1 1 0 0 0 0
        # Score:       0 0 0 0 1 1 2 3
        # Weight / 2:  2 2 2 2 2 2 2 2
        #
        # Score  T   Precision      Recall (/ 8)
        # -inf   0       1            8
        #  0     8       1            8
        #  1     12     9/12          8
        #  2     14    11/15          8
        #  3     16    11/16          8
        # inf    16    11/16          8
        result = m(parameters, labels)["gapr-balanced-ms"].item()
        self.assertAlmostEqual(result, 1)

        labels = torch.tensor([1, 1, 1, 1])
        parameters = torch.arange(len(labels))[:, None].float()  # (B, 1).
        result = m(parameters, labels)["gapr-balanced-ms"].item()
        self.assertAlmostEqual(result, 1)


class TestRecallK(TestCase):
    def test_simple(self):
        d = DiracDistribution(config={"dim": 1, "spherical": False})
        s = NegativeL2Scorer(d)
        labels = torch.tensor([1, 1, 0, 2, 0, 1])  # (B).
        parameters = torch.tensor([0, 0.5, 1.5, 3.5, 4, 5])[:, None]  # (B, 1).

        config = {"metrics": ["recall"], "recall_k_values": (1, 2, 3, 4, 5, 10), "prefetch_factor": 1}
        m = NearestNeighboursMetrics(d, s, config=config)(parameters, labels)
        # Item    Nearest   Same
        #
        # 0       12345     10001
        # 1       02345     10001
        # 2       10345     00010
        # 3       45210     00000 (excluded as one-element class).
        # 4       35210     00100
        # 5       43210     00011
        self.assertAlmostEqual(m["recall@1"], 2 / 5)
        self.assertAlmostEqual(m["recall@2"], 2 / 5)
        self.assertAlmostEqual(m["recall@3"], 3 / 5)
        self.assertAlmostEqual(m["recall@4"], 1)
        self.assertAlmostEqual(m["recall@5"], 1)
        self.assertAlmostEqual(m["recall@10"], 1)


class TestERCRecallK(TestCase):
    def test_simple(self):
        d = GMMDistribution(config={"dim": 1})
        s = NegativeL2Scorer(d)
        labels = torch.tensor([1, 1, 0, 2, 0, 1])  # (B).
        centers = torch.tensor([0, 0.5, 1.5, 3.5, 4, 5])  # (B).
        confidences = torch.tensor([0, 2, 4, 5, 3, 1]).float()  # (B).
        parameters = torch.stack([centers, -confidences], 1)  # (B, 2).

        config = {"metrics": ["erc-recall@1"], "recall_k_values": (1, 2, 3, 4, 5, 10), "prefetch_factor": 1}
        m = NearestNeighboursMetrics(d, s, config=config)(parameters, labels)
        # Item    Nearest   Confidence   Same
        #
        # 0       12345         0        1
        # 1       02345         2        1
        # 2       10345         4        0
        # 3       45210         5        0 (excluded as one-element class).
        # 4       35210         3        0
        # 5       43210         1        0
        #
        # Same ordered by descending confidence:
        # 0 0 1 0 1
        #
        # Metrics
        # 0/1 0/2 1/3 1/4 2/5
        #
        self.assertAlmostEqual(m["erc-recall@1"], 1 - np.mean([0, 0, 1/3, 1/4, 2/5]), places=6)


class TestERCMAPR(TestCase):
    def test_simple(self):
        d = GMMDistribution(config={"dim": 1})
        s = NegativeL2Scorer(d)

        labels = torch.tensor([1, 1, 0, 1, 2, 2, 0, 0, 1])  # (B).
        centers = torch.arange(len(labels)) ** 1.01  # (B, 1).
        confidences = torch.tensor([0, 2, 4, 6, 8, 7, 5, 3, 1]).float()  # (B).
        parameters = torch.stack([centers, -confidences], 1)  # (B, 2).
        m = NearestNeighboursMetrics(d, s, config={"metrics": ["erc-mapr"], "prefetch_factor": 1})(parameters, labels)

        # N  R   Nearest      Same      P       MAP@R   Confidence
        # 0  4   1, 2, 3      1 0 1   1, 2/3    5/9     0
        # 1  4   0, 2, 3      1 0 1   1, 2/3    5/9     2
        # 2  3   1, 3         0 0               0       4
        # 3  4   2, 4, 1      0 0 1   1/3       1/9     6
        # 4  2   3            0                 0       8
        # 5  2   4            1       1         1       7
        # 6  3   5, 7         0 1     1/2       1/4     5
        # 7  3   6, 8         1 0     1         1/2     3
        # 8  4   7, 6, 5      0 0 0             0       1
        #
        # MAP@R ordered by descending confidence:
        # 0 1 1/9 1/4 0 1/2 5/9 0 5/9
        #

        maprs = np.array([0, 1, 1/9, 1/4, 0, 1/2, 5/9, 0, 5/9])
        erc_mapr_gt = 1 - (np.cumsum(maprs) / np.arange(1, len(maprs) + 1)).mean()

        self.assertAlmostEqual(m["erc-mapr"], erc_mapr_gt, places=6)


class TestGroupedRecall(TestCase):
    def test_simple(self):
        d = DiracDistribution(config={"dim": 1})
        s = NegativeL2Scorer(d)
        config = {
            "metrics": ["recall"],
            "num_grouped_classes": 2,
            "grouping_seed": None  # Disable label shuffle.
        }

        labels = torch.tensor([1, 1, 0, 1, 2, 2, 0, 0, 1])  # (B).
        parameters = torch.arange(len(labels)) ** 1.01  # (B, 1).
        # Selected labels: 0, 1. Group with single label 2 is skipped.
        # Final parameters: [0, 1, 2, 3, 6, 7, 8].
        # Final Labels:     [1, 1, 0, 1, 0, 0, 1].
        # Recall:
        # 1, 1, 0, 0, 1, 1, 0.
        recall = GroupedNearestNeighboursMetrics(d, s, config=config)(parameters.unsqueeze(-1), labels)["grouped2-recall@1"]
        recall_gt = 4 / 7
        self.assertAlmostEqual(recall, recall_gt)

    def test_grouping(self):
        d = DiracDistribution(config={"dim": 1})
        s = NegativeL2Scorer(d)
        calculator = GroupedNearestNeighboursMetrics(d, s, config={"metrics": ["recall"]})

        for seed in range(5):
            with tmp_seed(seed), torch.no_grad():
                source_indices = [0, 1, 2, 3, 4, 5, 6]
                source_labels = [1, 3, 2, 3, 4, 3, 4]
                indices, labels = calculator._select_groups(
                    torch.tensor(source_indices),
                    torch.tensor(source_labels),
                    grouping_factor=2,
                    seed=seed
                )
                indices = [v.numpy().tolist() for v in indices]
                labels = [v.numpy().tolist() for v in labels]
                joined_indices = sum(indices, [])
                joined_labels = sum(labels, [])
                # All elements are in groups.
                self.assertEqual(list(sorted(joined_indices)), list(range(len(joined_labels))))
                # Labels are correctly assigned.
                for i, l in zip(joined_indices, joined_labels):
                    self.assertEqual(source_labels[i], l)
                # Each group has K labels.
                for l in labels:
                    self.assertEqual(len(set(l)), 2)


if __name__ == "__main__":
    main()
