import unittest
import random

from fig_ga_svm.data import GENE_POOL
from fig_ga_svm.optimizers.genetic_algorithm import GeneticAlgorithmOptimizer, BitVector

class TestGeneticAlgorithmOptimizer(unittest.TestCase):
    def setUp(self):
        self.optimizer = GeneticAlgorithmOptimizer()
        self.num_genes = min(5, len(GENE_POOL))
        # Create two random parents
        self.parent1 = self.optimizer._create_individual(self.num_genes)
        self.parent2 = self.optimizer._create_individual(self.num_genes)

    def test_blx_crossover(self):
        child1, child2 = self.optimizer._blx_crossover(self.parent1, self.parent2, self.num_genes, alfa=0.3)
        # Check children are tuples of correct length
        self.assertEqual(len(child1), len(GENE_POOL))
        self.assertEqual(len(child2), len(GENE_POOL))
        # Check exactly num_genes bits set to 1
        self.assertEqual(sum(child1), self.num_genes)
        self.assertEqual(sum(child2), self.num_genes)
        # Check all bits are 0 or 1
        self.assertTrue(all(bit in (0, 1) for bit in child1))
        self.assertTrue(all(bit in (0, 1) for bit in child2))

if __name__ == "__main__":
    unittest.main()
