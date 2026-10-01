"""Text helpers and ranking (research/textutil.py, research/rank.py)."""

import unittest

import numpy as np

from research import rank
from research.ir import check_text
from research.textutil import ascii_fold, clean_line, split_sentences, tokenize


class TextUtilTest(unittest.TestCase):
    """Cleaning, folding, splitting and tokenizing page text; checking text to be typed."""

    def test_citations_and_pronunciations_are_removed(self):
        """clean_line drops citation marks, [citation needed] and IPA pronunciations."""
        line = ("NASA selected lunar orbit rendezvous, a plan championed by John Houbolt (/ˈhuːboʊlt/) "
                "in 1962.[17][a] It was cheaper.[citation needed]")
        self.assertEqual(clean_line(line), "NASA selected lunar orbit rendezvous, a plan championed by John Houbolt "
                                           "in 1962. It was cheaper.")

    def test_useful_parentheses_stay(self):
        """clean_line keeps ordinary parentheses such as an abbreviation."""
        self.assertEqual(clean_line("Apollo Lunar Module (LM) landed."), "Apollo Lunar Module (LM) landed.")

    def test_ascii_fold(self):
        """Dashes, curly quotes, accents and ellipses become their ASCII equivalents."""
        self.assertEqual(ascii_fold("1961–1972 “Moon” café …"), '1961-1972 "Moon" cafe ...')

    def test_sentence_split_keeps_abbreviations_and_initials(self):
        """Sentences aren't split after initials, "U.S.", "e.g.", "Dr." or decimal points."""
        text = ("President John F. Kennedy spoke to Congress in May 1961. The U.S. program grew. "
                "It cost about 25.8 billion dollars, e.g. for the Saturn V. Dr. Braun led it.")
        self.assertEqual(split_sentences(text), [
            "President John F. Kennedy spoke to Congress in May 1961.",
            "The U.S. program grew.",
            "It cost about 25.8 billion dollars, e.g. for the Saturn V.",
            "Dr. Braun led it.",
        ])

    def test_tokenize_drops_stopwords_and_stems(self):
        """tokenize lowercases, drops stopwords and reduces words to their stems."""
        self.assertEqual(tokenize("The missions landed on the Moon"), ["mission", "land", "moon"])

    def test_check_text_flags_word_autocorrect_triggers(self):
        """check_text flags AutoCorrect triggers, non-ASCII and line breaks, but not citations."""
        self.assertTrue(check_text("Copyright (c) NASA"))
        self.assertTrue(check_text("from 1969--1972"))
        self.assertTrue(check_text("café"))
        self.assertTrue(check_text("two\nlines"))
        self.assertEqual(check_text("Apollo 11 landed in 1969 [1]."), [])


class RankTest(unittest.TestCase):
    """TF-IDF, TextRank, MMR and k-means on tiny examples."""

    def setUp(self):
        """Three related "sentences" about Apollo and one unrelated one, as token lists."""
        docs = [["moon", "land", "apollo"], ["moon", "apollo", "crew"], ["apollo", "rocket", "saturn"],
                ["cake", "recipe", "flour"]]
        self.X, self.vocab, self.idf = rank.tfidf_matrix(docs)
        self.S = rank.cosine_sim(self.X)

    def test_rows_are_normalised(self):
        """Every TF-IDF row has unit length (so dot products are cosine similarities)."""
        np.testing.assert_allclose(np.linalg.norm(self.X, axis=1), 1.0, rtol=1e-5)

    def test_textrank_favours_the_connected_sentences(self):
        """TextRank scores sum to 1, rank the unrelated sentence last and are deterministic."""
        r = rank.textrank(self.S, threshold=0.05)
        self.assertAlmostEqual(float(r.sum()), 1.0, places=5)
        self.assertLess(r[3], min(r[0], r[1], r[2]))
        np.testing.assert_allclose(r, rank.textrank(self.S, threshold=0.05))

    def test_mmr_avoids_near_duplicates(self):
        """MMR picks a less relevant but different sentence over a near-duplicate of one chosen."""
        S = np.array([[1, 0.95, 0.1], [0.95, 1, 0.1], [0.1, 0.1, 1]], dtype=np.float32)
        rel = np.array([1.0, 0.99, 0.6])
        self.assertEqual(rank.mmr_pick(rel, S, [1, 2], chosen=[0], lam=0.5), 2)

    def test_kmeans_separates_clusters(self):
        """k-means puts two well-separated pairs of points into two clusters."""
        X = np.array([[0, 0], [0, 0.1], [5, 5], [5, 5.1]], dtype=np.float32)
        labels, _ = rank.kmeans(X, 2, np.random.default_rng(0))
        self.assertEqual(labels[0], labels[1])
        self.assertEqual(labels[2], labels[3])
        self.assertNotEqual(labels[0], labels[2])


if __name__ == "__main__":
    unittest.main()
