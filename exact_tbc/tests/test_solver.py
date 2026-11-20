import math
import subprocess
import unittest
from pathlib import Path


BINARY = Path(__file__).resolve().parents[1] / "zhang_tbc" / "run"


def run_solver(graph_path):
    result = subprocess.run(
        [BINARY, graph_path], stdout=subprocess.PIPE, text=True, check=True
    )

    scores = {}
    for line in result.stdout.splitlines():
        fields = line.split()
        if len(fields) != 2:
            continue
        try:
            node = int(fields[0])
        except ValueError:
            continue
        scores[node] = float(fields[1])
    return scores


class TestAddition(unittest.TestCase):
    def test_commutativity(self):
        self.assertEqual(2 + 1, 1 + 2)


class TestSolver(unittest.TestCase):
    cases = [
        ("primitive_chain.txt", "primitive_chain_expected.txt"),
        ("equal_time_chain.txt", "equal_time_chain_expected.txt"),
        ("late_direct.txt", "late_direct_expected.txt"),
        ("diamond.txt", "diamond_expected.txt"),
        ("office_2000.txt", "office_2000_expected.txt"),
    ]

    def test_graphs(self):
        tests_dir = Path(__file__).parent

        for graph_file, expected_file in self.cases:
            with self.subTest(graph=graph_file):
                graph_path = tests_dir / graph_file
                expected_path = tests_dir / expected_file

                expected = {}
                for line in expected_path.read_text().splitlines():
                    node, score = line.split()
                    expected[int(node)] = float(score)

                actual = run_solver(graph_path)
                self.assertEqual(actual.keys(), expected.keys())
                rel_tol = 5e-6 if graph_file == "office_2000.txt" else 0.0
                for node, score in expected.items():
                    self.assertTrue(
                        math.isclose(actual[node], score, rel_tol=rel_tol),
                        f"node {node}: got {actual[node]}, expected {score}",
                    )

    def test_path_count_overflow(self):
        tests_dir = Path(__file__).parent
        # Each of the 31 hops has two possible timestamps: 2**31 shortest paths.
        result = subprocess.run(
            [BINARY, tests_dir / "overflow_chain.txt"], capture_output=True, text=True
        )
        self.assertGreater(result.returncode, 0)
        expected = (tests_dir / "overflow_chain_expected_stderr.txt").read_text()
        self.assertEqual(result.stderr.strip(), expected.strip())


if __name__ == "__main__":
    unittest.main()
