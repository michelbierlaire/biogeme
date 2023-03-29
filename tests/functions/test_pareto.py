"""
Test the patero module

:author: Michel Bierlaire
:date: Wed Mar 15 11:23:14 2023
"""

import unittest
import tempfile

import biogeme.pareto as par
import biogeme.exceptions as excep


class TestPareto(unittest.TestCase):
    def setUp(self):
        self.e1 = par.SetElement(element_id='e1', objectives=[1, 3])
        self.e2 = par.SetElement(element_id='e2', objectives=[4, 3])
        self.e3 = par.SetElement(element_id='e3', objectives=[1, 1])
        self.e4 = par.SetElement(element_id='e4', objectives=[4, 1])
        self.wrong_e1 = par.SetElement(element_id='e1', objectives=[4, 1])

    def test_set_element(self):
        r1 = self.e3.dominates(self.e2)
        self.assertEqual(r1, True)

        r2 = self.e3.dominates(self.e1)
        self.assertEqual(r2, True)

        r3 = self.e1.dominates(self.e4)
        self.assertEqual(r3, False)

        r4 = self.e4.dominates(self.e1)
        self.assertEqual(r4, False)

    def test_pareto(self):
        pareto = par.Pareto()

        pareto.add(self.e1)
        self.assertSetEqual(pareto.pareto, {self.e1})
        self.assertSetEqual(pareto.considered, {self.e1})
        self.assertSetEqual(pareto.removed, set())

        pareto.add(self.e2)
        self.assertSetEqual(pareto.pareto, {self.e1})
        self.assertSetEqual(pareto.considered, {self.e1, self.e2})
        self.assertSetEqual(pareto.removed, set())

        pareto.add(self.e3)
        self.assertSetEqual(pareto.pareto, {self.e3})
        self.assertSetEqual(pareto.considered, {self.e1, self.e2, self.e3})
        self.assertSetEqual(pareto.removed, {self.e1})

        pareto.add(self.e4)
        self.assertSetEqual(pareto.pareto, {self.e3})
        self.assertSetEqual(pareto.considered, {self.e1, self.e2, self.e3, self.e4})
        self.assertSetEqual(pareto.removed, {self.e1})

        with self.assertRaises(excep.biogemeError):
            pareto.add(self.wrong_e1)

    def test_dump_load(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            filename = tmpdirname + 'tmp.pickle'
            pareto = par.Pareto(filename=filename)
            pareto.add(self.e1)
            pareto.add(self.e2)
            pareto.add(self.e3)
            pareto.add(self.e4)
            pareto.dump()
            restore_pareto = par.Pareto(filename=filename)
            self.assertSetEqual(pareto.pareto, restore_pareto.pareto)
            self.assertSetEqual(pareto.considered, restore_pareto.considered)
            self.assertSetEqual(pareto.removed, restore_pareto.removed)


if __name__ == '__main__':
    unittest.main()
