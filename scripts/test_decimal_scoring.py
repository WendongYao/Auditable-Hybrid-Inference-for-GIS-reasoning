import unittest
from fractions import Fraction
import score

class DecimalBoundaryTests(unittest.TestCase):
    def test_signed_inclusive_edges(self):
        for p,g in [(7.3,7.1),(-2.,-2.2),(0.3,0.1),(-0.3,-0.1)]:
            self.assertTrue(score.numeric_within_tolerance(p,g,.2))
    def test_no_epsilon_or_rounding(self):
        self.assertFalse(score.numeric_within_tolerance('7.300000000000001','7.1','.2'))
        self.assertFalse(score.numeric_within_tolerance(float('nan'),1,.2))
        self.assertFalse(score.numeric_within_tolerance(1,1,-.2))
    def test_independent_rational_grid(self):
        for i in range(-100,101):
            for j in (-6,-5,-2,-1,0,1,2,5,6):
                p,g=str((i+j)/10),str(i/10)
                for t in ('.2','.5'):
                    self.assertEqual(score.numeric_within_tolerance(p,g,t),abs(Fraction(p)-Fraction(g))<=Fraction(t))
    def test_none_and_parse_semantics(self):
        g={'answer':'none','answer_numeric':None,'eval':{'metric':'num_or_none_within_tol','tolerance':{'value':.5}}}
        self.assertTrue(score.score_one(g,{'answer_parsed':'none'})[0])
        self.assertFalse(score.score_one(g,{'answer_parsed':1})[0])
        self.assertFalse(score.score_one(g,{'answer_parsed':None,'parse_error':'truncated'})[0])

if __name__=='__main__':unittest.main()
