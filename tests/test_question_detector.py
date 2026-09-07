import unittest

from engine.question_detector import is_substantive_question, should_attach_screen


class QuestionDetectorTests(unittest.TestCase):
    def test_questions(self):
        self.assertTrue(is_substantive_question("Can you explain dependency injection?"))
        self.assertTrue(is_substantive_question("Explain the difference between a process and a thread"))
        self.assertTrue(is_substantive_question("How would you optimize this algorithm"))

    def test_acknowledgements(self):
        self.assertFalse(is_substantive_question("Okay."))
        self.assertFalse(is_substantive_question("Great"))
        self.assertFalse(is_substantive_question("yes"))

    def test_screen_reference(self):
        self.assertTrue(should_attach_screen("Can you fix this code?"))
        self.assertFalse(should_attach_screen("What is dependency injection?"))


if __name__ == "__main__":
    unittest.main()
