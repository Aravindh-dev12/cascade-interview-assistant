import unittest

from engine.question_detector import (
    is_substantive_question,
    should_attach_camera,
    should_attach_screen,
)


class QuestionDetectorTests(unittest.TestCase):
    def test_questions(self):
        self.assertTrue(is_substantive_question("Can you explain dependency injection?"))
        self.assertTrue(
            is_substantive_question(
                "Explain the difference between a process and a thread"
            )
        )
        self.assertTrue(is_substantive_question("How would you optimize this algorithm"))

    def test_acknowledgements(self):
        self.assertFalse(is_substantive_question("Okay."))
        self.assertFalse(is_substantive_question("Great"))
        self.assertFalse(is_substantive_question("yes"))

    def test_screen_reference(self):
        self.assertTrue(should_attach_screen("Can you fix this code?"))
        self.assertFalse(should_attach_screen("What is dependency injection?"))

    def test_camera_reference(self):
        self.assertTrue(should_attach_camera("What am I holding in front of the camera?"))
        self.assertTrue(should_attach_camera("Look at this object"))
        # Camera prompts intentionally also use the overlay's existing visual slot.
        self.assertTrue(should_attach_screen("What do you see on the camera?"))
        self.assertFalse(should_attach_camera("Explain dependency injection"))


if __name__ == "__main__":
    unittest.main()
