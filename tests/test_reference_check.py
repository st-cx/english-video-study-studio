import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.reference_check import reference_issues


def words(text):
    return [{'text': token, 'start_ms': i * 500 + 100, 'end_ms': i * 500 + 400} for i, token in enumerate(text.split())]


class ReferenceCheckTests(unittest.TestCase):
    def test_content_replacement_is_found_despite_high_global_similarity(self):
        source = words("The elephants have really long fronts and that's cool.")
        issues = reference_issues(source, "The elephants have really long trunks and that's cool.")
        self.assertEqual(len(issues), 1)
        self.assertEqual((issues[0]['recognized'], issues[0]['reference']), ('fronts', 'trunks'))
        self.assertEqual((issues[0]['start_ms'], issues[0]['end_ms']), (2600, 2900))

    def test_alright_contractions_punctuation_and_fillers_are_compatible(self):
        self.assertEqual(reference_issues(words("Alright, um I don't think we're going to leave."), 'All right, I do not think we are going to leave!'), [])

    def test_numeric_written_and_spoken_forms_are_equivalent(self):
        self.assertEqual(reference_issues(words('It has one point three million words.'), 'It has 1,300,000 words.'), [])
        self.assertEqual(reference_issues(words('There are one hundred and thirty thousand words.'), 'There are 130,000 words.'), [])

    def test_numeric_difference_is_not_hidden_by_similar_surrounding_words(self):
        issues = reference_issues(words('The book has 13 million words.'), 'The book has 1.3 million words.')
        self.assertEqual(len(issues), 1)
        self.assertIn('数字', issues[0]['reason'])

    def test_missing_negation_is_reported(self):
        issues = reference_issues(words('I can read it.'), "I can't read it.")
        self.assertEqual(len(issues), 1)
        self.assertIn('否定', issues[0]['reason'])
        self.assertEqual(issues[0]['reference'], "can't")
        self.assertLessEqual(issues[0]['start_ms'], issues[0]['end_ms'])

    def test_name_and_initialism_differences_are_reported(self):
        self.assertTrue(reference_issues(words('I am George.'), 'I am Georgie.'))
        issues = reference_issues(words('It is from the US.'), 'It is from the UK.')
        self.assertIn('专名', issues[0]['reason'])

    def test_function_word_only_variation_is_not_a_content_dispute(self):
        self.assertEqual(reference_issues(words('It is in a brain.'), 'It is in the brain.'), [])

    def test_missing_content_is_anchored_to_neighboring_measured_words(self):
        source = words('The elephants are here.')
        issues = reference_issues(source, 'The large elephants are here.')
        self.assertEqual(issues[0]['reference'], 'large')
        self.assertEqual((issues[0]['start_ms'], issues[0]['end_ms']), (100, 900))

    def test_connected_tokens_share_measured_span_without_modifying_input(self):
        source = [{'text': "we don't", 'start_ms': 100, 'end_ms': 800}, {'text': 'know', 'start_ms': 900, 'end_ms': 1100}]
        original = [dict(word) for word in source]
        issues = reference_issues(source, 'we do know')
        self.assertIn('否定', issues[0]['reason'])
        self.assertEqual(source, original)
        issue = reference_issues([{'text': 'one point three', 'start_ms': 100, 'end_ms': 800}], 'thirteen')
        self.assertEqual(issue[0]['recognized'], 'one point three')

    def test_no_reference_or_no_speech_is_not_a_perfect_match_claim(self):
        self.assertEqual(reference_issues(words('Hello.'), ''), [])
        self.assertEqual(reference_issues([], 'Hello.'), [])

    def test_two_independent_quantities_are_not_added_into_an_equivalent_number(self):
        self.assertTrue(reference_issues(words('There are one and two elephants.'), 'There are three elephants.'))
        self.assertTrue(reference_issues(words('The code is one two.'), 'The code is three.'))

    def test_hyphenated_spelled_numbers_match_digits(self):
        self.assertEqual(reference_issues(words('There are thirty-two elephants.'), 'There are 32 elephants.'), [])

    def test_all_caps_reference_does_not_turn_articles_into_proper_names(self):
        self.assertEqual(reference_issues(words('There is a trunk.'), 'THERE IS THE TRUNK.'), [])

    def test_explicit_mt_abbreviation_matches_mount_without_fuzzy_name_matching(self):
        self.assertEqual(reference_issues(words('We climbed Mount Everest.'), 'We climbed Mt. Everest.'), [])
        self.assertTrue(reference_issues(words('We climbed Mount Rainier.'), 'We climbed Mt. Everest.'))

    def test_reference_confirmed_joined_numeric_spelling_preserves_measured_span(self):
        from app.transcript import normalize_orthography
        source = [{'text': '50', 'start_ms': 100, 'end_ms': 300},
                  {'text': ',000', 'start_ms': 300, 'end_ms': 600},
                  {'text': 'people.', 'start_ms': 700, 'end_ms': 1000}]
        normalized, edits = normalize_orthography(source, '50,000 people.')
        self.assertEqual((normalized[0]['start_ms'], normalized[0]['end_ms']), (100, 600))
        self.assertEqual(len(edits), 1)
        self.assertEqual(reference_issues(normalized, '50,000 people.'), [])

    def test_ok_is_an_acknowledgement_not_a_proper_name(self):
        self.assertEqual(reference_issues(words('OK so we can begin.'), 'We can begin.'), [])
        self.assertEqual(reference_issues(words('We can begin.'), 'OK, we can begin.'), [])
        self.assertTrue(reference_issues(words('It is from US.'), 'It is from UK.'))

    def test_leading_decimal_is_not_misread_as_an_integer(self):
        for spoken in ('0.01', '0 .01', 'zero point zero one'):
            self.assertEqual(reference_issues(words(f'Within {spoken} is enough.'), 'Within .01 is enough.'), [])
        self.assertTrue(reference_issues(words('Within 0 .0001 is enough.'), 'Within .001 is enough.'))
        self.assertTrue(reference_issues(words('Pi is 3 .159.'), 'Pi is 3.14159.'))

    def test_spoken_separator_matches_an_explicit_code_identifier(self):
        self.assertEqual(reference_issues(words('I say random dot rand int.'), 'I say random.randint.'), [])
        self.assertEqual(reference_issues(words('Save pi underscore approx.'), 'Save pi_approx.'), [])
        self.assertTrue(reference_issues(words('I say random dot rand int.'), 'I say random.randrange.'))
        self.assertTrue(reference_issues(words('Save pi underscore of prox.'), 'Save pi_approx.'))
        self.assertTrue(reference_issues(words('Place a dot here.'), 'Place here.'))




    def test_toward_towards_is_accepted_but_direction_changes_are_not(self):
        self.assertEqual(reference_issues(words('Move towards the screen.'), 'Move toward the screen.'), [])
        self.assertTrue(reference_issues(words('Move towards the screen.'), 'Move away from the screen.'))

    def test_s_expands_only_when_explicit_reference_has_the_same_noun_is(self):
        self.assertEqual(reference_issues(words("Division's applied to integers."), 'Division is applied to integers.'), [])
        self.assertTrue(reference_issues(words("Division's applied to integers."), 'Division has applied to integers.'))
        self.assertTrue(reference_issues(words("Alice's book is here."), 'Alice is here.'))

    def test_psets_spacing_matches_exact_spelling_without_hiding_plural_changes(self):
        self.assertEqual(reference_issues(words('The p sets are useful.'), 'The psets are useful.'), [])
        self.assertTrue(reference_issues(words('The p set is useful.'), 'The psets are useful.'))


if __name__ == '__main__':
    unittest.main(verbosity=2)
