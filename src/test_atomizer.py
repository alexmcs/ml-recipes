from unittest import TestCase
from atomizer import Atomizer


class TestAtomizer(TestCase):
    def test_normalize(self):
        self.assertEquals(
            Atomizer.normalize_token(
                'downloaded'
            ), 'download'
        )

        self.assertEquals(
            Atomizer.normalize_multitoken(
                'Downloaded Files'
            ), 'download file'
        )

    def test_atomize(self):
        self.assertEquals(
            Atomizer().atomize(
                'My dear friend'
            ),
            ['dear', 'friend']
        )

        self.assertEquals(
            Atomizer(ngram_range=(1, 2)).atomize(
                'My dear friend'
            ),
            ['dear', 'friend', 'dear friend']
        )

        self.assertEquals(
            Atomizer(ngram_range=(1, 2)).atomize(
                'red cubes'
            ),
            ['red', 'cube', 'red cube']
        )

        self.assertEquals(
            Atomizer(ngram_range=(1, 2)).atomize(
                'white white white box'
            ),
            ['white', 'white', 'white', 'box', 'white box']
        )

        self.assertEquals(
            Atomizer(ngram_range=(1, 2),
                     stop_words=['stop']).atomize(
                'Red stop Green stop stop stop Blue stop Yellow Orange'
            ),
            ['red', 'green', 'blue', 'yellow', 'orange', 'yellow orange']
        )

    def test_atomize_with_boosting(self):
        self.assertEquals(
            Atomizer(ngram_range=(1, 1),
                     boost_terms={'book': 3}).atomize(
                'book, door, window'
            ),
            ['book', 'book', 'book', 'door', 'window']
        )

        self.assertEquals(
            Atomizer(ngram_range=(1, 2),
                     stop_words=['div'],
                     boost_terms={'black stone': 3}).atomize(
                'black stone div red brick'
            ),
            ['black', 'stone', 'red', 'brick',
             'black stone', 'black stone', 'black stone',
             'red brick']
        )
