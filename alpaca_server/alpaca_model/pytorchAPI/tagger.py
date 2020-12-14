from .utils import create_batches, get_entities, prepare_dataset
import spacy
from torch.autograd import Variable
import torch
nlp = spacy.load('en_core_web_sm')

class Tagger(object):
    """A alpaca_model API that tags input sentence.

    Attributes:
        model: Model.
        preprocessor: Transformer. Preprocessing data for feature extraction.
        tokenizer: Tokenize input sentence. Default tokenizer is `str.split`.
    """
    def __init__(self, model, preprocessor, tokenizer=str.split):
        self.model = model
        self.preprocessor = preprocessor
        self.tokenizer = tokenizer
        self.usecuda = False

    def predict_proba(self, text):
        """Probability estimates.

        The returned estimates for all classes are ordered by the
        label of classes.

        Args:
            text : string, the input text.

        Returns:
            y : array-like, shape = [num_words, num_classes]
            Returns the probability of the word for each class in the alpaca_model,
        """
        assert isinstance(text, str)
        doc = nlp(text)
        words = [token.text for token in doc]
        dataset = prepare_dataset([words], None, self.preprocessor)
        testdata = create_batches(dataset, batch_size=1, str_words=True, tag_padded=False)


        words = testdata[0]['words']
        chars = testdata[0]['chars']
        caps = testdata[0]['caps']
        mask = testdata[0]['tagsmask']
        wordslen = testdata[0]['wordslen']
        charslen = testdata[0]['charslen']

        if self.usecuda:
            words = Variable(torch.LongTensor(words)).cuda()
            chars = Variable(torch.LongTensor(chars)).cuda()
            caps = Variable(torch.LongTensor(caps)).cuda()
            mask = Variable(torch.LongTensor(mask)).cuda()
        else:
            words = Variable(torch.LongTensor(words))
            chars = Variable(torch.LongTensor(chars))
            caps = Variable(torch.LongTensor(caps))
            mask = Variable(torch.LongTensor(mask))

        _, out = self.model.decode(words, chars, caps, wordslen, charslen, mask, usecuda=self.usecuda)

        return out

    def _get_tags(self, pred):
        tags = self.preprocessor.inverse_transform(pred)

        return tags

    def _build_response(self, sent, tags):
        doc = nlp(sent)
        words = [token.text for token in doc]
        res = {
            'words': words,
            'entities': [

            ],
            'tags': tags
        }
        chunks = get_entities(tags)

        for chunk_type, chunk_start, chunk_end in chunks:
            chunk_end += 1
            entity = {
                'text': ' '.join(words[chunk_start: chunk_end]),
                'type': chunk_type,
                'beginOffset': chunk_start,
                'endOffset': chunk_end
            }
            res['entities'].append(entity)

        return res

    def analyze(self, text):
        """Analyze text and return pretty format.

        Args:
            text: string, the input text.

        Returns:
            res: dict.
        """
        pred = self.predict_proba(text)
        tags = self._get_tags(pred)
        res = self._build_response(text, tags)

        return res

    def predict(self, text):
        """Predict using the alpaca_model.

        Args:
            text: string, the input text.

        Returns:
            tags: list, shape = (num_words,)
            Returns predicted values.
        """
        pred = self.predict_proba(text)
        tags = self._get_tags(pred)

        return tags
