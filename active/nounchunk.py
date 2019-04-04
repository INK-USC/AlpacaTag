import nltk

class NounChunk(object):
    def __init__(self, nlp):
        self.nlp = nlp

    def chunking(self, text):
        doc = self.nlp(text)

        chunklist = []
        for chunk in doc.noun_chunks:
            chunkdict = {}
            chunkdict['nounchunk'] = chunk.text
            chunkdict['left'] = chunk.root.left_edge.i
            chunkdict['right'] = chunk.root.left_edge.i + len(str.split(chunk.text))
            chunklist.append(chunkdict)

        res = self._build_response(chunklist, text)

        return res

    def _build_response(self, chunks, text):
        words = nltk.word_tokenize(text)
        res = {
            'words': words,
            'entities': [

            ]
        }

        for chunkdict in chunks:
            entity = {
                'text': chunkdict['nounchunk'],
                'type': None,
                'score': None,
                'beginOffset': chunkdict['left'],
                'endOffset': chunkdict['right']
            }
            res['entities'].append(entity)

        return res
