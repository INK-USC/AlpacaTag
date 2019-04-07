class NounChunk(object):
    def __init__(self, nlp):
        self.nlp = nlp

    def chunking(self, text):
        doc = self.nlp(text)
        words = [token.text for token in doc]
        chunklist = []
        for chunk in doc.noun_chunks:
            chunkdict = {}
            chunkdict['nounchunk'] = chunk.text
            chunkdict['left'] = chunk.start
            chunkdict['right'] = chunk.end
            chunklist.append(chunkdict)

        res = self._build_response(chunklist, text, words)

        return res

    def _build_response(self, chunks, text, words):
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
