import numpy as np
import spacy

GLOVE_DIR = '/media/data/glove/glove.6B'
nlp = spacy.load('en')

def glove_to_spacy(vectors_path, output_path, nlp):
    print('Loading GloVe vectors: {}'.format(vectors_path))
    with open(vectors_path, 'r') as file_:
        lines = file_.readlines()
        print('Assigning {:,} spaCy vectors'.format(len(lines)))
    for line in tqdm.tqdm(lines, leave=False):
        pieces = line.split(' ')
        word = pieces[0]
        vector = np.asarray([float(v) for v in pieces[1:]], dtype='f')
        nlp.vocab.set_vector(word, vector)
    print('Saving spaCy vector model: {}'.format(output_path))
    nlp.to_disk(output_path)
    print('Done.') 
