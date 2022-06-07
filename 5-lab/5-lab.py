from PIL import Image
import numpy as np
import hebb


def parse_image(filename):
  img = Image.open(filename, 'r').convert('L').getdata()
  img_arr = np.array(img)
  return np.where(img_arr == 255, -1, 1)


LETTERS_DIR = 'letters/'

IMGS = dict(
  a_valid=parse_image(LETTERS_DIR + 'a_valid.bmp'),
  a_invalid=parse_image(LETTERS_DIR + 'a_invalid.bmp'),
  n_valid=parse_image(LETTERS_DIR + 'n_valid.bmp'),
  n_invalid=parse_image(LETTERS_DIR + 'n_invalid.bmp'),
  i_valid=parse_image(LETTERS_DIR + 'i_valid.bmp'),
  i_invalid=parse_image(LETTERS_DIR + 'i_invalid.bmp'),
  r_valid=parse_image(LETTERS_DIR + 'r_valid.bmp'),
  r_invalid=parse_image(LETTERS_DIR + 'r_invalid.bmp'),
)

TRAINED = dict(
  a=hebb.new_entity_container(image=IMGS['a_valid']),
  n=hebb.new_entity_container(image=IMGS['n_valid']),
  i=hebb.new_entity_container(image=IMGS['i_valid']),
  r=hebb.new_entity_container(image=IMGS['r_valid']),
)

for letter in TRAINED:
  hebb.train(TRAINED[letter])

for key in IMGS:
  letters = hebb.identify(image=IMGS[key], trained=TRAINED)
  print(f'{key} (weights):')
  for letter in letters:
    print(f'\t{letter} = {letters[letter]}')
  print()
