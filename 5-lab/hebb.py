from dataclasses import dataclass
import numpy as np


@dataclass
class ImageEntity:
  array: np.ndarray
  weights: np.array
  input_signal: int = 1


__sum_output_signal = lambda x, weights: np.sum(x * weights) + weights[0]

def new_entity_container(image, input_signal=1):
  x0 = 1
  return ImageEntity(
    array=np.concatenate(([x0], image)),
    weights=np.zeros(len(image) + 1),
    input_signal=input_signal
  )


def train(container: ImageEntity, learning_rate=1):
  i = 0
  while True:
    i += 1
    container.weights = container.weights * learning_rate + container.array * container.input_signal
    s = __sum_output_signal(container.array, container.weights)
    breakpoint = s < 0 if container.input_signal == -1 else s > 0
    if breakpoint: return i


def identify(image, trained):
  image = np.concatenate(([1], image))
  result = dict()
  for letter in trained:
    s = __sum_output_signal(image, trained[letter].weights)
    result[letter] = s
  return dict(sorted(result.items(), key=lambda item: item[1], reverse=True))
