import aerosandbox.numpy as np
from typing import Union, List, Tuple


def get_index_of_unique(array: Union[np.ndarray, List, Tuple]):
    if len(np.unique(array)) == 1:
        return 0
    else:
        return np.argmax(
            np.abs(
                array - np.mean(array)
            )
        )
