import io
import threading
import numpy as np

def serialize(array):
    with io.BytesIO() as f:
        np.save(f, array)
        return f.getvalue()

def deserialize(data):
    with io.BytesIO(data) as f:
        return np.load(f)

class AtomicCounter:
    ''' From https://gist.github.com/benhoyt/8c8a8d62debe8e5aa5340373f9c509c7 '''
    def __init__(self, initial=0):
        """Initialize a new atomic counter to given initial value (default 0)."""
        self.value = initial
        self._lock = threading.Lock()

    def increment(self, num=1):
        """Atomically increment the counter by num (default 1) and return the
        new value.
        """
        with self._lock:
            self.value += num
            return self.value

def class_prediction_to_img(y_pred):
    assert len(y_pred.shape) == 3, "Input must have shape (height, width, num_classes)"
    height, width, num_classes = y_pred.shape
    img = np.zeros((height, width, 3), dtype=np.float32)
    img[:, :, 0] = 255 * y_pred[:, :, 0]
    return img

def get_random_string(length):
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    return "".join([alphabet[np.random.randint(0, len(alphabet))] for i in range(length)])
