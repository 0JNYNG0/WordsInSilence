import numpy as np


def balanced_schedule(labels, steps, batch_size):
    """Build a deterministic class-balanced index schedule.

    Parameters
    ----------
    labels : array_like
        One-dimensional, nonempty array of integer labels. Not mutated.
    steps : int
        Positive number of rows in the schedule.
    batch_size : int
        Positive number of columns in the schedule.

    Returns
    -------
    numpy.ndarray
        int64 array of shape (steps, batch_size) containing indices into
        ``labels``. Conceptual slot ``j`` (row-major) takes class
        ``classes[j % n_classes]`` where ``classes`` are the sorted unique
        label values, and uses occurrence number ``j // n_classes`` modulo
        that class's group length to select a member index, with members
        kept in original array order.
    """
    for name, value in (("steps", steps), ("batch_size", batch_size)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)
        ):
            raise TypeError(f"{name} must be a positive integer")
        if int(value) <= 0:
            raise ValueError(f"{name} must be a positive integer")
    steps = int(steps)
    batch_size = int(batch_size)

    arr = np.asarray(labels)
    if arr.ndim != 1:
        raise ValueError("labels must be a one-dimensional array")
    if arr.size == 0:
        raise ValueError("labels must be nonempty")
    if arr.dtype == np.bool_ or not np.issubdtype(arr.dtype, np.integer):
        raise TypeError("labels must have an integer dtype")

    classes, inverse = np.unique(arr, return_inverse=True)
    inverse = np.asarray(inverse).reshape(-1)
    n_classes = classes.shape[0]

    # Stable sort groups indices by class while preserving original order.
    order = np.argsort(inverse, kind="stable").astype(np.int64)
    counts = np.bincount(inverse, minlength=n_classes).astype(np.int64)
    starts = np.zeros(n_classes, dtype=np.int64)
    if n_classes > 1:
        starts[1:] = np.cumsum(counts)[:-1]

    total = steps * batch_size
    j = np.arange(total, dtype=np.int64)
    class_idx = j % n_classes
    occurrence = j // n_classes
    member = occurrence % counts[class_idx]
    result = order[starts[class_idx] + member]

    return result.astype(np.int64, copy=False).reshape(steps, batch_size)
