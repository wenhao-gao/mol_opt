""" Some common utility functions """


class CachedFunction:
    """
    Efficient function which caches previously computed values to avoid repeat computation.
    """

    def __init__(self, f: callable, cache: dict = None, transform: callable = None):
        """Init function

        :param f: The function to cache
        :type f: callable
        :param cache: dict mapping known inputs-> outputs of f, defaults to None
        :type cache: dict, optional
        :param transform: optional transform function to apply to values
            of f (e.g. scaling output to be in [0, 1]), defaults to None
        :type transform: callable, optional
        """
        self._f = f
        self._cache = cache
        if self._cache is None:
            self._cache = dict()
        self.transform = transform

    @property
    def cache(self):
        return self._cache

    def _batch_f_eval(self, input_list):
        return [self._f(x) for x in input_list]

    def _batch_transform(self, output_list):
        if self.transform is None:
            return output_list
        else:
            return [self.transform(x) for x in output_list]

    def __call__(self, inputs, batch=False):

        # Ensure it is in batch form
        if not batch:
            inputs = [inputs]

        # Eval function at non-cached inputs
        inputs_not_cached = [x for x in inputs if x not in self.cache]
        outputs_not_cached = self._batch_f_eval(inputs_not_cached)

        # Add new values to cache
        for x, y in zip(inputs_not_cached, outputs_not_cached):
            self._cache[x] = y

        # Get and transform outputs
        outputs = [self._cache[x] for x in inputs]
        outputs = self._batch_transform(outputs)

        # Undo batching if desired
        if not batch:
            assert len(outputs) == 1
            outputs = outputs[0]
        return outputs


class CachedBatchFunction(CachedFunction):
    """
    Special kind of cached function where f takes a batch of inputs
    instead of a single input. The input to the underlying function
    (and transform function) will always be a batch of data.
    """

    def _batch_f_eval(self, input_list):
        return self._f(input_list)
