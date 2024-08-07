import faiss
import numpy as np
from ltau_ff.utils import logs_to_pdfs#, logs_to_uncertainties


class UQEstimator:
    """A class for predicting a model's error PDFs by performing (weighted)
    averaging of PDFs computed on the nearest neighbors in descriptor
    space for points from the model's training set."""

    def __init__(
            self,
            pdfs,
            bins=None,
            descriptors=None,
            from_error_logs=False,
            nbins=None,
            range_limits=None,
            bin_spacing='log',
            index_type='IndexFlatL2',
            index_args=None,
            load_index=False,
            index_load_path=None,
            **kwargs,
            ):
        """
        Args:
            pdfs (np.ndarray): an array of size (N, N_b) containing the
                PDFs for all N training points defined over N_b bins. If `from_error_logs` is
                True, then `pdfs` should be a list of M arrays, where
                each array is a size (T_m, N) array of errors logged for each
                training point for an ensemble of M models. T_m is the number of
                training epochs for the m-th model in the ensemble.

            bins (np.ndarray): array of bin edges. If `from_error_logs` is None,
            must provide `nbins`, `range_limits`, and `bin_spacing`.

            descriptors (np.ndarray): an array of size (N, D), where D is the
                dimensionality of the descriptor space. Required if `load_index`
                is False

            from_error_logs (bool): If True, `pdfs` should be computed
                from an ensemble of error trajectories. Default is False.

            nbins (int): the number of bins to use for computing the PDFs. Only
                required if `from_error_logs` is True.

            range_limits (tuple[float]): the upper/lower limits of the bins. If
                not provided, uses the min/max error from `pdfs`. Only required
                if `from_error_logs` is True. 

            bin_spacing (str): one of "log" or "linear". Default is "log". Only
                required if `from_error_logs` is True.

            index_type: (str), One of ['IndexFlatL2', 'HNSW+IVFPQ']. Default is
                'IndexFlatL2'

            index_args (dict): additional arguments to be passed to the index
                constructor

            load_index (bool): if True, loads a saved indexer from `index_load_path`

            index_load_path (str): path to saved indexer
        """

        if from_error_logs:
            pdfs, bins = logs_to_pdfs(
                pdfs, bins, nbins, range_limits, bin_spacing
                )
            # uncertainties = logs_to_uncertainties(pdfs)

        if bins is None:
            raise RuntimeError("Must specify bin arguments if `bins` is None")

        self._pdfs = pdfs
        self._bins = bins
        self.bin_spacing = bin_spacing

        # Compute bin widths and midpoints
        # TODO: assumes bins are log-spaced? may work for linear too though
        log_midpoints = (np.log10(bins[:-1]) + np.log10(bins[1:])) / 2
        self._bin_midpoints = 10**log_midpoints
        self._bin_widths = np.diff(self._bins)

        if load_index:
            self.index = faiss.read_index(index_load_path)
        else:
            self._build_index(descriptors, index_type, index_args)


    def _build_index(self, descriptors, index_type, index_args):
        """Constructs the index used by FAISS for similarity search.

        Args:
            descriptors (np.ndarray): an array of size (N, D), where D is the
                dimensionality of the descriptor space.
            index_type: (str), One of ['IndexFlatL2', 'HNSW+IVFPQ'].
                default='IndexFlatL2'
            index_args: (dict), additional arguments to be passed to the index
                constructor
        """

        assert descriptors.shape[0] == self._pdfs.shape[0], f"Descriptors shape {descriptors.shape} does not match PDFs shape {self._pdfs.shape}"

        if index_type == 'IndexFlatL2':
            self.index = faiss.IndexFlatL2(descriptors.shape[1])
            self.index.add(descriptors)
        elif index_type == 'IndexHNSWFlat':
            for k in ['M', 'efConstruction', 'efSearch']:
                if k not in index_args:
                    raise RuntimeError(f'Key "{k}" missing from `index_args`')

            self.index = faiss.IndexHNSWFlat(
                descriptors.shape[1],
                int(index_args['M'])
            )

            self.index.hnsw.efConstruction = int(index_args['efConstruction'])
            self.index.add(descriptors)
            self.index.hnsw.efSearch = int(index_args['efSearch'])

        elif index_type == 'HNSW+IVFPQ':
            for k in ['nlist', 'M', 'nbits', 'hnsw_m', 'efConstruction', 'efSearch']:
                if k not in index_args:
                    raise RuntimeError(f'Key "{k}" missing from `index_args`')

            quantizer = faiss.IndexHNSWFlat(
                descriptors.shape[1],
                int(index_args['hnsw_m'])
            )

            self.index = faiss.IndexIVFPQ(
                quantizer,
                descriptors.shape[1],
                int(index_args['nlist']),
                int(index_args['M']),
                int(index_args['nbits']),
            )

            self.index.hnsw.efConstruction = index_args['efConstruction']
            self.index.train(descriptors)
            self.index.add(descriptors)
        else:
            raise NotImplementedError(f"Indexer '{index_type}' is not supported yet.")


    def __call__(self, query_descriptors, topk, atol=None, norm=False):
        """Returns the predicted PDF, averaged over each sample from
        `query_descriptors` using the `topk` nearest neighbors.

        Args:

            query_descriptors: np.ndarray
                (N, D) array of descriptors for N queries and descriptor size D

            topk: int
                Number of neighbors to average over

            atol: float or None
                If `atol` is not None, instead returns the CDFs for each atom
                evaluated at the specified atol. In other words, the likelihood
                that the prediction will have an error below atol.

            norm: bool
                If True, normalizes the PDFs so that they integrate to a value
                of 1. Default is False, since PDFs are assumed to already have
                been normalized.
        """ 
        I = self.get_neighbor_indices(query_descriptors, k=topk)

        p = self._pdfs[I]   # (nsamples, topk, nbins)
        p = p.mean(axis=1)  # (nsamples, nbins)

        if norm:
            p /= self._bin_widths[None, :]
            p /= np.sum(p*self._bin_widths[None, :], axis=1)[:, None]

        # NOTE: need to check which axis to avg over if XYZ coords included
        if atol is None:
            return p  # (nsamples, nbins)
        else:
            # p shape: (nsamples, nbins)
            x = np.where(self._bins >= atol)[0][0]
            return np.cumsum(p, axis=1)[:, x]  # CDF evaluated at atol

    def predict_errors(self, query_descriptors, topk):
        pdfs = self(query_descriptors, topk)

        # Getting expectation values for log-scaled bins
        if self.bin_spacing == 'log':
            log_midpoints = (np.log10(self._bins[:-1]) + np.log10(self._bins[1:])) / 2
        else:
            log_midpoints = self._bins[:-1] - self._bins[1:]

        midpoints = 10**log_midpoints
        bin_widths = np.diff(self._bins)

        return (pdfs*bin_widths*midpoints).sum(axis=-1)


    def get_neighbor_indices(self, query_descriptors, k=1):
        _, I = self.index.search(query_descriptors, k=k)
        return I


    def save(self, filename):
        faiss.write_index(self.index, filename)
