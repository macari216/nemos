import warnings
from numbers import Number
from typing import Any, Callable, Optional, Tuple, Union

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pynapple as nap

from ..base_regressor import BaseRegressor
from ..glm.initialize_parameters import get_inverse_function, non_finite_error
from ..glm.params import GLMParams
from ..inverse_link_function_utils import (
    check_inverse_link_function,
    link_function_from_string,
)
from ..pytrees import FeaturePytree
from ..regularizer import GroupLasso, Regularizer
from ..type_casting import is_numpy_array_like
from ..typing import DESIGN_INPUT_TYPE, UserProvidedParamsT
from . import utils
from .params import PPGLMParamsWithKey, X_ppglm, y_ppglm
from .validation import (
    PopulationPPGLMValidator,
    PPGLMValidator,
    to_pp_glm_params_with_key,
)


class MockPPGLM(BaseRegressor):
    """
    Minimal PP-GLM stand-in for testing PPGLMValidator logic.

    Parameters
    ----------
    history_window_dur :
        Duration of the spike history window in seconds. Must be > 0.
    n_basis_funcs :
        Number of basis functions for the spike history filter.
    eval_function :
        Function for evaluating basis functions at a given temporal delay.
    recording_time :
        Optional time support object defining valid recording epochs used for
        Monte Carlo sampling. If None, inferred from data during preprocessing.
        If provided, always overrides the time support of the input data.
    inverse_link_function :
        A function that maps the linear combination of predictors into a firing rate. The default is
        ``jnp.exp`` for the Poisson point process model.
    n_mc_samples :
        Number of Monte Carlo samples for compensator estimation.
    scan_size :
        Number of parallel scans to run during log-likelihood computation.
    random_key :
        random seed for Monte Carlo sampling.
    """

    def __init__(
        self,
        history_window_dur: float,
        n_basis_funcs: int,
        eval_function: Callable,
        n_mc_samples: int,
        recording_time: Optional[nap.IntervalSet] = None,
        inverse_link_function: Optional[Callable] = jnp.exp,
        scan_size: int = 1,
        random_key=jax.random.PRNGKey(0),
        regularizer: Optional[Union[str, Regularizer]] = None,
        regularizer_strength: Any = None,
        solver_name: str = None,
        solver_kwargs: dict = None,
    ):
        super().__init__(
            regularizer=regularizer,
            regularizer_strength=regularizer_strength,
            solver_name=solver_name,
            solver_kwargs=solver_kwargs,
        )

        self.history_window_dur = history_window_dur
        self.n_basis_funcs = n_basis_funcs

        # validator is a frozen dataclass — constructed explicitly with n_basis_funcs
        self._validator = PPGLMValidator(n_basis_funcs=self.n_basis_funcs)

        self.eval_function = eval_function
        self.recording_time = recording_time
        self.inverse_link_function = inverse_link_function
        self.n_mc_samples = n_mc_samples
        self.scan_size = scan_size
        self.random_key = random_key

        self._max_window = None
        self._mc_grid = None
        self.coef_: Optional[jnp.ndarray] = None
        self.intercept_: Optional[jnp.ndarray] = None

    @property
    def history_window_dur(self) -> float:
        return self._history_window_dur

    @history_window_dur.setter
    def history_window_dur(self, history_window_dur: float):
        if not isinstance(history_window_dur, Number) or history_window_dur <= 0:
            raise ValueError(
                f"`history_window_dur` must be a strictly positive float. Got {history_window_dur!r}."
            )
        self._history_window_dur = float(history_window_dur)

    @property
    def n_basis_funcs(self) -> int:
        return self._n_basis_funcs

    @n_basis_funcs.setter
    def n_basis_funcs(self, n_basis_funcs: int):
        if not isinstance(n_basis_funcs, Number) or n_basis_funcs < 1:
            raise ValueError(
                f"`n_basis_funcs` must be a positive integer. Got {n_basis_funcs!r}."
            )
        self._n_basis_funcs = int(n_basis_funcs)

    @property
    def eval_function(self) -> Callable:
        return self._eval_function

    @eval_function.setter
    def eval_function(self, eval_function):
        if not callable(eval_function):
            raise ValueError(
                f"`eval_function` must be a callable. Got {type(eval_function)!r}."
            )
        # probe output shape with a dummy input
        try:
            dummy = jnp.zeros(3)
            out = eval_function(dummy)
            out = jnp.asarray(out)
        except Exception as e:
            raise ValueError(
                f"`eval_function` must accept a single array argument. "
                f"Calling it with a dummy input raised: {e}"
            ) from e
        if out.ndim != 2 or out.shape[0] != 3:
            raise ValueError(
                f"`eval_function` must return an array of shape (n_eval_pts, n_basis_funcs). "
                f"Got shape {out.shape!r} for 3 evaluation points."
            )
        if hasattr(self, "_n_basis_funcs") and out.shape[1] != self._n_basis_funcs:
            raise ValueError(
                f"`eval_function` output has {out.shape[1]} basis functions, "
                f"but `n_basis_funcs` is {self._n_basis_funcs}."
            )
        self._eval_function = eval_function

    @property
    def recording_time(self) -> Optional[nap.IntervalSet]:
        return self._recording_time

    @recording_time.setter
    def recording_time(self, recording_time):
        """Setter for the recording time interval."""
        if recording_time is None:
            self._recording_time = None
        if isinstance(recording_time, nap.IntervalSet):
            self._recording_time = recording_time
        elif (
            isinstance(recording_time, (tuple, list, np.ndarray))
            and len(recording_time) == 2
        ):
            starts, ends = recording_time
            self._recording_time = nap.IntervalSet(
                start=np.asarray(starts),
                end=np.asarray(ends),
            )
        else:
            raise TypeError(
                f"recording_time must be a pynapple.IntervalSet or a tuple/list of "
                f"(start, end) arrays, got {type(recording_time)} of length {len(recording_time)}."
            )

    @property
    def inverse_link_function(self):
        """Inverse link function mapping the linear predictor to the response space.

        Always a callable. If ``None`` was passed at construction time, this is
        resolved to the observation model's default ``jnp.exp``.
        """
        return self._inverse_link_function

    @inverse_link_function.setter
    def inverse_link_function(self, inverse_link_function: Callable):
        """Validate and set the inverse link function.

        Parameters
        ----------
        inverse_link_function :
            One of:
            - ``None`` — use the observation model's default inverse link.
            - ``str`` — name of a built-in (e.g. ``"exp"``, ``"softplus"``; resolved by
              :func:`nemos.inverse_link_function_utils.resolve_inverse_link_function`.
            - ``Callable`` — a custom function. Must be JAX-traceable
              (differentiable) and return a ``jax.numpy.ndarray`` or scalar
              when called on a JAX array.

        Raises
        ------
        TypeError
            If the value is neither callable nor a string.
        ValueError
            If a callable is non-differentiable or returns an unsupported type.
        """
        if inverse_link_function is None:
            self._inverse_link_function = jnp.exp
        elif isinstance(inverse_link_function, str):
            self._inverse_link_function = link_function_from_string(
                inverse_link_function
            )
        else:
            check_inverse_link_function(inverse_link_function)
            self._inverse_link_function = inverse_link_function

    @property
    def n_mc_samples(self) -> int:
        return self._n_mc_samples

    @n_mc_samples.setter
    def n_mc_samples(self, n_mc_samples: int):
        if not isinstance(n_mc_samples, Number) or n_mc_samples < 1:
            raise ValueError(
                f"`n_mc_samples` must be a positive integer. Got {n_mc_samples!r}."
            )
        self._n_mc_samples = int(n_mc_samples)

    @property
    def scan_size(self) -> int:
        return self._scan_size

    @scan_size.setter
    def scan_size(self, scan_size: int):
        if not isinstance(scan_size, Number) or scan_size < 1:
            raise ValueError(
                f"`scan_size` must be a positive integer. Got {scan_size!r}."
            )
        self._scan_size = int(scan_size)

    @property
    def random_key(self):
        """Getter for the random seed."""
        return self._random_key.astype(jnp.uint32)

    @random_key.setter
    def random_key(self, value):
        """Setter for the random seed. Accepts a jax PRNGKey or a float seed."""
        if isinstance(value, int):
            key = jax.random.PRNGKey(value)
        else:
            key = jnp.asarray(value)
            self._validator.validate_random_key(key, dtype=jnp.uint32)
        self._random_key = key.astype(jnp.float64)

    def _get_model_params(self):
        """Pack coef_ and intercept_  into a params pytree.

        This method should be overwritten in case the parameter structure changes,
        or if new regression models will have a different parameter structure.
        """
        # Retrieve parameter tree
        return GLMParams(self.coef_, self.intercept_)

    def _set_model_params(self, params: GLMParams):
        """Unpack and store params pytree to coef_ and intercept_.

        This method should be overwritten in case the parameter structure changes,
        or if new regression models will have a different parameter structure.
        """
        # Store parameters
        self.coef_: DESIGN_INPUT_TYPE = params.coef
        self.intercept_: jnp.ndarray = params.intercept

    def _check_is_fit(self):
        missing = [
            name
            for name, val in (("coef_", self.coef_), ("intercept_", self.intercept_))
            if val is None
        ]
        if missing:
            raise ValueError(
                f"This {type(self).__name__} instance is not fitted yet. "
                f"Missing attributes: {missing}. Call fit() first."
            )

    def _initialize_intercept_matching_mean_rate(
        self,
        inverse_link_function: Callable,
        y: y_ppglm,
    ) -> jnp.ndarray:

        analytical_inv = get_inverse_function(inverse_link_function)

        means = (
            jnp.unique(y[1], return_counts=True)[1] / self.recording_time.tot_length()
        )

        if analytical_inv:
            out = analytical_inv(means)
            if jnp.any(jnp.isnan(out)):
                raise ValueError(
                    "Failed to initialize the model intercept as the inverse of the firing rate for "
                    "the provided link function. The mean firing rate has some non-positive values."
                )
            if jnp.any(~jnp.isfinite(out)):
                raise non_finite_error

            return out

    def _model_specific_initialization(
        self,
        X: X_ppglm,
        y: y_ppglm,
        **kwargs,
    ) -> GLMParams:

        empty_params = self._validator.get_empty_params(X, y)

        initial_intercept = self._initialize_intercept_matching_mean_rate(
            self._inverse_link_function, y
        )

        n_features = int(jnp.unique(X.ids).size * self.n_basis_funcs)
        initial_coef = jax.tree_util.tree_map(
            lambda x: jnp.zeros(n_features), empty_params.coef
        )

        init_params = eqx.tree_at(
            lambda p: (p.coef, p.intercept),
            empty_params,
            (initial_coef, initial_intercept),
        )

        self._validator.feature_mask_consistency(
            getattr(self, "_feature_mask", None), init_params
        )
        return init_params

    def initialize_params(
        self,
        X: DESIGN_INPUT_TYPE,
        y: jnp.ndarray,
    ) -> UserProvidedParamsT:
        """Initialize model parameters.

        Initialize coefficients with zeros and intercept by matching the mean firing rate.

        Parameters
        ----------
        X
            Input data, array of shape ``(n_time_bins, n_features)`` or pytree of same.
        y
            Target data, array of shape ``(n_time_bins,)`` for single neuron models or
            ``(n_time_bins, n_neurons)`` for population models.

        Returns
        -------
        params
            Initial parameter tuple of (coefficients, intercept).
        """
        X, y = self._preprocess_inputs(X, y)
        init_params = self._model_specific_initialization(X, y)
        return self._validator.from_model_params(init_params)

    # ------------------------------------------------------------------
    # PP-GLM-specific stubs
    # ------------------------------------------------------------------

    def _preprocess_inputs(
        self,
        X: Any,
        y: Optional[Any] = None,
        *args,
    ) -> Tuple[X_ppglm, y_ppglm]:
        """
        Preprocess spike timestamp inputs for fitting.

        Parameters
        ----------
        X : array-like or pynapple.Ts or pynapple.TsGroup or dict
            Event timestamps for the model predictors. Accepted types:

            - ``np.ndarray`` or ``list`` of shape ``(n_events,)`` — single predictor
            - ``list`` of ``np.ndarray`` / ``list``, length ``n_neurons``, each of
              shape ``(n_events_i,)`` — multiple predictors, ragged
            - ``dict`` mapping neuron id → ``np.ndarray`` or ``list`` of shape
              ``(n_events_i,)`` — multiple predictors with explicit ids
            - ``pynapple.Ts`` — single predictor
            - ``pynapple.TsGroup`` — multiple predictors

        y : array-like or pynapple.Ts or pynapple.TsGroup or dict
            Spike timestamps for the postsynaptic neuron(s). Same accepted types
            as `X`.

        Returns
        -------
        X : X_ppglm
            Preprocessed predictors with fields ``times`` (event timestamps) and
            ``ids`` (predictor neuron indices).
        y : y_ppglm
            Preprocessed spikes with fields ``times`` (spike timestamps), ``ids``
            (postsynaptic neuron indices), and ``idx`` (indices into event times).
        """

        #
        if self._recording_time is not None and (
            hasattr(X, "time_support") or hasattr(y, "time_support")
        ):
            warnings.warn(
                "recording_time was provided alongside pynapple inputs that carry their own "
                "time support. The data's time support will be overridden by recording_time."
            )

        # convert inputs to TsGroups and flatten into marked time series
        X = utils.to_tsgroup(X).to_tsd()
        y = utils.to_tsgroup(y).to_tsd()

        # validate  / set recording time
        if self._recording_time is None:
            self._recording_time = X.time_support.intersect(y.time_support)
            if len(self._recording_time) == 0:
                raise ValueError(
                    "X and y have no overlapping time support. "
                    "Ensure that predictor events and spikes occur within the same recording epochs."
                )
        else:
            self._validator.validate_span(X, y, self._recording_time)

        # define MC sampling grid nased on the recording time
        if self._mc_grid is None:
            self._mc_grid = utils.build_mc_sampling_grid(
                self._recording_time, self.n_mc_samples
            )

        # find indices of y into X
        event_times = jnp.asarray(X.t)
        spike_times = jnp.asarray(y.t)
        y_idx = jnp.searchsorted(event_times, spike_times)

        X = X_ppglm(
            times=event_times,
            ids=jnp.asarray(X.d, dtype=int),
        )

        y = y_ppglm(
            times=spike_times,
            ids=jnp.asarray(y.d, dtype=int),
            idx=y_idx,
        )

        # compute and set max history window
        if self._max_window is None:
            self._max_window = int(
                jnp.maximum(
                    utils.compute_max_window_size(
                        jnp.array([-self._history_window_dur, 0]),
                        spike_times,
                        event_times,
                    ),
                    utils.compute_max_window_size(
                        jnp.array([-self._history_window_dur, 0]),
                        event_times,
                        event_times,
                    ),
                )
            )

        # add max window padding
        X, y = utils.adjust_indices_and_spike_times(
            X, self._history_window_dur, self._max_window, y
        )

        if isinstance(self.regularizer, GroupLasso):
            if self.regularizer.mask is None and is_numpy_array_like(X)[1]:
                warnings.warn(
                    "Mask has not been set. Defaulting to a single group for all parameters. "
                    "Please see the documentation on GroupLasso regularization for defining a mask."
                )
            elif self.regularizer.mask is not None:
                self._wrap_grouplasso_mask(X, y)

        return X, y

    def fit(self, X: Any, y: Any, init_params=None):
        """Validate inputs and set zero params; no actual optimization."""
        self._validator.validate_inputs(X=X, y=y)

        X, y = self._preprocess_inputs(X, y)

        if init_params is not None:
            params = self._validator.validate_and_cast_params(init_params)
        else:
            params = self._model_specific_initialization(X, y)

        self._validator.validate_consistency(params, X=X, y=y)

        self._validator.validate_random_key(self._random_key, dtype=jnp.float64)
        params_with_key = to_pp_glm_params_with_key(params, self._random_key)

        self._set_model_params(params)
        return self

    def predict(self, X):
        self._check_is_fit()
        return jnp.array(0.0)

    def score(self, X, y, **kwargs):
        self._check_is_fit()
        return jnp.array(0.0)

    def simulate(self, *args, **kwargs):
        pass

    def save_params(self, *args, **kwargs):
        pass

    def update(self, *args, **kwargs):
        pass

    def _initialize_optimizer_and_state(self, *args, **kwargs):
        pass

    def _compute_loss(
        self,
        params: PPGLMParamsWithKey,
        X: X_ppglm,
        y: y_ppglm,
        *args,
        **kwargs,
    ) -> jnp.ndarray:
        return jnp.array(0.0)

    def _get_optimal_solver_params_config(self, *args, **kwargs):
        pass


class MockPopulationPPGLM(MockPPGLM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._validator = PopulationPPGLMValidator(n_basis_funcs=self.n_basis_funcs)

    def _model_specific_initialization(
        self,
        X: X_ppglm,
        y: y_ppglm,
        **kwargs,
    ) -> GLMParams:

        empty_params = self._validator.get_empty_params(X, y)

        initial_intercept = self._initialize_intercept_matching_mean_rate(
            self._inverse_link_function, y
        )

        n_features = int(jnp.unique(X.ids).size * self.n_basis_funcs)
        n_neurons = len(jnp.unique(y.ids))
        initial_coef = jax.tree_util.tree_map(
            lambda x: jnp.zeros((n_features, n_neurons)), empty_params.coef
        )

        init_params = eqx.tree_at(
            lambda p: (p.coef, p.intercept),
            empty_params,
            (initial_coef, initial_intercept),
        )

        self._validator.feature_mask_consistency(
            getattr(self, "_feature_mask", None), init_params
        )
        return init_params
