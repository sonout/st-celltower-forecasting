"""Baselines sharing the sequence input/output contract used by Graph WaveNet.

These implementations forecast every test window, using only its past inputs.
"""

import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from .base_model import BaseModel


class SequenceBaseline(BaseModel):
    def _target_inputs(self, X):
        indices = [self.config['feature_columns'].index(k) for k in self.config['target_columns']]
        return X[..., indices]

    def _timestamps(self, split, count):
        timestamps = self.config.get(f'timestamps_{split}')
        if timestamps is None or len(timestamps) != count:
            raise ValueError(f'{self.name} needs one timestamps_{split} entry per sequence.')
        return timestamps

    def _history(self, X, Y, split='train'):
        history = {}
        delta = pd.Timedelta(minutes=self.config.get('granularity_min', 15))
        inputs = self._target_inputs(X)
        for i, (start, origin, _) in enumerate(self._timestamps(split, len(X))):
            for j, values in enumerate(inputs[i]):
                history[pd.Timestamp(start) + j * delta] = values.copy()
            for j, values in enumerate(Y[i]):
                history[pd.Timestamp(origin) + (j + 1) * delta] = values.copy()
        return history


class HistoricalPatternModel(SequenceBaseline):
    """Mean training observation for each cell, weekday, and time of day."""

    def _slot(self, stamp):
        return stamp.dayofweek, (stamp.hour * 60 + stamp.minute) // self.config.get('granularity_min', 15)

    def train(self, X_train, Y_train, X_val=None, Y_val=None, static_features=None):
        history = self._history(X_train, Y_train)
        groups = {}
        for stamp, values in history.items():
            groups.setdefault(self._slot(stamp), []).append(values)
        self.patterns = {key: np.mean(values, axis=0) for key, values in groups.items()}
        self.fallback = np.mean(list(history.values()), axis=0)
        self.is_fitted = True

    def predict(self, X_test, static_features=None):
        super().predict(X_test, static_features)
        delta = pd.Timedelta(minutes=self.config.get('granularity_min', 15))
        return np.array([
            [self.patterns.get(self._slot(pd.Timestamp(origin) + step * delta), self.fallback)
             for step in range(1, self.output_steps + 1)]
            for _, origin, _ in self._timestamps('test', len(X_test))
        ], dtype=np.float32)


class SeasonalNaiveModel(SequenceBaseline):
    """Repeat the last observed season; fall back to persistence if unavailable."""

    def train(self, X_train, Y_train, X_val=None, Y_val=None, static_features=None):
        self.history = self._history(X_train, Y_train)
        if X_val is not None and Y_val is not None:
            # Validation observations are already in the past when the test period starts.
            self.history.update(self._history(X_val, Y_val, split='val'))
        self.period = int(self.config.get('model_params', {}).get('seasonal_period', 96))
        if self.period < 1:
            raise ValueError('seasonal_period must be positive.')
        self.is_fitted = True

    def predict(self, X_test, static_features=None):
        super().predict(X_test, static_features)
        delta = pd.Timedelta(minutes=self.config.get('granularity_min', 15))
        history = dict(self.history)
        predictions = []
        for values, (start, origin, _) in zip(self._target_inputs(X_test), self._timestamps('test', len(X_test))):
            for j, observed in enumerate(values):
                history[pd.Timestamp(start) + j * delta] = observed
            forecast = []
            for step in range(1, self.output_steps + 1):
                lookup = pd.Timestamp(origin) + (((step - 1) % self.period) + 1 - self.period) * delta
                forecast.append(history.get(lookup, values[-1]))
            predictions.append(forecast)
        return np.asarray(predictions, dtype=np.float32)


class TreeSequenceModel(SequenceBaseline):
    """Learn direct forecasts from flattened history, pooling samples across cells."""

    def _features(self, X, static_features):
        features = X.transpose(0, 2, 1, 3).reshape(len(X) * X.shape[2], -1)
        if static_features is not None:
            static = np.tile(np.asarray(static_features, dtype=np.float32), (len(X), 1))
            features = np.concatenate([features, static], axis=1)
        return features

    def _targets(self, Y):
        return Y.transpose(0, 2, 1, 3).reshape(len(Y) * Y.shape[2], -1)

    def predict(self, X_test, static_features=None):
        super().predict(X_test, static_features)
        values = self._predict_flat(self._features(X_test, static_features))
        return np.asarray(values).reshape(len(X_test), X_test.shape[2], self.output_steps, self.n_targets).transpose(0, 2, 1, 3)


class RandomForestModel(TreeSequenceModel):
    def train(self, X_train, Y_train, X_val=None, Y_val=None, static_features=None):
        params = dict(self.config.get('model_params', {}))
        params.setdefault('random_state', self.config.get('random_seed', 42))
        params.setdefault('n_jobs', 1)
        self.estimator = RandomForestRegressor(**params)
        self.estimator.fit(self._features(X_train, static_features), self._targets(Y_train))
        self.is_fitted = True

    def _predict_flat(self, features):
        return self.estimator.predict(features)


class XGBoostModel(TreeSequenceModel):
    def train(self, X_train, Y_train, X_val=None, Y_val=None, static_features=None):
        from xgboost import XGBRegressor
        params = dict(self.config.get('model_params', {}))
        params.setdefault('random_state', self.config.get('random_seed', 42))
        params.setdefault('n_jobs', 1)
        params.setdefault('tree_method', 'hist')
        features, targets = self._features(X_train, static_features), self._targets(Y_train)
        validation = X_val is not None and Y_val is not None
        if validation:
            val_features, val_targets = self._features(X_val, static_features), self._targets(Y_val)
        else:
            params.pop('early_stopping_rounds', None)
        self.estimators = []
        for column in range(targets.shape[1]):
            estimator = XGBRegressor(**params)
            kwargs = {'eval_set': [(val_features, val_targets[:, column])], 'verbose': False} if validation else {}
            estimator.fit(features, targets[:, column], **kwargs)
            self.estimators.append(estimator)
        self.is_fitted = True

    def _predict_flat(self, features):
        return np.column_stack([estimator.predict(features) for estimator in self.estimators])


class ARIMAModel(SequenceBaseline):
    """Fit training-series parameters, then condition on each test window's past."""

    def train(self, X_train, Y_train, X_val=None, Y_val=None, static_features=None):
        from statsmodels.tsa.arima.model import ARIMA
        params = self.config.get('model_params', {})
        order = tuple(params.get(k, default) for k, default in [('p', 1), ('d', 1), ('q', 0)])
        seasonal = tuple(params.get('seasonal_order', (0, 0, 0, 0))) if params.get('seasonal', False) else (0, 0, 0, 0)
        history = self._history(X_train, Y_train)
        series = np.array([history[t] for t in sorted(history)])
        self.fitted = {}
        for cell in range(series.shape[1]):
            for target in range(self.n_targets):
                try:
                    self.fitted[cell, target] = ARIMA(
                        series[:, cell, target], order=order, seasonal_order=seasonal,
                        enforce_stationarity=False, enforce_invertibility=False,
                    ).fit()
                except (ValueError, np.linalg.LinAlgError) as error:
                    warnings.warn(f'ARIMA cell {cell}, target {target} uses persistence: {error}')
        self.is_fitted = True

    def predict(self, X_test, static_features=None):
        super().predict(X_test, static_features)
        inputs = self._target_inputs(X_test)
        result = np.repeat(inputs[:, -1:, :, :], self.output_steps, axis=1).copy()
        for (cell, target), fitted in self.fitted.items():
            for sample in range(len(X_test)):
                try:
                    conditioned = fitted.apply(inputs[sample, :, cell, target], refit=False)
                    forecast = np.asarray(conditioned.forecast(self.output_steps))
                    if np.isfinite(forecast).all():
                        result[sample, :, cell, target] = forecast
                except (ValueError, np.linalg.LinAlgError) as error:
                    warnings.warn(f'ARIMA sample {sample} uses persistence: {error}')
        return result
