import numpy as np


def sharpen(input_array: np.ndarray):
	return np.mean(input_array / input_array.std())


def construct_long_short_portfolio(
		predictions: np.ndarray, returns: np.ndarray,
		mask: np.ndarray, weights: np.ndarray = None,
		low: float = 0.1, high: float = 0.1, normalize: bool = True):

	mask_by_columns = np.sum(mask.astype(int), axis=1)
	mask_by_columns_cum_sum = np.cumsum(mask_by_columns)
	predictions_split = np.split(predictions, mask_by_columns_cum_sum)[:-1]
	returns_split = np.split(returns, mask_by_columns_cum_sum)[:-1]

	# value weighted
	to_weight = False
	if weights is not None:
		to_weight = True
		weights_split = np.split(weights, mask_by_columns_cum_sum)[:-1]

	portfolio_returns = list()
	for j in range(mask.shape[0]):
		returns_j = returns_split[j]
		prediction_j = predictions_split[j]
		if to_weight:
			weight_j = weights_split[j]
			return_prediction_weight_j_k = [
				(returns_j[k], prediction_j[k], weight_j[k]) for k in range(mask_by_columns[j])]
		else:
			return_prediction_weight_j_k = [
				(returns_j[k], prediction_j[k], 1) for k in range(mask_by_columns[j])]
		return_prediction_weight_j_k_sorted = sorted(return_prediction_weight_j_k, key=lambda t: t[1])
		n_low = int(low * mask_by_columns[j])
		n_high = int(high * mask_by_columns[j])

		portfolio_return_high = 0.0
		if n_high:
			value_sum_high = 0.0
			for k in range(n_high):
				portfolio_return_high += return_prediction_weight_j_k_sorted[-k-1][0] * return_prediction_weight_j_k_sorted[-k-1][2]
				value_sum_high += return_prediction_weight_j_k_sorted[-k-1][2]
			if normalize:
				portfolio_return_high /= value_sum_high

		portfolio_return_low = 0.0
		if n_low:
			portfolio_return_low = 0.0
			value_sum_low = 0.0
			for k in range(n_low):
				portfolio_return_low += return_prediction_weight_j_k_sorted[k][0] * return_prediction_weight_j_k_sorted[k][2]
				value_sum_low += return_prediction_weight_j_k_sorted[k][2]
			if normalize:
				portfolio_return_low /= value_sum_low

		portfolio_returns.append(portfolio_return_high - portfolio_return_low)
	return np.array(portfolio_returns)