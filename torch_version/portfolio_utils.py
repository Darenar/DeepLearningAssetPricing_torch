import numpy as np

from .data_loader import FinanceDataset


def sharpe(input_array: np.ndarray) -> float:
	return float(np.mean(input_array / input_array.std()))


def construct_long_short_portfolio(
		predictions: np.ndarray, returns: np.ndarray,
		mask: np.ndarray, weights: np.ndarray = None,
		low: float = 0.1, high: float = 0.1, normalize: bool = True):

	mask_by_columns = np.sum(mask.astype(int), axis=1)
	mask_by_columns_cum_sum = np.cumsum(mask_by_columns)
	predictions_split = np.split(predictions, mask_by_columns_cum_sum)[:-1]
	returns_split = np.split(returns, mask_by_columns_cum_sum)[:-1]

	# value weighted
	weights_split = None
	if weights is not None:
		weights_split = np.split(weights, mask_by_columns_cum_sum)[:-1]

	portfolio_returns = list()
	for j in range(mask.shape[0]):
		returns_j = returns_split[j]
		prediction_j = predictions_split[j]
		if weights_split is not None:
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


def decompose_returns(predicted_returns: np.ndarray, dataset: FinanceDataset):
	returns_array = dataset.individual_data.return_array
	mask_array = dataset.individual_data.mask

	returns_masked = returns_array[mask_array]
	splits = np.sum(mask_array, axis=1).cumsum()[:-1]
	pred_return_split = np.split(predicted_returns, splits)
	true_return_split = np.split(returns_masked, splits)

	predicted_returns_hat_list = list()
	residual_list = list()
	for true_r, pred_r in zip(true_return_split, pred_return_split):
		pred_r_hat_i = pred_r.dot(true_r) / pred_r.dot(pred_r) * pred_r
		res_r_i = true_r - pred_r_hat_i
		predicted_returns_hat_list.append(pred_r_hat_i)
		residual_list.append(res_r_i)

	returns_hat = np.zeros_like(mask_array, dtype=float)
	residuals = np.zeros_like(mask_array, dtype=float)
	returns_hat[mask_array] = np.concatenate(predicted_returns_hat_list)
	residuals[mask_array] = np.concatenate(residual_list)
	return returns_hat, residuals, mask_array, returns_array


def calculate_ev(residuals: np.ndarray, mask: np.ndarray, returns: np.ndarray):
	num_firm_per_time = np.sum(mask, axis=1)
	numerator = np.mean(np.square(residuals).sum(axis=1) / num_firm_per_time)
	denominator = np.mean(
		np.square(returns * mask).sum(axis=1) / num_firm_per_time)
	return 1 - numerator / denominator


def calculate_xs_r_squared(residuals: np.ndarray, mask: np.ndarray, returns: np.ndarray, weighted: bool = False):
	num_time_per_firm = np.sum(mask, axis=0)
	numerator = np.square(residuals.sum(axis=0) / num_time_per_firm)
	if weighted:
		numerator *= num_time_per_firm
	numerator = np.mean(numerator)
	denominator = np.square((returns * mask).sum(axis=0) / num_time_per_firm)
	if weighted:
		denominator *= num_time_per_firm
	denominator = np.mean(denominator)

	return 1 - numerator/denominator


def calculate_statistics(predicted_returns: np.ndarray, dataset: FinanceDataset):
	returns_hat, residuals, mask, returns = decompose_returns(predicted_returns, dataset)
	ev_stats = calculate_ev(residuals, mask, returns)
	xs_r_squared = calculate_xs_r_squared(residuals, mask, returns)
	xs_r_squared_weighted = calculate_xs_r_squared(residuals, mask, returns, weighted=True)
	return ev_stats, xs_r_squared, xs_r_squared_weighted
