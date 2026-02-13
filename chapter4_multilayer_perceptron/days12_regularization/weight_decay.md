# weight decay(aka. L2 regularization)

the core target:
reduce model complexity by adding an L2 penalty term (the sum of squared weights/2) to the loss function. This penalty restricts the overall magnitude(大小) of all weight parameters, forcing them to stay small and balanced—preventing a few large weights from dominating the model’s predictions, and ultimately mitigating(减轻) the risk of overfitting.
