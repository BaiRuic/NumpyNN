from sklearn.datasets import load_digits
input, target = load_digits(return_X_y=True)


print(input.shape, target.shape)