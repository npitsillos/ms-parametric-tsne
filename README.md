# Multiscale Parametric t-SNE
A multiscale parametric t-SNE implementation in PyTorch adapted from [Multiscale Parametric t-SNE](https://github.com/FrancescoCrecchi/Multiscale-Parametric-t-SNE) which is implemented in [Keras](https://keras.io).

This repository features both a parametric t-SNE and its multiscale extension.

# Usage
Clone this repository and execute `cd ms-parametric-tsne`. Then make changes to the `main.py` file to load in your dataset.  Here is an example of running the model on the MNIST dataset:

```python
model = TSNE_Net()
device = torch.device("cuda") if torch.cuda.is_available() else torch.cuda("cpu")
model.to(device)
model.float()
X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=0)

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('msp_tsne', (MultiscaleParametricTSNE(model, SummaryWriter("test"), device=device,
                                    n_components=2,
                                    n_iter=250,
                                    verbose=1)))
])

# Fit
X_train_2d = pipe.fit_transform(X_train)
plot(X_train_2d.detach().cpu().numpy(), y_train, 10, "Training Set")
# Transform
X_test_2d = pipe.transform(X_test)
plot(X_test_2d.detach().cpu().numpy(), y_test, 10, "Test Set")
```