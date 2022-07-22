import torch
import torch.nn as nn
import torch.nn.functional as F

from tsne.utils import plot
from sklearn.pipeline import Pipeline
from sklearn.datasets import load_digits
from tsne.parametric_tsne import ParametricTSNE
from sklearn.preprocessing import StandardScaler
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from tsne.ms_parametric_tsne import MultiscaleParametricTSNE

class TSNE_Net(nn.Module):

    def __init__(self):
        super(TSNE_Net, self).__init__()

        self.fc1 = nn.Linear(64, 1000)
        self.fc2 = nn.Linear(1000, 500)
        self.fc3 = nn.Linear(500, 250)

        self.output = nn.Linear(250, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        return self.output(x)

if __name__ == "__main__":
    
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