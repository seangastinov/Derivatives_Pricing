import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


class TwinNetwork(nn.Module):
    def __init__(self, input_size, neurons_per_layer=50, hidden_layers=4, output_size=1):
        super(TwinNetwork, self).__init__()
        
        layers = []
        for i in range(hidden_layers):
            layer = nn.Linear(input_size if i == 0 else neurons_per_layer, neurons_per_layer)
            nn.init.kaiming_uniform_(layer.weight)
            nn.init.uniform_(layer.bias)
            layers.append(layer)
            layers.append(nn.Softplus()) 
        
        # Output layer
        output_layer = nn.Linear(neurons_per_layer, output_size)
        nn.init.kaiming_uniform_(output_layer.weight)
        nn.init.uniform_(output_layer.bias)
        layers.append(output_layer)
        
        self.model_layer = nn.Sequential(*layers)
    
    def forward(self, x):
        # Ensure we can compute gradients with respect to input
        x.requires_grad_(True)
        return self.model_layer(x)

def compute_gradients(y_pred, x):
    """
    Compute dy/dx prediction
    Args:
        y_pred: predicted output
        x: input tensor
    Returns:
        gradients with respect to input
    """

    gradients = torch.autograd.grad(
        outputs=y_pred,
        inputs=x,
        grad_outputs=torch.ones_like(y_pred),
        create_graph=True,  # needed for second-order gradients during training
        retain_graph=True   # needed to reuse computational graph
    )[0]
    return gradients

def custom_loss(y_pred, y_true, x_input, dydx_true, normalization_factor):
    """
    Implements the cost function C:
    C = (1/m)∑(ŷᵢ(w) - ỹᵢ)² + (1/m)∑∑(1/||X̄ⱼ||²)[x̂ᵢⱼ(w) - x̃ᵢⱼ]²
    
    Args:
        y_pred: predicted values
        y_true: true values
        x_input: input features
        dydx_true: true derivatives (X̄ⱼ in the equation)
    """
    batch_size = y_pred.size(0)
    
    dydx_pred = compute_gradients(y_pred, x_input)

    # First term: MSE for predictions
    main_diff = (y_pred - y_true)**2
    main_loss = torch.mean(main_diff)

    # Second term: MSE for gradients
    wrt_diff = normalization_factor*((dydx_pred - dydx_true)**2)
    wrt_diff_sum = torch.sum(wrt_diff, dim=1)
    wrt_loss = torch.mean(wrt_diff_sum)
    
    return main_loss + wrt_loss

def custom_loss_test(y_pred, y_pred_normalized, y_true, x_input, dydx_true, normalization_factor_test):
    """
    Implements the cost function C:
    C = (1/m)∑(ŷᵢ(w) - ỹᵢ)² + (1/m)∑∑(1/||X̄ⱼ||²)[x̂ᵢⱼ(w) - x̃ᵢⱼ]²
    
    Args:
        y_pred: predicted values
        y_true: true values
        x_input: input features
        dydx_true: true derivatives (X̄ⱼ in the equation)
    """
    batch_size = y_pred.size(0)
    
    # Compute predicted gradients (x̂ᵢⱼ(w) in the equation)
    dydx_pred = compute_gradients(y_pred_normalized, x_input)


    # First term: MSE for predictions
    main_diff = (y_pred - y_true)**2
    main_loss = torch.mean(main_diff)

    # Second term: MSE for gradients
    wrt_diff = normalization_factor_test*((dydx_pred - dydx_true)**2)
    wrt_diff_sum = torch.sum(wrt_diff, dim=1)
    wrt_loss = torch.mean(wrt_diff_sum)
    
    return main_loss + wrt_loss

def train(model, train_loader, epochs, learning_rate, gradient_clipping):
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    model.train()
    best_loss = float('inf')
    best_model_state = None
    best_optimizer_state = None

    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (x, y, dydx_true) in enumerate(train_loader):
            optimizer.zero_grad()

            # Forward pass
            y_pred = model(x)

            #Recalculate lamda_t for each batch
            x_input_np = x.detach().numpy()
            dydx_true_np = dydx_true.detach().numpy()
            lamda_t = calculate_lamda_t(x_input_np, dydx_true_np)
            lamda_t_tensor = torch.tensor(lamda_t, dtype=torch.float32)

            # Compute loss
            loss = custom_loss(y_pred, y, x, dydx_true, normalization_factor=lamda_t_tensor)
            
            # Backward pass and optimization
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)

            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == epochs - 1:
            print(f'Epoch [{epoch+1}/{epochs}], Average Loss: {avg_loss:.4f}')

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_state = model.state_dict()
            best_optimizer_state = optimizer.state_dict()

    print(f"Best loss: {best_loss}")
    torch.save(best_model_state, "model_best.pth")
    torch.save(best_optimizer_state, "optimizer_best.pth")



def load_data(file_name):
    
    df_dataset = pd.read_csv(file_name)
  
    inputs_array = df_dataset[['lm', 'r', 'tau', 'theta', 'sigma', 'rho', 'kappa', 'v0']].to_numpy()  
    y_array = df_dataset['P_hat'].to_numpy()
    labels_difflabels_array = df_dataset[["diff wrt lm", "diff wrt r", "diff wrt tau", "diff wrt theta",
                                          "diff wrt sigma", "diff wrt rho", "diff wrt kappa", "diff wrt v0"]].to_numpy()                                     
  
    inputs_array = inputs_array
    y_array = y_array
    labels_difflabels_array = labels_difflabels_array

    return inputs_array, y_array, labels_difflabels_array

def split_dataset(x, y, dydx, level=0.7):
    indices = np.arange(len(x))

    train_indices, test_indices = train_test_split(indices, train_size=level, random_state=42)
  
    x_train = x[train_indices]
    y_train = y[train_indices]
    dydx_train = dydx[train_indices]

    x_test = x[test_indices]
    y_test = y[test_indices]
    dydx_test = dydx[test_indices]

    return x_train, x_test, y_train, y_test, dydx_train, dydx_test


def normalize_dataset(x_t, y_t, dy_dx_t):
    x_t_mean = np.mean(x_t, axis=0)
    x_t_std = np.std(x_t, axis=0)
    y_t_mean = np.mean(y_t, axis=0)
    y_t_std = np.std(y_t, axis=0)

    # Normalize the input features and target values
    x_t_n = (x_t - x_t_mean) / x_t_std
    y_t_n = (y_t - y_t_mean) / y_t_std
    
    # Normalize the derivatives or gradients
    dy_dx_t_n = dy_dx_t * x_t_std / y_t_std
    
    # Calculate the normalization factor for the derivative
    lam_t = 1.0 / np.sqrt(np.mean(dy_dx_t_n**2,axis=0))

    return x_t_n, x_t_mean, x_t_std, y_t_n, y_t_mean, y_t_std, dy_dx_t_n, lam_t


 
def calculate_lamda_t(x_input_np, dydx_true_np):
    dy_dx_t_n = dydx_true_np * np.std(x_input_np, axis=0) / np.std(x_input_np)
    lam_t = 1.0 / np.sqrt(np.mean(dy_dx_t_n**2,axis=0))
    return lam_t
 
 

def evaluate_model(model_loaded, x_test_n, y_test, dydx_test_n, normalization_factor, x_train_mean, x_train_std, y_train_mean, y_train_std):
    model_loaded.eval()
    
    # Convert test data to tensors
    x_test_tensor = torch.tensor(x_test_n, dtype=torch.float32, requires_grad=True)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
    dydx_test_n = torch.tensor(dydx_test_n, dtype=torch.float32)
    # Enable gradient computation for evaluation
    with torch.enable_grad():
        # Predict on test set
        y_pred_normalized = model_loaded(x_test_tensor)
        
        # Denormalize predictions
        y_pred = y_pred_normalized * y_train_std + y_train_mean
        
        # Calculate custom loss
        custom_loss_value = custom_loss_test(y_pred, y_pred_normalized, y_test_tensor, x_test_tensor, dydx_test_n, normalization_factor).item()
        # Calculate MSE
        mse = torch.mean((y_pred - y_test_tensor) ** 2).item()
    
    return mse, custom_loss_value
