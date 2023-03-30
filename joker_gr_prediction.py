import torch
import pandas as pd
from datetime import datetime
from flask import Flask, render_template

# Load the data from the Excel file into a pandas dataframe
data = pd.read_excel('joker_data.xlsx')

# Define the input and output columns
X = data[['num1', 'num2', 'num3', 'num4', 'num5', 'joker']].values
y = data[['num1', 'num2', 'num3', 'num4', 'num5', 'joker']].values

# Convert the data to Torch tensors
X = torch.tensor(X).float()
y = torch.tensor(y).float()

# Define the model architecture
model = torch.nn.Sequential(
    torch.nn.Linear(6, 64),
    torch.nn.ReLU(),
    torch.nn.Linear(64, 32),
    torch.nn.ReLU(),
    torch.nn.Linear(32, 6)
)

# Define the loss function and optimizer
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Train the model
for epoch in range(1000):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()

# Use the trained model to make predictions
predictions = model(X)

# Print and export the predicted lottery numbers as integers, without including 0
predictions = predictions.int()
predictions = torch.clamp(predictions, min=1, max=45)
print(predictions)

now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
filename = f'lottery_predictions_{now}.txt'
with open(filename, 'w') as f:
    for row in predictions:
        row_str = ' '.join(str(num) for num in row if num != 0) + '\n'
        f.write(row_str)
print(f'Predictions saved to {filename}.')

app = Flask(__name__)

@app.route('/')
def index():
    # Make predictions and get the lottery numbers
    # Convert predictions to a list of integers
    lottery_numbers = predictions.round().int().tolist()
        
    # Render the HTML template and pass the lottery numbers as a parameter
    return render_template('index.html', lottery_numbers=lottery_numbers)

if __name__ == '__main__':
    app.run()