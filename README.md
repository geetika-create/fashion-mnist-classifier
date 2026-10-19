# Fashion-MNIST Classifier

A multi-layer perceptron built in PyTorch that classifies 28x28 grayscale images of clothing into 10 categories (T-shirt, trouser, pullover, dress, coat, sandal, shirt, sneaker, bag, ankle boot).

## Model

Fully connected network: 784 → 256 → 128 → 32 → 10, ReLU activations, trained with Adam (lr = 0.001) and cross-entropy loss for 10 epochs at batch size 32.

## Run

```bash
pip install -r requirements.txt
python fashion_mnist_ann.py
```

The script downloads Fashion-MNIST, trains the model, plots the training loss, saves the weights to `models/ann.pt`, and reports test accuracy.

## Results

Test accuracy: **88%**
