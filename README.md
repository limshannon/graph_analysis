# graph_analysis

01. ChebNet
Line 17: Creates the synthetic graph.

Lines 20–28: Run a for-loop to add node attributes to the graph. Use a random choice to select age, sex, location, occupation, testing status, symptoms, vaccination status, and mobility details.

Lines 31–32: Add contact details as edge weights.

Lines 35–44: Add label details (infected or not infected) using conditions such as high mobility, testing status, vaccination status, and symptom details.

Line 51: Creates an edge index of the graph, which is the default input used in this library.

Lines 54–55: Create an edge weight tensor using the contact details of the graph.

Line 59: Creates a DataFrame of all the node features.

Lines 62–64: Select relevant features and convert them into a PyTorch tensor.

Lines 67–68: Create a tensor of node labels and change the categorical variables into numerical ones.

Lines 71–75: Split the nodes and labels into training and testing sets in a ratio of 80:20 using a stratified split. This ensures equal proportions of infected and not infected cases in both sets.

Lines 77–81: Create a train_mask and test_mask using the indexes of the training and testing sets.

Lines 84–86: Create a torch_geometric.data.Data object using the tensors we created from the graph data.

Lines 93–94: Set seed for PyTorch and select device for training (CPU or GPU).

Lines 96–100: Define a class and initialize two Chebyshev convolution layers. The layers take three input parameters—namely, the input dimension, the output dimension, and K, which is the order of the Chebyshev polynomial used in the convolution operation.

Lines 102–107: Define the forward pass of the neural network. The method takes input as node features, edge indexes, and edge weights of the graph. We use two dropout layers with a probability of 0.1 and the first convolutional layer with the ReLU activation function.

Lines 110–111: Create a ChebNet object with an input dimension the same as the node feature size, a hidden layer dimension of 32, and an output dimension the same as the number of classes, i.e., 2 (infected and not infected).

Line 114: Creates an Adam optimizer with a learning rate of 0.01.

Lines 117–124: Define the training function where the model inputs the node features, edge index, and edge weight. We use a cross-entropy loss function for optimization.

Line 127: The @torch.no_grad() decorator tells PyTorch not to track gradients for the code inside the test() function. This speeds up the computation and reduces memory use since we don’t need to update the model’s parameters during testing.

Lines 128–135: Define the testing function, which evaluates the model after training. We use the train_mask and test_mask to compute the training accuracy and testing accuracy of the model.

Lines 138–140: Train the model for 100 epochs and perform the evaluation.

Lines 142–145: Print the training and testing accuracy scores.

Artifacts of the model
We can generate different artifacts during the training process, like the plot of the loss function with epochs. Here's the loss function obtained for the graph neural network that was trained for 100 epochs:


Model loss vs epochs

Model loss vs epochs
The training accuracy and the testing accuracies are greater than 80%, which shows that the model has good predictive performance. The model performance can be improved by doing hyperparameter tuning. We can use Optuna, the automatic hyperparameter tuning library, for this purpose.

This is how we formulate and evaluate a node classification problem on a biological or contact tracing network. The same idea and code can be used for any problem setting and graph of any kind. The node features change according to the graph type.

Sometimes we don't have any node features readily available. In those cases, we can use the following as node features:

A node degree

Node embedding (using graph embedding methods)

Graph properties (local coloring numbers, and so on)

Random numbers
