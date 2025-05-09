/*
 ============================================================================
 Name        : EnhancedNN.c
 Author      : Claude (based on Darius Malysiak's original)
 Version     : 1.0
 Description : Enhanced Neural Network implementation with variability, dropout,
               and proper weight initialization for MNIST dataset
 ============================================================================
 */

#include "libmin.h"
// Removed time.h since it might not be available in your environment

//================ CONFIGURATION ================
// Network architecture
#define INPUT_SIZE 784      // 28x28 pixel images for MNIST
#define HIDDEN_SIZE 128     // Size of hidden layer
#define OUTPUT_SIZE 10      // 10 digits (0-9) for MNIST
#define MINI_BATCH_SIZE 32  // Mini-batch size for SGD

// Training parameters
#define MAX_EPOCHS 10
#define LEARNING_RATE 0.01
#define MOMENTUM 0.9
#define DROPOUT_RATE 0.5    // Probability of keeping a neuron active

// Initialization parameters
#define XAVIER_INIT_FACTOR 2.0  // Xavier/Glorot initialization

// Data type used throughout the network
typedef double DTYPE;

//================ UTILITY FUNCTIONS ================

// ReLU activation function: max(0, x)
inline DTYPE relu(DTYPE x) {
    return x > 0 ? x : 0;
}

// Derivative of ReLU
inline DTYPE relu_derivative(DTYPE x) {
    return x > 0 ? 1 : 0;
}

// Sigmoid activation function: 1/(1+e^(-x))
inline DTYPE sigmoid(DTYPE x) {
    return 1.0 / (1.0 + libmin_exp(-x));
}

// Derivative of sigmoid
inline DTYPE sigmoid_derivative(DTYPE x) {
    DTYPE s = sigmoid(x);
    return s * (1.0 - s);
}

// Softmax function for output layer
void softmax(DTYPE* input, DTYPE* output, int size) {
    DTYPE max_val = input[0];
    for (int i = 1; i < size; i++) {
        if (input[i] > max_val) max_val = input[i];
    }
    
    DTYPE sum = 0;
    for (int i = 0; i < size; i++) {
        output[i] = libmin_exp(input[i] - max_val);
        sum += output[i];
    }
    
    for (int i = 0; i < size; i++) {
        output[i] /= sum;
    }
}

// Custom implementation of natural logarithm since libmin_log isn't available
DTYPE custom_log(DTYPE x) {
    // Use the relation: log(x) = log(1+y) where y = x-1, for x close to 1
    // For values not close to 1, we can use: log(x) = log(x/m) + log(m)
    // where m is a power of 2, so log(m) is easy to compute
    
    // Handle special cases
    if (x <= 0) return -1000.0; // Return a very negative number for log(0) or negative inputs
    
    // Scale x to [0.5, 1) range by finding m = 2^k such that x/m is in [0.5, 1)
    DTYPE m = 1.0;
    int k = 0;
    if (x >= 1.0) {
        while (x >= 2.0) { x /= 2.0; m *= 2.0; k++; }
    } else {
        while (x < 0.5) { x *= 2.0; m /= 2.0; k--; }
    }
    
    // Now x is in [0.5, 1) and log(original_x) = log(x) + k*log(2)
    
    // Compute log(x) using Taylor series for log(1+y) where y = x-1
    DTYPE y = x - 1.0;
    DTYPE y2 = y * y;
    DTYPE y3 = y2 * y;
    DTYPE y4 = y3 * y;
    DTYPE y5 = y4 * y;
    
    // log(1+y) ≈ y - y²/2 + y³/3 - y⁴/4 + y⁵/5 for |y| < 1
    DTYPE log_x = y - y2/2.0 + y3/3.0 - y4/4.0 + y5/5.0;
    
    // log(2) ≈ 0.693147180559945
    DTYPE log_2 = 0.693147180559945;
    
    return log_x + k * log_2;
}

// Cross-entropy loss function
DTYPE cross_entropy_loss(DTYPE* predictions, int target, int num_classes) {
    // Add small epsilon to avoid log(0)
    DTYPE epsilon = 1e-15;
    DTYPE prob = predictions[target];
    if (prob < epsilon) prob = epsilon;
    return -custom_log(prob);
}

// Generate random number in range [0, 1]
DTYPE random_uniform() {
    return (DTYPE)libmin_rand() / (DTYPE)RAND_MAX;
}

// Generate random number from normal distribution using Box-Muller transform
DTYPE random_normal(DTYPE mean, DTYPE stddev) {
    DTYPE u1 = random_uniform();
    DTYPE u2 = random_uniform();
    
    // Avoid log(0)
    if (u1 < 1e-8) u1 = 1e-8;
    
    DTYPE z0 = libmin_sqrt(-2.0 * custom_log(u1)) * libmin_cos(2.0 * M_PI * u2);
    return mean + z0 * stddev;
}

//================ NETWORK STRUCTURE ================

typedef struct {
    // Hidden layer
    DTYPE* hidden_weights;      // [INPUT_SIZE][HIDDEN_SIZE]
    DTYPE* hidden_biases;       // [HIDDEN_SIZE]
    DTYPE* hidden_z;            // Pre-activation outputs [HIDDEN_SIZE]
    DTYPE* hidden_activations;  // Post-activation outputs [HIDDEN_SIZE]
    DTYPE* hidden_dropout_mask; // Dropout mask [HIDDEN_SIZE]
    
    // Output layer
    DTYPE* output_weights;      // [HIDDEN_SIZE][OUTPUT_SIZE]
    DTYPE* output_biases;       // [OUTPUT_SIZE]
    DTYPE* output_z;            // Pre-activation outputs [OUTPUT_SIZE]
    DTYPE* output_activations;  // Post-activation outputs [OUTPUT_SIZE]
    
    // Gradients and momentums for SGD with momentum
    DTYPE* d_hidden_weights;    // Gradient for hidden weights
    DTYPE* d_hidden_biases;     // Gradient for hidden biases
    DTYPE* d_output_weights;    // Gradient for output weights
    DTYPE* d_output_biases;     // Gradient for output biases
    
    DTYPE* m_hidden_weights;    // Momentum for hidden weights
    DTYPE* m_hidden_biases;     // Momentum for hidden biases
    DTYPE* m_output_weights;    // Momentum for output weights
    DTYPE* m_output_biases;     // Momentum for output biases
    
    // Training stats
    DTYPE total_loss;
    int correct_predictions;
    int total_examples;
    
    // Is the network in training mode? (affects dropout)
    int training_mode;
} NeuralNetwork;

// Initialize memory for the neural network
void init_network(NeuralNetwork* network) {
    // Allocate memory for weights and biases
    network->hidden_weights = (DTYPE*)libmin_malloc(INPUT_SIZE * HIDDEN_SIZE * sizeof(DTYPE));
    network->hidden_biases = (DTYPE*)libmin_malloc(HIDDEN_SIZE * sizeof(DTYPE));
    network->hidden_z = (DTYPE*)libmin_malloc(HIDDEN_SIZE * sizeof(DTYPE));
    network->hidden_activations = (DTYPE*)libmin_malloc(HIDDEN_SIZE * sizeof(DTYPE));
    network->hidden_dropout_mask = (DTYPE*)libmin_malloc(HIDDEN_SIZE * sizeof(DTYPE));
    
    network->output_weights = (DTYPE*)libmin_malloc(HIDDEN_SIZE * OUTPUT_SIZE * sizeof(DTYPE));
    network->output_biases = (DTYPE*)libmin_malloc(OUTPUT_SIZE * sizeof(DTYPE));
    network->output_z = (DTYPE*)libmin_malloc(OUTPUT_SIZE * sizeof(DTYPE));
    network->output_activations = (DTYPE*)libmin_malloc(OUTPUT_SIZE * sizeof(DTYPE));
    
    // Initialize gradients and momentums
    network->d_hidden_weights = (DTYPE*)libmin_malloc(INPUT_SIZE * HIDDEN_SIZE * sizeof(DTYPE));
    network->d_hidden_biases = (DTYPE*)libmin_malloc(HIDDEN_SIZE * sizeof(DTYPE));
    network->d_output_weights = (DTYPE*)libmin_malloc(HIDDEN_SIZE * OUTPUT_SIZE * sizeof(DTYPE));
    network->d_output_biases = (DTYPE*)libmin_malloc(OUTPUT_SIZE * sizeof(DTYPE));
    
    network->m_hidden_weights = (DTYPE*)libmin_malloc(INPUT_SIZE * HIDDEN_SIZE * sizeof(DTYPE));
    network->m_hidden_biases = (DTYPE*)libmin_malloc(HIDDEN_SIZE * sizeof(DTYPE));
    network->m_output_weights = (DTYPE*)libmin_malloc(HIDDEN_SIZE * OUTPUT_SIZE * sizeof(DTYPE));
    network->m_output_biases = (DTYPE*)libmin_malloc(OUTPUT_SIZE * sizeof(DTYPE));
    
    // Initialize momentum arrays to zero
    for (int i = 0; i < INPUT_SIZE * HIDDEN_SIZE; i++) {
        network->m_hidden_weights[i] = 0.0;
    }
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        network->m_hidden_biases[i] = 0.0;
    }
    for (int i = 0; i < HIDDEN_SIZE * OUTPUT_SIZE; i++) {
        network->m_output_weights[i] = 0.0;
    }
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        network->m_output_biases[i] = 0.0;
    }
    
    network->training_mode = 1;  // Default to training mode
    network->total_loss = 0.0;
    network->correct_predictions = 0;
    network->total_examples = 0;
}

// Free memory used by the neural network
void free_network(NeuralNetwork* network) {
    libmin_free(network->hidden_weights);
    libmin_free(network->hidden_biases);
    libmin_free(network->hidden_z);
    libmin_free(network->hidden_activations);
    libmin_free(network->hidden_dropout_mask);
    
    libmin_free(network->output_weights);
    libmin_free(network->output_biases);
    libmin_free(network->output_z);
    libmin_free(network->output_activations);
    
    libmin_free(network->d_hidden_weights);
    libmin_free(network->d_hidden_biases);
    libmin_free(network->d_output_weights);
    libmin_free(network->d_output_biases);
    
    libmin_free(network->m_hidden_weights);
    libmin_free(network->m_hidden_biases);
    libmin_free(network->m_output_weights);
    libmin_free(network->m_output_biases);
}

// Initialize the weights with Xavier/Glorot initialization
void initialize_weights(NeuralNetwork* network) {
    // Xavier initialization factor for hidden layer
    DTYPE hidden_factor = libmin_sqrt(XAVIER_INIT_FACTOR / (DTYPE)INPUT_SIZE);
    
    // Initialize hidden layer weights
    for (int i = 0; i < INPUT_SIZE; i++) {
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            int idx = i * HIDDEN_SIZE + j;
            network->hidden_weights[idx] = random_normal(0.0, hidden_factor);
        }
    }
    
    // Initialize hidden layer biases to small values
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        network->hidden_biases[i] = 0.1 * random_normal(0.0, 1.0);
    }
    
    // Xavier initialization factor for output layer
    DTYPE output_factor = libmin_sqrt(XAVIER_INIT_FACTOR / (DTYPE)HIDDEN_SIZE);
    
    // Initialize output layer weights
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            int idx = i * OUTPUT_SIZE + j;
            network->output_weights[idx] = random_normal(0.0, output_factor);
        }
    }
    
    // Initialize output layer biases to small values
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        network->output_biases[i] = 0.1 * random_normal(0.0, 1.0);
    }
}

// Generate a new dropout mask for the hidden layer
void generate_dropout_mask(NeuralNetwork* network) {
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        // Each neuron has DROPOUT_RATE probability of being active
        network->hidden_dropout_mask[i] = (random_uniform() < DROPOUT_RATE) ? 1.0 / DROPOUT_RATE : 0.0;
    }
}

// Forward pass through the network
void forward(NeuralNetwork* network, DTYPE* input) {
    // Hidden layer forward pass
    for (int j = 0; j < HIDDEN_SIZE; j++) {
        DTYPE sum = network->hidden_biases[j];
        for (int i = 0; i < INPUT_SIZE; i++) {
            sum += input[i] * network->hidden_weights[i * HIDDEN_SIZE + j];
        }
        network->hidden_z[j] = sum;
        network->hidden_activations[j] = relu(sum);
        
        // Apply dropout during training
        if (network->training_mode) {
            network->hidden_activations[j] *= network->hidden_dropout_mask[j];
        }
    }
    
    // Output layer forward pass
    for (int j = 0; j < OUTPUT_SIZE; j++) {
        DTYPE sum = network->output_biases[j];
        for (int i = 0; i < HIDDEN_SIZE; i++) {
            sum += network->hidden_activations[i] * network->output_weights[i * OUTPUT_SIZE + j];
        }
        network->output_z[j] = sum;
    }
    
    // Apply softmax to output layer
    softmax(network->output_z, network->output_activations, OUTPUT_SIZE);
}

// Backpropagation to compute gradients
void backward(NeuralNetwork* network, DTYPE* input, int target) {
    // Output layer error (cross-entropy loss gradient with softmax)
    DTYPE* output_error = (DTYPE*)libmin_malloc(OUTPUT_SIZE * sizeof(DTYPE));
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        output_error[i] = network->output_activations[i] - (i == target ? 1.0 : 0.0);
    }
    
    // Hidden layer error
    DTYPE* hidden_error = (DTYPE*)libmin_malloc(HIDDEN_SIZE * sizeof(DTYPE));
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        hidden_error[i] = 0.0;
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            hidden_error[i] += output_error[j] * network->output_weights[i * OUTPUT_SIZE + j];
        }
        hidden_error[i] *= relu_derivative(network->hidden_z[i]);
        
        // Apply dropout mask to hidden error
        if (network->training_mode) {
            hidden_error[i] *= network->hidden_dropout_mask[i];
        }
    }
    
    // Compute gradients for output layer
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            int idx = i * OUTPUT_SIZE + j;
            network->d_output_weights[idx] = network->hidden_activations[i] * output_error[j];
        }
    }
    
    for (int j = 0; j < OUTPUT_SIZE; j++) {
        network->d_output_biases[j] = output_error[j];
    }
    
    // Compute gradients for hidden layer
    for (int i = 0; i < INPUT_SIZE; i++) {
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            int idx = i * HIDDEN_SIZE + j;
            network->d_hidden_weights[idx] = input[i] * hidden_error[j];
        }
    }
    
    for (int j = 0; j < HIDDEN_SIZE; j++) {
        network->d_hidden_biases[j] = hidden_error[j];
    }
    
    libmin_free(output_error);
    libmin_free(hidden_error);
}

// Update weights with SGD and momentum
void update_weights(NeuralNetwork* network, DTYPE learning_rate, DTYPE momentum) {
    // Update hidden layer weights
    for (int i = 0; i < INPUT_SIZE * HIDDEN_SIZE; i++) {
        network->m_hidden_weights[i] = momentum * network->m_hidden_weights[i] + 
                                      learning_rate * network->d_hidden_weights[i];
        network->hidden_weights[i] -= network->m_hidden_weights[i];
    }
    
    // Update hidden layer biases
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        network->m_hidden_biases[i] = momentum * network->m_hidden_biases[i] + 
                                     learning_rate * network->d_hidden_biases[i];
        network->hidden_biases[i] -= network->m_hidden_biases[i];
    }
    
    // Update output layer weights
    for (int i = 0; i < HIDDEN_SIZE * OUTPUT_SIZE; i++) {
        network->m_output_weights[i] = momentum * network->m_output_weights[i] + 
                                      learning_rate * network->d_output_weights[i];
        network->output_weights[i] -= network->m_output_weights[i];
    }
    
    // Update output layer biases
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        network->m_output_biases[i] = momentum * network->m_output_biases[i] + 
                                     learning_rate * network->d_output_biases[i];
        network->output_biases[i] -= network->m_output_biases[i];
    }
}

// Train the network for one mini-batch
void train_minibatch(NeuralNetwork* network, DTYPE** inputs, int* targets, int batch_size, 
                     DTYPE learning_rate, DTYPE momentum) {
    // Initialize gradients to zero
    for (int i = 0; i < INPUT_SIZE * HIDDEN_SIZE; i++) {
        network->d_hidden_weights[i] = 0.0;
    }
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        network->d_hidden_biases[i] = 0.0;
    }
    for (int i = 0; i < HIDDEN_SIZE * OUTPUT_SIZE; i++) {
        network->d_output_weights[i] = 0.0;
    }
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        network->d_output_biases[i] = 0.0;
    }
    
    // Accumulate gradients for each example in the mini-batch
    for (int b = 0; b < batch_size; b++) {
        // Generate a new dropout mask for each example
        generate_dropout_mask(network);
        
        // Forward pass
        forward(network, inputs[b]);
        
        // Compute loss
        DTYPE loss = cross_entropy_loss(network->output_activations, targets[b], OUTPUT_SIZE);
        network->total_loss += loss;
        
        // Compute accuracy
        int predicted_class = 0;
        DTYPE max_prob = network->output_activations[0];
        for (int i = 1; i < OUTPUT_SIZE; i++) {
            if (network->output_activations[i] > max_prob) {
                max_prob = network->output_activations[i];
                predicted_class = i;
            }
        }
        if (predicted_class == targets[b]) {
            network->correct_predictions++;
        }
        network->total_examples++;
        
        // Temporary arrays for single example gradients
        DTYPE* temp_d_hidden_weights = (DTYPE*)libmin_malloc(INPUT_SIZE * HIDDEN_SIZE * sizeof(DTYPE));
        DTYPE* temp_d_hidden_biases = (DTYPE*)libmin_malloc(HIDDEN_SIZE * sizeof(DTYPE));
        DTYPE* temp_d_output_weights = (DTYPE*)libmin_malloc(HIDDEN_SIZE * OUTPUT_SIZE * sizeof(DTYPE));
        DTYPE* temp_d_output_biases = (DTYPE*)libmin_malloc(OUTPUT_SIZE * sizeof(DTYPE));
        
        // Compute gradients for this example
        backward(network, inputs[b], targets[b]);
        
        // Save gradients to temporary arrays
        for (int i = 0; i < INPUT_SIZE * HIDDEN_SIZE; i++) {
            temp_d_hidden_weights[i] = network->d_hidden_weights[i];
        }
        for (int i = 0; i < HIDDEN_SIZE; i++) {
            temp_d_hidden_biases[i] = network->d_hidden_biases[i];
        }
        for (int i = 0; i < HIDDEN_SIZE * OUTPUT_SIZE; i++) {
            temp_d_output_weights[i] = network->d_output_weights[i];
        }
        for (int i = 0; i < OUTPUT_SIZE; i++) {
            temp_d_output_biases[i] = network->d_output_biases[i];
        }
        
        // Accumulate gradients for the mini-batch
        for (int i = 0; i < INPUT_SIZE * HIDDEN_SIZE; i++) {
            network->d_hidden_weights[i] += temp_d_hidden_weights[i] / batch_size;
        }
        for (int i = 0; i < HIDDEN_SIZE; i++) {
            network->d_hidden_biases[i] += temp_d_hidden_biases[i] / batch_size;
        }
        for (int i = 0; i < HIDDEN_SIZE * OUTPUT_SIZE; i++) {
            network->d_output_weights[i] += temp_d_output_weights[i] / batch_size;
        }
        for (int i = 0; i < OUTPUT_SIZE; i++) {
            network->d_output_biases[i] += temp_d_output_biases[i] / batch_size;
        }
        
        // Free temporary arrays
        libmin_free(temp_d_hidden_weights);
        libmin_free(temp_d_hidden_biases);
        libmin_free(temp_d_output_weights);
        libmin_free(temp_d_output_biases);
    }
    
    // Update weights with accumulated gradients
    update_weights(network, learning_rate, momentum);
}

// Set the network to training mode (enable dropout)
void set_training_mode(NeuralNetwork* network, int training) {
    network->training_mode = training;
}

// Predict a single example
int predict(NeuralNetwork* network, DTYPE* input) {
    // Disable dropout for prediction
    int old_mode = network->training_mode;
    network->training_mode = 0;
    
    // Forward pass
    forward(network, input);
    
    // Find the class with highest probability
    int predicted_class = 0;
    DTYPE max_prob = network->output_activations[0];
    for (int i = 1; i < OUTPUT_SIZE; i++) {
        if (network->output_activations[i] > max_prob) {
            max_prob = network->output_activations[i];
            predicted_class = i;
        }
    }
    
    // Restore original training mode
    network->training_mode = old_mode;
    
    return predicted_class;
}

// Function to shuffle the training data
void shuffle_data(DTYPE** inputs, int* targets, int n) {
    for (int i = n - 1; i > 0; i--) {
        // Generate a random index j such that 0 <= j <= i
        int j = libmin_rand() % (i + 1);
        
        // Swap inputs[i] and inputs[j]
        DTYPE* temp_input = inputs[i];
        inputs[i] = inputs[j];
        inputs[j] = temp_input;
        
        // Swap targets[i] and targets[j]
        int temp_target = targets[i];
        targets[i] = targets[j];
        targets[j] = temp_target;
    }
}

// Generate synthetic data for MNIST (for demonstration purposes)
void generate_synthetic_mnist(DTYPE** train_images, int* train_labels, int num_samples) {
    for (int i = 0; i < num_samples; i++) {
        int digit = i % 10;  // Cycle through digits 0-9
        train_labels[i] = digit;
        
        // Initialize image to all zeros
        for (int j = 0; j < INPUT_SIZE; j++) {
            train_images[i][j] = 0.0;
        }
        
        // Create a simple representation of the digit
        // (This is a very simplistic approach - real MNIST data would be used in practice)
        switch (digit) {
            case 0:
                // Draw a circle
                for (int j = 0; j < 28; j++) {
                    for (int k = 0; k < 28; k++) {
                        if ((j-14)*(j-14) + (k-14)*(k-14) < 81 && 
                            (j-14)*(j-14) + (k-14)*(k-14) > 49) {
                            train_images[i][j*28+k] = 1.0;
                        }
                    }
                }
                break;
            case 1:
                // Draw a vertical line
                for (int j = 7; j < 21; j++) {
                    for (int k = 13; k < 15; k++) {
                        train_images[i][j*28+k] = 1.0;
                    }
                }
                break;
            // Add patterns for other digits...
            default:
                // For other digits, just add some random noise
                for (int j = 0; j < INPUT_SIZE; j++) {
                    if (random_uniform() < 0.2) {
                        train_images[i][j] = random_uniform();
                    }
                }
                // Add a distinct pattern based on the digit
                int center_x = 14;
                int center_y = 14;
                int radius = 5 + digit;
                for (int j = 0; j < 28; j++) {
                    for (int k = 0; k < 28; k++) {
                        int dist = (j-center_y)*(j-center_y) + (k-center_x)*(k-center_x);
                        if (dist < radius*radius) {
                            train_images[i][j*28+k] = 0.75 + 0.25 * random_uniform();
                        }
                    }
                }
                break;
        }
        
        // Add some random noise to make it more realistic
        for (int j = 0; j < INPUT_SIZE; j++) {
            train_images[i][j] += 0.1 * random_uniform();
            // Clip to [0, 1]
            if (train_images[i][j] > 1.0) train_images[i][j] = 1.0;
            if (train_images[i][j] < 0.0) train_images[i][j] = 0.0;
        }
    }
}

// Train the network on MNIST dataset
void train_mnist(NeuralNetwork* network, int num_epochs) {
    // In a real-world scenario, you would load the actual MNIST dataset
    // Here we'll generate some synthetic data for demonstration purposes
    int num_train_samples = 1000;
    
    // Allocate memory for training data
    DTYPE** train_images = (DTYPE**)libmin_malloc(num_train_samples * sizeof(DTYPE*));
    for (int i = 0; i < num_train_samples; i++) {
        train_images[i] = (DTYPE*)libmin_malloc(INPUT_SIZE * sizeof(DTYPE));
    }
    int* train_labels = (int*)libmin_malloc(num_train_samples * sizeof(int));
    
    // Generate synthetic MNIST data
    generate_synthetic_mnist(train_images, train_labels, num_train_samples);
    
    // Training loop
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        // Reset training stats
        network->total_loss = 0.0;
        network->correct_predictions = 0;
        network->total_examples = 0;
        
        // Shuffle the training data
        shuffle_data(train_images, train_labels, num_train_samples);
        
        // Mini-batch training
        for (int batch_start = 0; batch_start < num_train_samples; batch_start += MINI_BATCH_SIZE) {
            int batch_size = MINI_BATCH_SIZE;
            if (batch_start + batch_size > num_train_samples) {
                batch_size = num_train_samples - batch_start;
            }
            
            // Create arrays for this mini-batch
            DTYPE** batch_images = &train_images[batch_start];
            int* batch_labels = &train_labels[batch_start];
            
            // Train on this mini-batch
            train_minibatch(network, batch_images, batch_labels, batch_size, 
                            LEARNING_RATE, MOMENTUM);
        }
        
        // Calculate and display epoch statistics
        DTYPE avg_loss = network->total_loss / network->total_examples;
        DTYPE accuracy = (DTYPE)network->correct_predictions / network->total_examples;
        
        libmin_printf("Epoch %d: Loss = %f, Accuracy = %f%%\n", 
                      epoch+1, avg_loss, accuracy * 100.0);
    }
    
    // Free training data
    for (int i = 0; i < num_train_samples; i++) {
        libmin_free(train_images[i]);
    }
    libmin_free(train_images);
    libmin_free(train_labels);
}

// Main entry point
int main(void) {
    // Seed random number generator with different value each time
    // In a real system, you might use time(NULL) but we're using a 
    // different value on each run to demonstrate variability
    unsigned int seed = 10;  // Change this value for different runs
    libmin_srand(seed);
    libmin_printf("Using random seed: %u\n", seed);
    
    // Initialize the neural network
    NeuralNetwork network;
    init_network(&network);
    
    // Initialize weights
    initialize_weights(&network);
    
    // Train the network on MNIST
    train_mnist(&network, MAX_EPOCHS);
    
    // Test the network with a sample digit
    DTYPE test_image[INPUT_SIZE];
    // Initialize test image with a simple "1" digit
    for (int i = 0; i < INPUT_SIZE; i++) {
        test_image[i] = 0.0;
    }
    for (int i = 7; i < 21; i++) {
        for (int j = 13; j < 15; j++) {
            test_image[i*28+j] = 1.0;
        }
    }
    
    int predicted = predict(&network, test_image);
    libmin_printf("Test prediction: %d\n", predicted);
    
    // Free network resources
    free_network(&network);
    
    libmin_success();
    return 0;
}