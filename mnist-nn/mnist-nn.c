/*
 ============================================================================
 Name        : EnhancedNN_MNIST.c
 Author      : Claude (based on Darius Malysiak's original, with MNIST
 integration and evaluation) Version     : 1.2 Description : Enhanced Neural
 Network implementation with variability, dropout, and proper weight
 initialization, integrated with MNIST dataset, including training and test set
 evaluation.
 ============================================================================
 */

#include "libmin.h"
#include <stdio.h> // Included for host file loading. Remove if not using ENABLE_HOST_FILE_LOADING

//================ CONFIGURATION ================
// Network architecture
#define INPUT_SIZE 784     // 28x28 pixel images for MNIST
#define HIDDEN_SIZE 128    // Size of hidden layer
#define OUTPUT_SIZE 10     // 10 digits (0-9) for MNIST
#define MINI_BATCH_SIZE 32 // Mini-batch size for SGD

// Training parameters
#define MAX_EPOCHS 10
#define LEARNING_RATE 0.01
#define MOMENTUM 0.9
#define DROPOUT_RATE                                                           \
  0.5 // Probability of *keeping* a neuron active (inverted dropout scaling)

// Initialization parameters
#define XAVIER_INIT_FACTOR                                                     \
  1.0 // Common factor for Xavier/Glorot. Can be 1.0 or 2.0.

// Data type used throughout the network
typedef double DTYPE;

// MNIST Data Files
#define TRAIN_IMAGES_FILE "train-images.idx3-ubyte"
#define TRAIN_LABELS_FILE "train-labels.idx1-ubyte"
#define TEST_IMAGES_FILE "t10k-images.idx3-ubyte"
#define TEST_LABELS_FILE "t10k-labels.idx1-ubyte"

// Number of MNIST samples to load
#define NUM_TRAIN_SAMPLES_TO_LOAD 60000 // Max 60000 for full training set
#define NUM_TEST_SAMPLES_TO_LOAD 10000  // Max 10000 for full test set

// Enable this macro to use standard C file I/O for loading MNIST on a host
// system. For embedded targets, this should be disabled, and MNIST data should
// be pre-loaded.
#define ENABLE_HOST_FILE_LOADING

//================ GLOBAL MNIST DATA STORAGE (Loaded from files)
//================
// Training data
DTYPE **mnist_train_images = NULL;
int *mnist_train_labels = NULL;
int num_actual_train_samples_loaded = 0;

// Test data (will be loaded by evaluation function)
DTYPE **mnist_test_images = NULL;
int *mnist_test_labels = NULL;
int num_actual_test_samples_loaded = 0;

//================ UTILITY FUNCTIONS ================

inline DTYPE relu(DTYPE x) { return x > 0 ? x : 0; }

inline DTYPE relu_derivative(DTYPE x) { return x > 0 ? 1 : 0; }

void softmax(DTYPE *input, DTYPE *output, int size) {
  if (size <= 0)
    return;
  DTYPE max_val = input[0];
  for (int i = 1; i < size; i++) {
    if (input[i] > max_val)
      max_val = input[i];
  }

  DTYPE sum = 0;
  for (int i = 0; i < size; i++) {
    output[i] = libmin_exp(input[i] - max_val);
    sum += output[i];
  }

  if (sum == 0) { // Avoid division by zero
    for (int i = 0; i < size; i++)
      output[i] = 1.0 / size;
    return;
  }

  for (int i = 0; i < size; i++) {
    output[i] /= sum;
  }
}

DTYPE custom_log(DTYPE x) {
  if (x == 0)
    return -1000.0;
  if (x < 0)
    return -1000.0;
  if (x == 1.0)
    return 0.0;

  int k = 0;
  while (x >= 1.5) {
    x /= 2.0;
    k++;
  }
  while (x < 0.75) {
    x *= 2.0;
    k--;
  }

  DTYPE y = x - 1.0;
  DTYPE term = y;
  DTYPE log_val = y;
  DTYPE y_pow = y;
  for (int i = 2; i < 15; ++i) {
    y_pow *= -y;
    term = y_pow / i;
    log_val += term;
  }
  DTYPE log_2 = 0.6931471805599453;
  return log_val + k * log_2;
}

DTYPE cross_entropy_loss(DTYPE *predictions, int target_index,
                         int num_classes) {
  DTYPE epsilon = 1e-12;
  DTYPE prob = predictions[target_index];
  if (prob < epsilon)
    prob = epsilon;
  if (prob > 1.0 - epsilon)
    prob = 1.0 - epsilon;
  return -custom_log(prob);
}

DTYPE random_uniform() { return (DTYPE)libmin_rand() / (DTYPE)RAND_MAX; }

DTYPE random_normal(DTYPE mean, DTYPE stddev) {
  DTYPE u1, u2;
  do {
    u1 = random_uniform();
  } while (u1 == 0.0);
  u2 = random_uniform();
  DTYPE z0 = libmin_sqrt(-2.0 * custom_log(u1)) * libmin_cos(2.0 * M_PI * u2);
  return mean + z0 * stddev;
}

//================ NETWORK STRUCTURE ================

typedef struct {
  DTYPE *hidden_weights;
  DTYPE *hidden_biases;
  DTYPE *hidden_z;
  DTYPE *hidden_activations;
  DTYPE *hidden_dropout_mask;

  DTYPE *output_weights;
  DTYPE *output_biases;
  DTYPE *output_z;
  DTYPE *output_activations;

  DTYPE *d_hidden_weights;
  DTYPE *d_hidden_biases;
  DTYPE *d_output_weights;
  DTYPE *d_output_biases;

  DTYPE *m_hidden_weights;
  DTYPE *m_hidden_biases;
  DTYPE *m_output_weights;
  DTYPE *m_output_biases;

  DTYPE current_epoch_loss;
  int current_epoch_correct_predictions;
  int current_epoch_total_examples;

  int training_mode;
} NeuralNetwork;

void init_network(NeuralNetwork *network) {
  network->hidden_weights =
      (DTYPE *)libmin_malloc(INPUT_SIZE * HIDDEN_SIZE * sizeof(DTYPE));
  network->hidden_biases = (DTYPE *)libmin_malloc(HIDDEN_SIZE * sizeof(DTYPE));
  network->hidden_z = (DTYPE *)libmin_malloc(HIDDEN_SIZE * sizeof(DTYPE));
  network->hidden_activations =
      (DTYPE *)libmin_malloc(HIDDEN_SIZE * sizeof(DTYPE));
  network->hidden_dropout_mask =
      (DTYPE *)libmin_malloc(HIDDEN_SIZE * sizeof(DTYPE));

  network->output_weights =
      (DTYPE *)libmin_malloc(HIDDEN_SIZE * OUTPUT_SIZE * sizeof(DTYPE));
  network->output_biases = (DTYPE *)libmin_malloc(OUTPUT_SIZE * sizeof(DTYPE));
  network->output_z = (DTYPE *)libmin_malloc(OUTPUT_SIZE * sizeof(DTYPE));
  network->output_activations =
      (DTYPE *)libmin_malloc(OUTPUT_SIZE * sizeof(DTYPE));

  network->d_hidden_weights =
      (DTYPE *)libmin_calloc(INPUT_SIZE * HIDDEN_SIZE, sizeof(DTYPE));
  network->d_hidden_biases = (DTYPE *)libmin_calloc(HIDDEN_SIZE, sizeof(DTYPE));
  network->d_output_weights =
      (DTYPE *)libmin_calloc(HIDDEN_SIZE * OUTPUT_SIZE, sizeof(DTYPE));
  network->d_output_biases = (DTYPE *)libmin_calloc(OUTPUT_SIZE, sizeof(DTYPE));

  network->m_hidden_weights =
      (DTYPE *)libmin_calloc(INPUT_SIZE * HIDDEN_SIZE, sizeof(DTYPE));
  network->m_hidden_biases = (DTYPE *)libmin_calloc(HIDDEN_SIZE, sizeof(DTYPE));
  network->m_output_weights =
      (DTYPE *)libmin_calloc(HIDDEN_SIZE * OUTPUT_SIZE, sizeof(DTYPE));
  network->m_output_biases = (DTYPE *)libmin_calloc(OUTPUT_SIZE, sizeof(DTYPE));

  if (!network->hidden_weights || !network->hidden_biases ||
      !network->hidden_z || !network->hidden_activations ||
      !network->hidden_dropout_mask || !network->output_weights ||
      !network->output_biases || !network->output_z ||
      !network->output_activations || !network->d_hidden_weights ||
      !network->d_hidden_biases || !network->d_output_weights ||
      !network->d_output_biases || !network->m_hidden_weights ||
      !network->m_hidden_biases || !network->m_output_weights ||
      !network->m_output_biases) {
    libmin_printf("ERROR: Memory allocation failed in init_network\n");
    libmin_fail(1);
  }

  network->training_mode = 1;
  network->current_epoch_loss = 0.0;
  network->current_epoch_correct_predictions = 0;
  network->current_epoch_total_examples = 0;
}

void free_network(NeuralNetwork *network) {
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

void initialize_weights(NeuralNetwork *network) {
  DTYPE hidden_stddev = libmin_sqrt(XAVIER_INIT_FACTOR / (DTYPE)INPUT_SIZE);
  for (int i = 0; i < INPUT_SIZE * HIDDEN_SIZE; i++)
    network->hidden_weights[i] = random_normal(0.0, hidden_stddev);
  for (int i = 0; i < HIDDEN_SIZE; i++)
    network->hidden_biases[i] = 0.0;

  DTYPE output_stddev = libmin_sqrt(XAVIER_INIT_FACTOR / (DTYPE)HIDDEN_SIZE);
  for (int i = 0; i < HIDDEN_SIZE * OUTPUT_SIZE; i++)
    network->output_weights[i] = random_normal(0.0, output_stddev);
  for (int i = 0; i < OUTPUT_SIZE; i++)
    network->output_biases[i] = 0.0;
}

void generate_dropout_mask(NeuralNetwork *network) {
  for (int i = 0; i < HIDDEN_SIZE; i++) {
    if (random_uniform() < DROPOUT_RATE) {
      network->hidden_dropout_mask[i] = 1.0 / DROPOUT_RATE;
    } else {
      network->hidden_dropout_mask[i] = 0.0;
    }
  }
}

void forward(NeuralNetwork *network, DTYPE *input) {
  for (int j = 0; j < HIDDEN_SIZE; j++) {
    DTYPE sum = network->hidden_biases[j];
    for (int i = 0; i < INPUT_SIZE; i++)
      sum += input[i] * network->hidden_weights[i * HIDDEN_SIZE + j];
    network->hidden_z[j] = sum;
    network->hidden_activations[j] = relu(sum);
    if (network->training_mode)
      network->hidden_activations[j] *= network->hidden_dropout_mask[j];
  }
  for (int j = 0; j < OUTPUT_SIZE; j++) {
    DTYPE sum = network->output_biases[j];
    for (int i = 0; i < HIDDEN_SIZE; i++)
      sum += network->hidden_activations[i] *
             network->output_weights[i * OUTPUT_SIZE + j];
    network->output_z[j] = sum;
  }
  softmax(network->output_z, network->output_activations, OUTPUT_SIZE);
}

void backward(NeuralNetwork *network, DTYPE *input, int target_idx) {
  DTYPE *delta_output = (DTYPE *)libmin_malloc(OUTPUT_SIZE * sizeof(DTYPE));
  for (int i = 0; i < OUTPUT_SIZE; i++)
    delta_output[i] =
        network->output_activations[i] - (i == target_idx ? 1.0 : 0.0);

  for (int i = 0; i < HIDDEN_SIZE; i++) {
    for (int j = 0; j < OUTPUT_SIZE; j++) {
      network->d_output_weights[i * OUTPUT_SIZE + j] =
          network->hidden_activations[i] * delta_output[j];
    }
  }
  for (int j = 0; j < OUTPUT_SIZE; j++)
    network->d_output_biases[j] = delta_output[j];

  DTYPE *delta_hidden = (DTYPE *)libmin_malloc(HIDDEN_SIZE * sizeof(DTYPE));
  for (int i = 0; i < HIDDEN_SIZE; i++) {
    DTYPE sum_error = 0.0;
    for (int j = 0; j < OUTPUT_SIZE; j++)
      sum_error +=
          delta_output[j] * network->output_weights[i * OUTPUT_SIZE + j];
    delta_hidden[i] = sum_error * relu_derivative(network->hidden_z[i]);
    if (network->training_mode)
      delta_hidden[i] *= network->hidden_dropout_mask[i];
  }

  for (int i = 0; i < INPUT_SIZE; i++) {
    for (int j = 0; j < HIDDEN_SIZE; j++) {
      network->d_hidden_weights[i * HIDDEN_SIZE + j] =
          input[i] * delta_hidden[j];
    }
  }
  for (int j = 0; j < HIDDEN_SIZE; j++)
    network->d_hidden_biases[j] = delta_hidden[j];

  libmin_free(delta_output);
  libmin_free(delta_hidden);
}

void update_weights(NeuralNetwork *network, DTYPE learning_rate,
                    DTYPE momentum_val) {
  for (int i = 0; i < INPUT_SIZE * HIDDEN_SIZE; i++) {
    network->m_hidden_weights[i] = momentum_val * network->m_hidden_weights[i] +
                                   learning_rate * network->d_hidden_weights[i];
    network->hidden_weights[i] -= network->m_hidden_weights[i];
  }
  for (int i = 0; i < HIDDEN_SIZE; i++) {
    network->m_hidden_biases[i] = momentum_val * network->m_hidden_biases[i] +
                                  learning_rate * network->d_hidden_biases[i];
    network->hidden_biases[i] -= network->m_hidden_biases[i];
  }
  for (int i = 0; i < HIDDEN_SIZE * OUTPUT_SIZE; i++) {
    network->m_output_weights[i] = momentum_val * network->m_output_weights[i] +
                                   learning_rate * network->d_output_weights[i];
    network->output_weights[i] -= network->m_output_weights[i];
  }
  for (int i = 0; i < OUTPUT_SIZE; i++) {
    network->m_output_biases[i] = momentum_val * network->m_output_biases[i] +
                                  learning_rate * network->d_output_biases[i];
    network->output_biases[i] -= network->m_output_biases[i];
  }
}

void train_minibatch(NeuralNetwork *network, DTYPE **batch_inputs,
                     int *batch_targets, int current_batch_size,
                     DTYPE learning_rate, DTYPE momentum_val) {
  DTYPE *acc_d_hidden_weights =
      (DTYPE *)libmin_calloc(INPUT_SIZE * HIDDEN_SIZE, sizeof(DTYPE));
  DTYPE *acc_d_hidden_biases =
      (DTYPE *)libmin_calloc(HIDDEN_SIZE, sizeof(DTYPE));
  DTYPE *acc_d_output_weights =
      (DTYPE *)libmin_calloc(HIDDEN_SIZE * OUTPUT_SIZE, sizeof(DTYPE));
  DTYPE *acc_d_output_biases =
      (DTYPE *)libmin_calloc(OUTPUT_SIZE, sizeof(DTYPE));

  if (!acc_d_hidden_weights || !acc_d_hidden_biases || !acc_d_output_weights ||
      !acc_d_output_biases) {
    libmin_printf("ERROR: Memory allocation failed for gradient accumulators "
                  "in train_minibatch\n");
    libmin_free(acc_d_hidden_weights);
    libmin_free(acc_d_hidden_biases);
    libmin_free(acc_d_output_weights);
    libmin_free(acc_d_output_biases);
    libmin_fail(1);
  }

  for (int b = 0; b < current_batch_size; b++) {
    if (network->training_mode)
      generate_dropout_mask(network);
    forward(network, batch_inputs[b]);

    DTYPE loss = cross_entropy_loss(network->output_activations,
                                    batch_targets[b], OUTPUT_SIZE);
    network->current_epoch_loss += loss;

    int predicted_class = 0;
    DTYPE max_prob = network->output_activations[0];
    for (int i = 1; i < OUTPUT_SIZE; i++)
      if (network->output_activations[i] > max_prob) {
        max_prob = network->output_activations[i];
        predicted_class = i;
      }
    if (predicted_class == batch_targets[b])
      network->current_epoch_correct_predictions++;
    network->current_epoch_total_examples++;

    backward(network, batch_inputs[b], batch_targets[b]);

    for (int i = 0; i < INPUT_SIZE * HIDDEN_SIZE; i++)
      acc_d_hidden_weights[i] += network->d_hidden_weights[i];
    for (int i = 0; i < HIDDEN_SIZE; i++)
      acc_d_hidden_biases[i] += network->d_hidden_biases[i];
    for (int i = 0; i < HIDDEN_SIZE * OUTPUT_SIZE; i++)
      acc_d_output_weights[i] += network->d_output_weights[i];
    for (int i = 0; i < OUTPUT_SIZE; i++)
      acc_d_output_biases[i] += network->d_output_biases[i];
  }

  if (current_batch_size > 0) {
    for (int i = 0; i < INPUT_SIZE * HIDDEN_SIZE; i++)
      network->d_hidden_weights[i] =
          acc_d_hidden_weights[i] / current_batch_size;
    for (int i = 0; i < HIDDEN_SIZE; i++)
      network->d_hidden_biases[i] = acc_d_hidden_biases[i] / current_batch_size;
    for (int i = 0; i < HIDDEN_SIZE * OUTPUT_SIZE; i++)
      network->d_output_weights[i] =
          acc_d_output_weights[i] / current_batch_size;
    for (int i = 0; i < OUTPUT_SIZE; i++)
      network->d_output_biases[i] = acc_d_output_biases[i] / current_batch_size;
  }

  libmin_free(acc_d_hidden_weights);
  libmin_free(acc_d_hidden_biases);
  libmin_free(acc_d_output_weights);
  libmin_free(acc_d_output_biases);

  update_weights(network, learning_rate, momentum_val);
}

void set_training_mode(NeuralNetwork *network, int training) {
  network->training_mode = training;
}

int predict_single(NeuralNetwork *network, DTYPE *input) {
  int original_mode = network->training_mode;
  set_training_mode(network, 0);
  forward(network, input);
  int predicted_class = 0;
  DTYPE max_prob = network->output_activations[0];
  for (int i = 1; i < OUTPUT_SIZE; i++)
    if (network->output_activations[i] > max_prob) {
      max_prob = network->output_activations[i];
      predicted_class = i;
    }
  set_training_mode(network, original_mode);
  return predicted_class;
}

void shuffle_data(DTYPE **image_data, int *label_data, int n) {
  if (n <= 1)
    return;
  for (int i = n - 1; i > 0; i--) {
    int j = libmin_rand() % (i + 1);
    DTYPE *temp_image = image_data[i];
    image_data[i] = image_data[j];
    image_data[j] = temp_image;
    int temp_label = label_data[i];
    label_data[i] = label_data[j];
    label_data[j] = temp_label;
  }
}

//================ MNIST DATA LOADING ================

uint32_t read_int_big_endian(MFILE *mf) {
  unsigned char bytes[4];
  if (libmin_mread(bytes, 4, mf) != 4) {
    libmin_printf("ERROR: Failed to read 4 bytes for integer.\n");
    libmin_fail(1);
  }
  return ((uint32_t)bytes[0] << 24) | ((uint32_t)bytes[1] << 16) |
         ((uint32_t)bytes[2] << 8) | ((uint32_t)bytes[3]);
}

int load_mnist_labels_from_mfile(MFILE *mf, int **labels_array_ptr,
                                 int *num_items_ptr, int max_items_to_load) {
  uint32_t magic_number = read_int_big_endian(mf);
  if (magic_number != 0x00000801) {
    libmin_printf("ERROR: Invalid magic number for MNIST label file: %u\n",
                  magic_number);
    return 0;
  }
  uint32_t num_items_total = read_int_big_endian(mf);
  *num_items_ptr = (num_items_total < (uint32_t)max_items_to_load)
                       ? num_items_total
                       : max_items_to_load;
  *labels_array_ptr = (int *)libmin_malloc(*num_items_ptr * sizeof(int));
  if (!(*labels_array_ptr)) {
    libmin_printf("ERROR: Failed to allocate memory for labels.\n");
    return 0;
  }
  unsigned char label_byte;
  for (int i = 0; i < *num_items_ptr; i++) {
    if (libmin_mread(&label_byte, 1, mf) != 1) {
      libmin_printf("ERROR: Failed to read label %d.\n", i);
      libmin_free(*labels_array_ptr);
      *labels_array_ptr = NULL;
      return 0;
    }
    (*labels_array_ptr)[i] = (int)label_byte;
  }
  if (num_items_total > (uint32_t)(*num_items_ptr))
    libmin_printf("INFO: Loaded %d labels out of %u total available labels.\n",
                  *num_items_ptr, num_items_total);
  return 1;
}

int load_mnist_images_from_mfile(MFILE *mf, DTYPE ***images_array_ptr,
                                 int *num_items_ptr, int *rows_ptr,
                                 int *cols_ptr, int max_items_to_load) {
  uint32_t magic_number = read_int_big_endian(mf);
  if (magic_number != 0x00000803) {
    libmin_printf("ERROR: Invalid magic number for MNIST image file: %u\n",
                  magic_number);
    return 0;
  }
  uint32_t num_images_total = read_int_big_endian(mf);
  *rows_ptr = read_int_big_endian(mf);
  *cols_ptr = read_int_big_endian(mf);
  if (*rows_ptr * *cols_ptr != INPUT_SIZE) {
    libmin_printf("ERROR: MNIST image dimensions (%d x %d) do not match "
                  "INPUT_SIZE (%d).\n",
                  *rows_ptr, *cols_ptr, INPUT_SIZE);
    return 0;
  }
  *num_items_ptr = (num_images_total < (uint32_t)max_items_to_load)
                       ? num_images_total
                       : max_items_to_load;
  *images_array_ptr = (DTYPE **)libmin_malloc(*num_items_ptr * sizeof(DTYPE *));
  if (!(*images_array_ptr)) {
    libmin_printf("ERROR: Failed to allocate memory for image pointers.\n");
    return 0;
  }
  int image_size_bytes = (*rows_ptr) * (*cols_ptr);
  unsigned char *pixel_buffer =
      (unsigned char *)libmin_malloc(image_size_bytes * sizeof(unsigned char));
  if (!pixel_buffer) {
    libmin_printf("ERROR: Failed to allocate memory for pixel buffer.\n");
    libmin_free(*images_array_ptr);
    *images_array_ptr = NULL;
    return 0;
  }
  for (int i = 0; i < *num_items_ptr; i++) {
    (*images_array_ptr)[i] =
        (DTYPE *)libmin_malloc(image_size_bytes * sizeof(DTYPE));
    if (!(*images_array_ptr)[i]) {
      libmin_printf("ERROR: Failed to allocate memory for image %d.\n", i);
      for (int k = 0; k < i; k++) { // This loop is fine on its own
        libmin_free((*images_array_ptr)[k]);
      }
      libmin_free(*images_array_ptr); // Free the array of pointers
      *images_array_ptr = NULL;
      libmin_free(pixel_buffer); // Free the pixel buffer
      return 0;                  // Exit the function
    }
    if (libmin_mread(pixel_buffer, image_size_bytes, mf) !=
        (size_t)image_size_bytes) {
      libmin_printf("ERROR: Failed to read image %d data.\n", i);
      for (int k = 0; k <= i; k++)
        libmin_free((*images_array_ptr)[k]);
      libmin_free(*images_array_ptr);
      *images_array_ptr = NULL;
      libmin_free(pixel_buffer);
      return 0;
    }
    for (int j = 0; j < image_size_bytes; j++)
      (*images_array_ptr)[i][j] = (DTYPE)pixel_buffer[j] / 255.0;
  }
  libmin_free(pixel_buffer);
  if (num_images_total > (uint32_t)(*num_items_ptr))
    libmin_printf("INFO: Loaded %d images out of %u total available images.\n",
                  *num_items_ptr, num_images_total);
  return 1;
}

#ifdef ENABLE_HOST_FILE_LOADING
int host_load_file_into_mfile_data(MFILE *mfile, const char *filename,
                                   uint8_t **data_buffer_ptr) {
  FILE *f = fopen(filename, "rb");
  if (!f) {
    libmin_printf("HOST_LOADER ERROR: Cannot open file %s\n", filename);
    return 0;
  }
  fseek(f, 0, SEEK_END);
  long file_size_long = ftell(f);
  fseek(f, 0, SEEK_SET);
  if (file_size_long < 0) {
    libmin_printf("HOST_LOADER ERROR: Cannot get size of file %s\n", filename);
    fclose(f);
    return 0;
  }
  mfile->data_sz = (size_t)file_size_long;
  *data_buffer_ptr = (uint8_t *)libmin_malloc(mfile->data_sz);
  if (!(*data_buffer_ptr)) {
    libmin_printf(
        "HOST_LOADER ERROR: Cannot allocate memory (%zu bytes) for file %s\n",
        mfile->data_sz, filename);
    fclose(f);
    return 0;
  }
  if (fread(*data_buffer_ptr, 1, mfile->data_sz, f) != mfile->data_sz) {
    libmin_printf("HOST_LOADER ERROR: Cannot read file %s into buffer\n",
                  filename);
    libmin_free(*data_buffer_ptr);
    *data_buffer_ptr = NULL;
    fclose(f);
    return 0;
  }
  fclose(f);
  mfile->data = *data_buffer_ptr;
  mfile->fname = (char *)filename;
  mfile->rdptr = 0;
  libmin_printf("HOST_LOADER INFO: Successfully loaded %s (%zu bytes)\n",
                filename, mfile->data_sz);
  return 1;
}
#endif

void train_nn_on_mnist(NeuralNetwork *network, int num_epochs) {
  int mnist_rows, mnist_cols;
#ifdef ENABLE_HOST_FILE_LOADING
  MFILE train_images_mfile;
  MFILE train_labels_mfile;
  uint8_t *train_images_buffer = NULL;
  uint8_t *train_labels_buffer = NULL;
  libmin_printf("Attempting to load MNIST training images...\n");
  if (!host_load_file_into_mfile_data(&train_images_mfile, TRAIN_IMAGES_FILE,
                                      &train_images_buffer)) {
    libmin_printf("Failed to load training images. Exiting.\n");
    libmin_fail(1);
  }
  libmin_mopen(&train_images_mfile, "r");
  if (!load_mnist_images_from_mfile(&train_images_mfile, &mnist_train_images,
                                    &num_actual_train_samples_loaded,
                                    &mnist_rows, &mnist_cols,
                                    NUM_TRAIN_SAMPLES_TO_LOAD)) {
    libmin_printf("Failed to parse training images. Exiting.\n");
    libmin_mclose(&train_images_mfile);
    libmin_free(train_images_buffer);
    libmin_fail(1);
  }
  libmin_mclose(&train_images_mfile);
  libmin_printf("Attempting to load MNIST training labels...\n");
  if (!host_load_file_into_mfile_data(&train_labels_mfile, TRAIN_LABELS_FILE,
                                      &train_labels_buffer)) {
    libmin_printf("Failed to load training labels. Exiting.\n");
    for (int i = 0; i < num_actual_train_samples_loaded; ++i)
      libmin_free(mnist_train_images[i]);
    libmin_free(mnist_train_images);
    libmin_free(train_images_buffer);
    libmin_fail(1);
  }
  libmin_mopen(&train_labels_mfile, "r");
  int num_labels_loaded;
  if (!load_mnist_labels_from_mfile(&train_labels_mfile, &mnist_train_labels,
                                    &num_labels_loaded,
                                    NUM_TRAIN_SAMPLES_TO_LOAD)) {
    libmin_printf("Failed to parse training labels. Exiting.\n");
    for (int i = 0; i < num_actual_train_samples_loaded; ++i)
      libmin_free(mnist_train_images[i]);
    libmin_free(mnist_train_images);
    libmin_free(train_images_buffer);
    libmin_mclose(&train_labels_mfile);
    libmin_free(train_labels_buffer);
    libmin_fail(1);
  }
  libmin_mclose(&train_labels_mfile);
  if (num_actual_train_samples_loaded != num_labels_loaded) {
    libmin_printf(
        "ERROR: Mismatch in number of loaded images (%d) and labels (%d).\n",
        num_actual_train_samples_loaded, num_labels_loaded);
    for (int i = 0; i < num_actual_train_samples_loaded; ++i)
      libmin_free(mnist_train_images[i]);
    libmin_free(mnist_train_images);
    if (mnist_train_labels)
      libmin_free(mnist_train_labels);
    libmin_free(train_images_buffer);
    libmin_free(train_labels_buffer);
    libmin_fail(1);
  }
  libmin_printf("Successfully loaded %d MNIST training examples.\n",
                num_actual_train_samples_loaded);
#else
  libmin_printf("INFO: ENABLE_HOST_FILE_LOADING is not defined. MNIST data "
                "loading from disk is disabled.\n");
  libmin_fail(1);
  return;
#endif
  if (num_actual_train_samples_loaded == 0) {
    libmin_printf("No training data loaded. Cannot train.\n");
#ifdef ENABLE_HOST_FILE_LOADING
    if (train_images_buffer)
      libmin_free(train_images_buffer);
    if (train_labels_buffer)
      libmin_free(train_labels_buffer);
#endif
    return;
  }
  set_training_mode(network, 1);
  for (int epoch = 0; epoch < num_epochs; epoch++) {
    network->current_epoch_loss = 0.0;
    network->current_epoch_correct_predictions = 0;
    network->current_epoch_total_examples = 0;
    shuffle_data(mnist_train_images, mnist_train_labels,
                 num_actual_train_samples_loaded);
    for (int batch_start = 0; batch_start < num_actual_train_samples_loaded;
         batch_start += MINI_BATCH_SIZE) {
      int current_batch_size = MINI_BATCH_SIZE;
      if (batch_start + current_batch_size > num_actual_train_samples_loaded)
        current_batch_size = num_actual_train_samples_loaded - batch_start;
      if (current_batch_size <= 0)
        continue;
      train_minibatch(network, &mnist_train_images[batch_start],
                      &mnist_train_labels[batch_start], current_batch_size,
                      LEARNING_RATE, MOMENTUM);
    }
    DTYPE avg_loss = network->current_epoch_total_examples > 0
                         ? network->current_epoch_loss /
                               network->current_epoch_total_examples
                         : 0;
    DTYPE accuracy = network->current_epoch_total_examples > 0
                         ? (DTYPE)network->current_epoch_correct_predictions /
                               network->current_epoch_total_examples
                         : 0;
    libmin_printf("Epoch %d/%d: Avg Loss = %f, Accuracy = %f%%\n", epoch + 1,
                  num_epochs, avg_loss, accuracy * 100.0);
  }
  for (int i = 0; i < num_actual_train_samples_loaded; i++)
    libmin_free(mnist_train_images[i]);
  libmin_free(mnist_train_images);
  libmin_free(mnist_train_labels);
  mnist_train_images = NULL;
  mnist_train_labels = NULL;
  num_actual_train_samples_loaded = 0;
#ifdef ENABLE_HOST_FILE_LOADING
  if (train_images_buffer)
    libmin_free(train_images_buffer);
  if (train_labels_buffer)
    libmin_free(train_labels_buffer);
#endif
}

void evaluate_nn_on_mnist_test_set(NeuralNetwork *network) {
  libmin_printf("\nEvaluating on MNIST test set...\n");
  int mnist_rows, mnist_cols;
  uint8_t *test_images_buffer = NULL;
  uint8_t *test_labels_buffer = NULL; // For host-loaded raw data

#ifdef ENABLE_HOST_FILE_LOADING
  MFILE test_images_mfile;
  MFILE test_labels_mfile;
  libmin_printf("Attempting to load MNIST test images...\n");
  if (!host_load_file_into_mfile_data(&test_images_mfile, TEST_IMAGES_FILE,
                                      &test_images_buffer)) {
    libmin_printf("Failed to load test images. Skipping evaluation.\n");
    return;
  }
  libmin_mopen(&test_images_mfile, "r");
  if (!load_mnist_images_from_mfile(&test_images_mfile, &mnist_test_images,
                                    &num_actual_test_samples_loaded,
                                    &mnist_rows, &mnist_cols,
                                    NUM_TEST_SAMPLES_TO_LOAD)) {
    libmin_printf("Failed to parse test images. Skipping evaluation.\n");
    libmin_mclose(&test_images_mfile);
    libmin_free(test_images_buffer);
    return;
  }
  libmin_mclose(&test_images_mfile);
  libmin_printf("Attempting to load MNIST test labels...\n");
  if (!host_load_file_into_mfile_data(&test_labels_mfile, TEST_LABELS_FILE,
                                      &test_labels_buffer)) {
    libmin_printf("Failed to load test labels. Skipping evaluation.\n");
    for (int i = 0; i < num_actual_test_samples_loaded; ++i)
      libmin_free(mnist_test_images[i]);
    if (mnist_test_images)
      libmin_free(mnist_test_images);
    mnist_test_images = NULL;
    libmin_free(test_images_buffer);
    return;
  }
  libmin_mopen(&test_labels_mfile, "r");
  int num_labels_loaded_test;
  if (!load_mnist_labels_from_mfile(&test_labels_mfile, &mnist_test_labels,
                                    &num_labels_loaded_test,
                                    NUM_TEST_SAMPLES_TO_LOAD)) {
    libmin_printf("Failed to parse test labels. Skipping evaluation.\n");
    for (int i = 0; i < num_actual_test_samples_loaded; ++i)
      libmin_free(mnist_test_images[i]);
    if (mnist_test_images)
      libmin_free(mnist_test_images);
    mnist_test_images = NULL;
    libmin_free(test_images_buffer);
    libmin_mclose(&test_labels_mfile);
    libmin_free(test_labels_buffer);
    return;
  }
  libmin_mclose(&test_labels_mfile);
  if (num_actual_test_samples_loaded != num_labels_loaded_test) {
    libmin_printf("ERROR: Mismatch in number of loaded test images (%d) and "
                  "labels (%d).\n",
                  num_actual_test_samples_loaded, num_labels_loaded_test);
    for (int i = 0; i < num_actual_test_samples_loaded; ++i)
      libmin_free(mnist_test_images[i]);
    if (mnist_test_images)
      libmin_free(mnist_test_images);
    mnist_test_images = NULL;
    if (mnist_test_labels)
      libmin_free(mnist_test_labels);
    mnist_test_labels = NULL;
    libmin_free(test_images_buffer);
    libmin_free(test_labels_buffer);
    return;
  }
  libmin_printf("Successfully loaded %d MNIST test examples.\n",
                num_actual_test_samples_loaded);
#else
  libmin_printf("INFO: ENABLE_HOST_FILE_LOADING is not defined. MNIST test "
                "data loading from disk is disabled.\n");
  libmin_printf("INFO: Skipping test set evaluation.\n");
  return;
#endif

  if (num_actual_test_samples_loaded == 0) {
    libmin_printf("No test data loaded. Skipping evaluation.\n");
#ifdef ENABLE_HOST_FILE_LOADING
    if (test_images_buffer)
      libmin_free(test_images_buffer);
    if (test_labels_buffer)
      libmin_free(test_labels_buffer);
#endif
    return;
  }

  int correct_predictions = 0;
  set_training_mode(network, 0); // Crucial: set to inference mode

  for (int i = 0; i < num_actual_test_samples_loaded; i++) {
    int predicted_label = predict_single(network, mnist_test_images[i]);
    if (predicted_label == mnist_test_labels[i]) {
      correct_predictions++;
    }
  }

  DTYPE test_accuracy = 0.0;
  if (num_actual_test_samples_loaded > 0) {
    test_accuracy = (DTYPE)correct_predictions / num_actual_test_samples_loaded;
  }
  libmin_printf("Test Set Evaluation: Accuracy = %f%% (%d / %d)\n",
                test_accuracy * 100.0, correct_predictions,
                num_actual_test_samples_loaded);

  for (int i = 0; i < num_actual_test_samples_loaded; i++)
    libmin_free(mnist_test_images[i]);
  libmin_free(mnist_test_images);
  libmin_free(mnist_test_labels);
  mnist_test_images = NULL;
  mnist_test_labels = NULL;
  num_actual_test_samples_loaded = 0;
#ifdef ENABLE_HOST_FILE_LOADING
  if (test_images_buffer)
    libmin_free(test_images_buffer);
  if (test_labels_buffer)
    libmin_free(test_labels_buffer);
#endif
}

// Main entry point
int main(void) {
  unsigned int seed = 42;
  libmin_srand(seed);
  libmin_printf("EnhancedNN for MNIST with libmin\nRandom seed: %u\n", seed);

  NeuralNetwork network;
  init_network(&network);
  initialize_weights(&network);

  libmin_printf("Network initialized. Starting training...\n");
  train_nn_on_mnist(&network, MAX_EPOCHS);

  evaluate_nn_on_mnist_test_set(&network); // Evaluate on test set

  DTYPE test_image_one[INPUT_SIZE];
  for (int i = 0; i < INPUT_SIZE; ++i)
    test_image_one[i] = 0.0;
  for (int r = 5; r < 23; ++r)
    for (int c = 13; c <= 14; ++c)
      test_image_one[r * 28 + c] = 1.0;
  int predicted_digit = predict_single(&network, test_image_one);
  libmin_printf("\nFinal test prediction for a synthetic '1': %d\n",
                predicted_digit);

  free_network(&network);
  libmin_printf("Network freed. Execution complete.\n");

  libmin_success();
  return 0;
}