#include <cmath>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <iostream>

const int NUM_INPUTS = 4;
const int BLOCK_SIZE = 256;

__global__ void initWeights(float *weights, int size, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        curandState state;
        curand_init(seed, idx, 0, &state);
        weights[idx] = curand_uniform(&state) - 0.5f;
    }
}

__device__ float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }
__device__ float sigmoid_derivative(float x) { return x * (1.0f - x); }

__global__ void feedforward(float *inputs, float *weights_ih, float *weights_ho, float *hidden_output, float *output, int input_size, int hidden_size, int output_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < hidden_size) {
        float sum = 0.0f;
        for (int i = 0; i < input_size; ++i) {
            sum += inputs[i] * weights_ih[i * hidden_size + idx];
        }
        hidden_output[idx] = sigmoid(sum);
    }

    __syncthreads();

    if (idx < output_size) {
        float sum = 0.0f;
        for (int i = 0; i < hidden_size; ++i) {
            sum += hidden_output[i] * weights_ho[i * output_size + idx];
        }
        output[idx] = sigmoid(sum);
    }
}

__global__ void backpropagation(float *inputs, float *hidden_output, float *output, float *weights_ih, float *weights_ho, float *expected_output, float learning_rate, int input_size, int hidden_size, int output_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < output_size) {
        float error = expected_output[idx] - output[idx];
        float delta_output = error * sigmoid_derivative(output[idx]);

        for (int i = 0; i < hidden_size; ++i) {
            float delta_weight = learning_rate * delta_output * hidden_output[i];
            atomicAdd(&weights_ho[i * output_size + idx], delta_weight);
        }

        for (int i = 0; i < hidden_size; ++i) {
            float hidden_error = 0.0f;
            for (int j = 0; j < output_size; ++j) {
                hidden_error += delta_output * weights_ho[i * output_size + j];
            }
            float delta_hidden = hidden_error * sigmoid_derivative(hidden_output[i]);

            for (int j = 0; j < input_size; ++j) {
                float delta_weight = learning_rate * delta_hidden * inputs[j];
                atomicAdd(&weights_ih[j * hidden_size + i], delta_weight);
            }
        }
    }
}

class NeuralNetwork {
private:
    int input_size, hidden_size, output_size;
    float *d_weights_ih, *d_weights_ho;
    float *d_inputs, *d_hidden_output, *d_output, *d_expected_output;

public:
    NeuralNetwork(int input_size, int hidden_size, int output_size)
        : input_size(input_size), hidden_size(hidden_size), output_size(output_size) {
        cudaMalloc(&d_weights_ih, input_size * hidden_size * sizeof(float));
        cudaMalloc(&d_weights_ho, hidden_size * output_size * sizeof(float));
        cudaMalloc(&d_inputs, input_size * sizeof(float));
        cudaMalloc(&d_hidden_output, hidden_size * sizeof(float));
        cudaMalloc(&d_output, output_size * sizeof(float));
        cudaMalloc(&d_expected_output, output_size * sizeof(float));

        int grid_size = (input_size * hidden_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
        initWeights<<<grid_size, BLOCK_SIZE>>>(d_weights_ih, input_size * hidden_size, time(NULL));

        grid_size = (hidden_size * output_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
        initWeights<<<grid_size, BLOCK_SIZE>>>(d_weights_ho, hidden_size * output_size, time(NULL));
    }

    ~NeuralNetwork() {
        cudaFree(d_weights_ih);
        cudaFree(d_weights_ho);
        cudaFree(d_inputs);
        cudaFree(d_hidden_output);
        cudaFree(d_output);
        cudaFree(d_expected_output);
    }

    void train(float *inputs, float *expected_output, float learning_rate) {
        cudaMemcpy(d_inputs, inputs, input_size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_expected_output, expected_output, output_size * sizeof(float), cudaMemcpyHostToDevice);

        int grid_size = (std::max(hidden_size, output_size) + BLOCK_SIZE - 1) / BLOCK_SIZE;

        feedforward<<<grid_size, BLOCK_SIZE>>>(d_inputs, d_weights_ih, d_weights_ho, d_hidden_output, d_output, input_size, hidden_size, output_size);

        backpropagation<<<grid_size, BLOCK_SIZE>>>(d_inputs, d_hidden_output, d_output, d_weights_ih, d_weights_ho, d_expected_output, learning_rate, input_size, hidden_size, output_size);
    }

    void predict(float *inputs, float *output) {
        cudaMemcpy(d_inputs, inputs, input_size * sizeof(float), cudaMemcpyHostToDevice);

        int grid_size = (std::max(hidden_size, output_size) + BLOCK_SIZE - 1) / BLOCK_SIZE;

        feedforward<<<grid_size, BLOCK_SIZE>>>(d_inputs, d_weights_ih, d_weights_ho, d_hidden_output, d_output, input_size, hidden_size, output_size);

        cudaMemcpy(output, d_output, output_size * sizeof(float), cudaMemcpyDeviceToHost);
    }
};

void printResults(NeuralNetwork &nn, float inputs[NUM_INPUTS][2], const char *gateName) {
    std::cout << gateName << " model in progress." << std::endl;
    for (int i = 0; i < NUM_INPUTS; ++i) {
        float output;
        nn.predict(inputs[i], &output);
        std::cout << "[" << inputs[i][0] << " " << inputs[i][1] << "] -> " << output << std::endl;
    }
    std::cout << std::endl;
}

int main(void) {
    NeuralNetwork and_gate(2, 4, 1);
    NeuralNetwork or_gate(2, 4, 1);
    NeuralNetwork nand_gate(2, 4, 1);
    NeuralNetwork nor_gate(2, 4, 1);
    NeuralNetwork xor_gate(2, 4, 1);

    float inputs[NUM_INPUTS][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    float and_outputs[NUM_INPUTS] = {0, 0, 0, 1};
    float or_outputs[NUM_INPUTS] = {0, 1, 1, 1};
    float nand_outputs[NUM_INPUTS] = {1, 1, 1, 0};
    float nor_outputs[NUM_INPUTS] = {1, 0, 0, 0};
    float xor_outputs[NUM_INPUTS] = {0, 1, 1, 0};

    for (int epoch = 0; epoch < 20'000; ++epoch) {
        for (int i = 0; i < NUM_INPUTS; ++i) {
            and_gate.train(inputs[i], &and_outputs[i], 0.1f);
            or_gate.train(inputs[i], &or_outputs[i], 0.1f);
            nand_gate.train(inputs[i], &nand_outputs[i], 0.1f);
            nor_gate.train(inputs[i], &nor_outputs[i], 0.1f);
            xor_gate.train(inputs[i], &xor_outputs[i], 0.1f);
        }
    }

    printResults(and_gate, inputs, "AND");
    printResults(or_gate, inputs, "OR");
    printResults(nand_gate, inputs, "NAND");
    printResults(nor_gate, inputs, "NOR");
    printResults(xor_gate, inputs, "XOR");

    return 0;
}
