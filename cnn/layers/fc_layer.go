package layers

import (
	"math"
	"math/rand"
)

// Sigmoid activation function
func sigmoid(x float32) float32 {
	return 1.0 / (1.0 + float32(math.Exp(-float64(x))))
}

// Inverse derivative of the sigmoid function
func invDerivSigmoid(x float32) float32 {
	return x * (1.0 - x)
}

// FullyConnectedLayer represents a fully connected layer in the CNN
type FullyConnectedLayer struct {
	InputSize  int
	InputWidth int
	InputDepth int
	OutputSize int
	Weights    [][]float32
	Biases     []float32
	Input      []float32
	Output     []float32
}

// NewFullyConnectedLayer creates a new FullyConnectedLayer object with the specified parameters
func NewFullyConnectedLayer(inputWidth, inputDepth, outputSize int) *FullyConnectedLayer {
	inputSize := inputDepth * (inputWidth * inputWidth)
	biases := make([]float32, outputSize)
	weights := make([][]float32, inputSize)

	// Use He initialization with a mean of 0.0 and standard deviation of sqrt(2 / input_neurons)
	normal := rand.New(rand.NewSource(42))
	standardDeviation := float32(math.Sqrt(2.0 / float64(inputSize*inputDepth)))

	for i := range weights {
		weights[i] = make([]float32, outputSize)
		for j := range weights[i] {
			weights[i][j] = float32(normal.NormFloat64()) * standardDeviation
		}
	}

	return &FullyConnectedLayer{
		InputSize:  inputSize,
		InputWidth: inputWidth,
		InputDepth: inputDepth,
		OutputSize: outputSize,
		Weights:    weights,
		Biases:     biases,
		Input:      nil,
		Output:     make([]float32, outputSize),
	}
}

// Flatten a 3D vector into a 1D vector
func flatten(squares [][][]float32) []float32 {
	flatData := make([]float32, 0)

	for _, square := range squares {
		for _, row := range square {
			flatData = append(flatData, row...)
		}
	}

	return flatData
}

// ForwardPropagate performs forward propagation through the FullyConnectedLayer
func (fcl *FullyConnectedLayer) ForwardPropagate(matrixInput [][][]float32) [][][]float32 {
	// Flatten the input matrix into a 1D vector
	input := flatten(matrixInput)
	// Store the input for backpropagation
	fcl.Input = input
	for j := 0; j < fcl.OutputSize; j++ {
		// Calculate the weighted sum of the inputs
		fcl.Output[j] = fcl.Biases[j]
		for i := 0; i < fcl.InputSize; i++ {
			fcl.Output[j] += input[i] * fcl.Weights[i][j]
		}
		// Apply the sigmoid activation function to the output
		fcl.Output[j] = sigmoid(fcl.Output[j])
	}

	// Format the output to be a 3D vector
	formattedOutput := [][][]float32{{fcl.Output}}
	return formattedOutput
}

// BackPropagate performs backpropagation through the FullyConnectedLayer
func (fcl *FullyConnectedLayer) BackPropagate(matrixError [][][]float32) [][][]float32 {
	// Flatten the error matrix into a 1D vector
	errorData := matrixError[0][0]
	for j := 0; j < fcl.OutputSize; j++ {
		errorData[j] *= invDerivSigmoid(fcl.Output[j])
	}

	flatError := make([]float32, fcl.InputSize)

	// Update the weights and biases according to their derivatives
	for j := 0; j < fcl.OutputSize; j++ {
		fcl.Biases[j] -= errorData[j] * learningRate
		for i := 0; i < fcl.InputSize; i++ {
			flatError[i] += errorData[j] * fcl.Weights[i][j]
			fcl.Weights[i][j] -= errorData[j] * fcl.Input[i] * learningRate
		}
	}

	// Format the error to be a 3D vector
	prevError := make([][][]float32, fcl.InputDepth)
	for i := 0; i < fcl.InputDepth; i++ {
		prevError[i] = make([][]float32, fcl.InputWidth)
		for j := 0; j < fcl.InputWidth; j++ {
			prevError[i][j] = make([]float32, fcl.InputWidth)
			for k := 0; k < fcl.InputWidth; k++ {
				index := i*fcl.InputWidth*fcl.InputWidth + j*fcl.InputWidth + k
				prevError[i][j][k] = flatError[index]
			}
		}
	}

	return prevError
}

// GetOutput returns the output value at the specified index
func (fcl *FullyConnectedLayer) GetOutput(index int) float32 {
	return fcl.Output[index]
}

/*

func main() {
	// Example usage...
	inputWidth := 8
	inputDepth := 3
	outputSize := 10

	fcLayer := NewFullyConnectedLayer(inputWidth, inputDepth, outputSize)

	// Example forward propagation
	inputData := make([][][]float32, inputDepth)
	for i := range inputData {
		inputData[i] = make([][]float32, inputWidth)
		for j := range inputData[i] {
			inputData[i][j] = make([]float32, inputWidth)
			for k := range inputData[i][j] {
				inputData[i][j][k] = float32(i + j + k)
			}
		}
	}

	outputData := fcLayer.ForwardPropagate(inputData)
	fmt.Println("Forward Propagation Output:")
	printOutput(outputData)

	// Example backpropagation
	errorData := make([][][]float32, 1)
	errorData[0] = make([][]float32, 1)
	errorData[0][0] = make([]float32, outputSize)
	for i := range errorData[0][0] {
		errorData[0][0][i] = float32(i)
	}

	prevErrorData := fcLayer.BackPropagate(errorData)
	fmt.Println("\nBackpropagation Output:")
	printOutput(prevErrorData)
}

// Helper function to print the output matrix
func printOutput(output [][][]float32) {
	for i := range output {
		for j := range output[i] {
			fmt.Printf("%v\n", output[i][j])
		}
	}
}
*/
