package layers

import (
	"math"
	"math/rand"
)

// ConvLayer represents a convolutional layer in the CNN
type ConvLayer struct {
	InputSize  int
	InputDepth int
	NumFilters int
	KernelSize int
	OutputSize int
	Stride     int
	Biases     []float32
	Kernels    [][][][]float32
	Input      [][][]float32
	Output     [][][]float32
}

// NewConvLayer creates a new ConvLayer object with the specified parameters
func NewConvLayer(inputSize, inputDepth, numFilters, kernelSize, stride int) *ConvLayer {
	biases := make([]float32, numFilters)
	kernels := make([][][][]float32, numFilters)

	// Use He initialization with a mean of 0.0 and standard deviation of sqrt(2 / (inputDepth * kernelSize^2))
	normal := rand.New(rand.NewSource(42))
	standardDeviation := float32(math.Sqrt(2.0 / float64(inputDepth*kernelSize*kernelSize)))

	for f := 0; f < numFilters; f++ {
		biases[f] = 0.1
		kernels[f] = make([][][]float32, inputDepth)
		for i := 0; i < inputDepth; i++ {
			kernels[f][i] = make([][]float32, kernelSize)
			for j := 0; j < kernelSize; j++ {
				kernels[f][i][j] = make([]float32, kernelSize)
				for k := 0; k < kernelSize; k++ {
					kernels[f][i][j][k] = float32(normal.NormFloat64()) * standardDeviation
				}
			}
		}
	}

	outputSize := ((inputSize - kernelSize) / stride) + 1

	return &ConvLayer{
		InputSize:  inputSize,
		InputDepth: inputDepth,
		NumFilters: numFilters,
		KernelSize: kernelSize,
		OutputSize: outputSize,
		Stride:     stride,
		Biases:     biases,
		Kernels:    kernels,
		Input:      nil,
		Output:     make([][][]float32, numFilters),
	}
}

// ForwardPropagate performs forward propagation through the ConvLayer
func (cl *ConvLayer) ForwardPropagate(input [][][]float32) [][][]float32 {
	cl.Input = cloneInput(input)
	cl.Output = make([][][]float32, cl.NumFilters)

	for f := 0; f < cl.NumFilters; f++ {
		cl.Output[f] = make([][]float32, cl.OutputSize)
		for i := 0; i < cl.OutputSize; i++ {
			cl.Output[f][i] = make([]float32, cl.OutputSize)
			for j := 0; j < cl.OutputSize; j++ {
				cl.Output[f][i][j] = cl.Biases[f]

				for f_i := 0; f_i < cl.InputDepth; f_i++ {
					for y_k := 0; y_k < cl.KernelSize; y_k++ {
						for x_k := 0; x_k < cl.KernelSize; x_k++ {
							val := cl.Input[f_i][i*cl.Stride+y_k][j*cl.Stride+x_k]
							cl.Output[f][i][j] += cl.Kernels[f][f_i][y_k][x_k] * val
						}
					}
				}
			}
		}
	}

	// Apply ReLU activation function
	for f := 0; f < cl.NumFilters; f++ {
		for i := 0; i < cl.OutputSize; i++ {
			for j := 0; j < cl.OutputSize; j++ {
				cl.Output[f][i][j] = max(0.0, cl.Output[f][i][j])
			}
		}
	}

	return cloneInput(cl.Output)
}

// BackPropagate performs backpropagation through the ConvLayer
func (cl *ConvLayer) BackPropagate(error [][][]float32) [][][]float32 {
	prevError := make([][][]float32, cl.InputDepth)
	newKernels := cloneKernels(cl.Kernels)

	for f_i := 0; f_i < cl.InputDepth; f_i++ {
		prevError[f_i] = make([][]float32, cl.InputSize)
		for i := 0; i < cl.InputSize; i++ {
			prevError[f_i][i] = make([]float32, cl.InputSize)
		}
	}

	for y := 0; y < cl.OutputSize; y++ {
		for x := 0; x < cl.OutputSize; x++ {
			left := x * cl.Stride
			top := y * cl.Stride

			for f := 0; f < cl.NumFilters; f++ {
				if cl.Output[f][y][x] > 0.0 {
					cl.Biases[f] -= error[f][y][x] * learningRate

					for y_k := 0; y_k < cl.KernelSize; y_k++ {
						for x_k := 0; x_k < cl.KernelSize; x_k++ {
							for f_i := 0; f_i < cl.InputDepth; f_i++ {
								prevError[f_i][top+y_k][left+x_k] +=
									cl.Kernels[f][f_i][y_k][x_k] * error[f][y][x]

								newKernels[f][f_i][y_k][x_k] -=
									cl.Input[f_i][top+y_k][left+x_k] * error[f][y][x] * learningRate
							}
						}
					}
				}
			}
		}
	}

	cl.Kernels = newKernels

	return prevError
}

// GetOutput returns the output value at the specified index
func (cl *ConvLayer) GetOutput(index int) float32 {
	panic("Convolutional layers should not be accessed directly.")
}

// Helper function to clone the input matrix
func cloneInput(input [][][]float32) [][][]float32 {
	clonedInput := make([][][]float32, len(input))
	for i := range input {
		clonedInput[i] = make([][]float32, len(input[i]))
		for j := range input[i] {
			clonedInput[i][j] = make([]float32, len(input[i][j]))
			copy(clonedInput[i][j], input[i][j])
		}
	}
	return clonedInput
}

// Helper function to clone the kernels matrix
func cloneKernels(kernels [][][][]float32) [][][][]float32 {
	clonedKernels := make([][][][]float32, len(kernels))
	for i := range kernels {
		clonedKernels[i] = make([][][]float32, len(kernels[i]))
		for j := range kernels[i] {
			clonedKernels[i][j] = make([][]float32, len(kernels[i][j]))
			for k := range kernels[i][j] {
				clonedKernels[i][j][k] = make([]float32, len(kernels[i][j][k]))
				copy(clonedKernels[i][j][k], kernels[i][j][k])
			}
		}
	}
	return clonedKernels
}

// Helper function to find the maximum of two float32 values
func max(a, b float32) float32 {
	if a > b {
		return a
	}
	return b
}
