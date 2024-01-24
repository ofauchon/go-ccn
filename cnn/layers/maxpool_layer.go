package layers

// MaxPoolingLayer represents a max pooling layer in the CNN
type MaxPoolingLayer struct {
	InputSize    int
	InputDepth   int
	KernelSize   int
	OutputSize   int
	Stride       int
	Output       [][][]float32
	HighestIndex [][][][2]int  // Tuple (top, left) representing the position of the highest value
	PrevError    [][][]float32 // Add this line

}

// NewMaxPoolingLayer creates a new MaxPoolingLayer object with the specified parameters
// inputSize: The spatial dimensions (width and height) of the input.
// inputDepth: number of channels or feature maps in the input data
// kernelSize:  The size of the pooling window used for max pooling.
// stride: The step size or interval at which the pooling window moves over the input.
func NewMaxPoolingLayer(inputSize, inputDepth, kernelSize, stride int) *MaxPoolingLayer {
	outputSize := ((inputSize - kernelSize) / stride) + 1

	// Create and return a new MaxPoolingLayer with the initialized parameters and slices
	mpl := &MaxPoolingLayer{
		InputSize:    inputSize,
		InputDepth:   inputDepth,
		KernelSize:   kernelSize,
		OutputSize:   outputSize,
		Stride:       stride,
		Output:       make([][][]float32, inputDepth),
		HighestIndex: make([][][][2]int, inputDepth),
		PrevError:    make([][][]float32, inputDepth),
	}

	// Initialize output slices
	for f := 0; f < mpl.InputDepth; f++ {
		mpl.Output[f] = make([][]float32, mpl.OutputSize)
		mpl.HighestIndex[f] = make([][][2]int, mpl.OutputSize)

		for y := 0; y < mpl.OutputSize; y++ {
			mpl.Output[f][y] = make([]float32, mpl.OutputSize)
			mpl.HighestIndex[f][y] = make([][2]int, mpl.OutputSize)
		}
	}

	// Initialize the prevError slices
	for i := range mpl.PrevError {
		mpl.PrevError[i] = make([][]float32, inputSize)
		for j := range mpl.PrevError[i] {
			mpl.PrevError[i][j] = make([]float32, inputSize)
		}
	}

	return mpl
}

// ForwardPropagate reduces the size of the input by using max pooling
func (mpl *MaxPoolingLayer) ForwardPropagate(input [][][]float32) [][][]float32 {
	// Loop through each output position in the output volume
	for y := 0; y < mpl.OutputSize; y++ {
		for x := 0; x < mpl.OutputSize; x++ {
			// Calculate the top-left corner of the receptive field
			left := x * mpl.Stride
			top := y * mpl.Stride
			for f := 0; f < mpl.InputDepth; f++ {
				mpl.Output[f][y][x] = -1.0
				// Loop through each position in the receptive field
				// and find the highest value
				for yP := 0; yP < mpl.KernelSize; yP++ {
					for xP := 0; xP < mpl.KernelSize; xP++ {
						val := input[f][top+yP][left+xP]
						if val > mpl.Output[f][y][x] {
							mpl.Output[f][y][x] = val

							// Store the position of the highest value for backpropagation
							mpl.HighestIndex[f][y][x] = [2]int{top + yP, left + xP}
						}
					}
				}
			}
		}
	}
	return mpl.Output
}

// BackPropagate back propagates the error in a max pooling layer.
// Takes in the error matrix and returns the previous error matrix
func (mpl *MaxPoolingLayer) BackPropagate(error [][][]float32) [][][]float32 {

	// Iterate through the output neurons
	for y := 0; y < mpl.OutputSize; y++ {
		for x := 0; x < mpl.OutputSize; x++ {
			// Input depth will always be the same as output depth
			for f := 0; f < mpl.InputDepth; f++ {
				pos := mpl.HighestIndex[f][y][x]
				// Update the input error value with the corresponding output error value
				mpl.PrevError[f][pos[0]][pos[1]] = error[f][y][x]
			}
		}
	}

	// Return the previous error vector
	return mpl.PrevError
}

// GetOutput returns the output value at the specified index
func (mpl *MaxPoolingLayer) GetOutput(index int) float32 {
	panic("Max pooling layers should not be accessed directly.")
}
