// Package layers provide various layers
package layers

// MaxPoolingLayer represents a max pooling layer in the CNN
type MaxPoolingLayer struct {
	InputSize    int
	InputDepth   int
	PoolSize     int
	OutputSize   int
	Stride       int
	Output       [][][]float32
	HighestIndex [][][][]int   // Tuple (x,y,z)[2]int representing the position of the highest value // TODO: WTF
	PrevError    [][][]float32 // Add this line

}

// NewMaxPoolingLayer creates a new custom MaxPooling layer.
// inputSize is the spatial dimensions (width and height) of the input.
// inputDepth is the number of channels or feature maps in the input data
// poolSize is the spatial dimensions (width and height) of the pooling window used.
// stride is the step size or interval at which the pooling window moves over the input.
// It returns an initialized MaxPoolingLayer object
func NewMaxPoolingLayer(inputSize, inputDepth, poolSize, stride int) *MaxPoolingLayer {
	// Determine the output size based on the input size and the pool size
	outputSize := ((inputSize - poolSize) / stride) + 1

	// Create and return a new MaxPoolingLayer with the initialized parameters and slices
	mpl := &MaxPoolingLayer{
		InputSize:  inputSize,
		InputDepth: inputDepth,
		PoolSize:   poolSize,
		OutputSize: outputSize,
		Stride:     stride,
	}

	mpl.Output = make3D[float32](inputDepth, outputSize, outputSize)
	mpl.HighestIndex = make4D[int](inputDepth, outputSize, outputSize, 2)
	mpl.PrevError = make3D[float32](inputDepth, inputSize, inputSize)

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
				for yP := 0; yP < mpl.PoolSize; yP++ {
					for xP := 0; xP < mpl.PoolSize; xP++ {
						val := input[f][top+yP][left+xP]
						if val > mpl.Output[f][y][x] {
							mpl.Output[f][y][x] = val

							// Store the position of the highest value for backpropagation
							mpl.HighestIndex[f][y][x] = []int{top + yP, left + xP}
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
