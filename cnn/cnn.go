package cnn

import (
	"github.com/ofauchon/go-cnn/cnn/layers"
)

// Layer interface represents the common behavior of ConvLayer, MaxPoolingLayer, and FullyConnectedLayer
type Layer interface {
	ForwardPropagate(input [][][]float32) [][][]float32
	BackPropagate(error [][][]float32) [][][]float32
	GetOutput(index int) float32
}

// CNN represents a Convolutional Neural Network
type CNN struct {
	Layers []Layer
}

// NewCNN creates a new empty CNN object
func NewCNN() *CNN {
	return &CNN{Layers: []Layer{}}
}

// AddConvLayer adds a convolutional layer to the neural network
func (c *CNN) AddConvLayer(inputSize, inputDepth, numFilters, kernelSize, stride int) {
	convLayer := layers.NewConvLayer(inputSize, inputDepth, numFilters, kernelSize, stride)
	c.Layers = append(c.Layers, convLayer)
}

// AddMaxPoolingLayer adds a max pooling layer to the neural network
func (c *CNN) AddMaxPoolingLayer(inputSize, inputDepth, kernelSize, stride int) {
	mxplLayer := layers.NewMaxPoolingLayer(inputSize, inputDepth, kernelSize, stride)
	c.Layers = append(c.Layers, mxplLayer)
}

// AddFullyConnectedLayer adds a fully connected layer to the neural network
func (c *CNN) AddFullyConnectedLayer(inputWidth, inputDepth, outputSize int) {
	fclLayer := layers.NewFullyConnectedLayer(inputWidth, inputDepth, outputSize)
	c.Layers = append(c.Layers, fclLayer)
}

// ForwardPropagate performs forward propagation through the CNN
func (c *CNN) ForwardPropagate(image [][][]float32) []float32 {
	output := image

	// Forward propagate through each layer of the network
	for _, layer := range c.Layers {
		output = layer.ForwardPropagate(output)
	}

	// Flatten and return the output of the final layer
	return flattenOutput(output)
}

// LastLayerError calculates the error of the last layer of the network
// It returns a slice with errors correction for every neuron
func (c *CNN) LastLayerError(label int) [][][]float32 {
	var error []float32

	corrFactor := (2.0 / 10.0)

	// Calculate the error for each output neuron
	for i := 0; i < 10; i++ {
		desired := float32(0)
		if label == i {
			desired = 1
		}
		lastIndex := len(c.Layers) - 1
		error = append(error, float32(corrFactor)*(c.Layers[lastIndex].GetOutput(i)-desired))
	}

	return [][][]float32{{error}}
}

// BackPropagate performs backpropagation through the CNN
func (c *CNN) BackPropagate(label int) {
	// Retrieve the last layer error to backpropagate
	error := c.LastLayerError(label)

	// Iterate backwards through the layers and backpropagate the error
	for i := len(c.Layers) - 1; i >= 0; i-- {
		error = c.Layers[i].BackPropagate(error)
	}
}

// flattenOutput flattens the output of the final layer
func flattenOutput(output [][][]float32) []float32 {
	return output[0][0]
}
