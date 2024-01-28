package cnn

import (
	"encoding/json"

	"github.com/ofauchon/go-cnn/cnn/layers"
)

type LayerInfo struct {
	Type       string      // Layer type identifier (FullyConnectedLayer, ConvLayer, MaxPoolingLayer)
	Properties interface{} // Layer-specific properties
}

func EncodeCNN(cnn *CNN) []byte {
	// Create a slice to store LayerInfo structs
	layerInfos := []LayerInfo{}

	// Iterate through each layer in the CNN
	for _, layer := range cnn.Layers {
		switch layer := layer.(type) {
		case *layers.FullyConnectedLayer:
			layerInfo := LayerInfo{
				Type: "FullyConnectedLayer",
				Properties: struct {
					InputSize  int
					InputWidth int
					InputDepth int
					OutputSize int
					Weights    [][]float32
					Biases     []float32
					Input      []float32
					Output     []float32
				}{
					layer.InputSize, layer.InputWidth, layer.InputDepth, layer.OutputSize, layer.Weights, layer.Biases, layer.Input, layer.Output},
			}
			layerInfos = append(layerInfos, layerInfo)
		case *layers.ConvLayer:
			layerInfo := LayerInfo{
				Type: "ConvLayer",
				Properties: struct {
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
				}{
					layer.InputSize, layer.InputDepth, layer.NumFilters, layer.KernelSize, layer.OutputSize, layer.Stride, layer.Biases, layer.Kernels, layer.Input, layer.Output},
			}
			layerInfos = append(layerInfos, layerInfo)
		case *layers.MaxPoolingLayer:
			layerInfo := LayerInfo{
				Type: "MaxPoolingLayer",
				Properties: struct {
					InputSize    int
					InputDepth   int
					PoolSize     int
					OutputSize   int
					Stride       int
					Output       [][][]float32
					HighestIndex [][][][]int   // Tuple (x,y,z)[2]int representing the position of the highest value
					PrevError    [][][]float32 // Add this line
				}{
					layer.InputSize, layer.InputDepth, layer.PoolSize, layer.OutputSize, layer.Stride, layer.Output, layer.HighestIndex, layer.PrevError},
			}
			layerInfos = append(layerInfos, layerInfo)
		}
	}

	// Convert layerInfos into JSON-like representation
	jsonData, err := json.Marshal(layerInfos)
	if err != nil {
		panic(err)
	}

	return jsonData
}

func DecodeCNN(jsonData []byte) CNN {
	// Decode JSON-like representation into layerInfos slice
	var layerInfos []LayerInfo
	err := json.Unmarshal(jsonData, &layerInfos)
	if err != nil {
		panic(err)
	}

	// Create the CNN struct
	cnn := CNN{}

	// Iterate through layerInfos and reconstruct layers
	for _, layerInfo := range layerInfos {
		switch layerInfo.Type {
		case "FullyConnectedLayer":
			fullyConnectedLayer := &layers.FullyConnectedLayer{}

			d, err := json.Marshal(layerInfo.Properties)
			if err != nil {
				panic(err)
			}

			err = json.Unmarshal(d, &fullyConnectedLayer)
			if err != nil {
				panic(err)
			}
			cnn.Layers = append(cnn.Layers, fullyConnectedLayer)

		case "ConvLayer":
			convLayer := &layers.ConvLayer{}

			d, err := json.Marshal(layerInfo.Properties)
			if err != nil {
				panic(err)
			}

			err = json.Unmarshal(d, &convLayer)
			if err != nil {
				panic(err)
			}
			cnn.Layers = append(cnn.Layers, convLayer)

		case "MaxPoolingLayer":
			maxPoolingLayer := &layers.MaxPoolingLayer{}

			d, err := json.Marshal(layerInfo.Properties)
			if err != nil {
				panic(err)
			}

			err = json.Unmarshal(d, &maxPoolingLayer)
			if err != nil {
				panic(err)
			}
			cnn.Layers = append(cnn.Layers, maxPoolingLayer)
		}
	}

	return cnn
}
