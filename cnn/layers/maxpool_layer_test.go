package layers

import (
	"reflect"
	"testing"
)

func TestMPLForwardPropagate(t *testing.T) {
	// Create an instance of your MaxPoolingLayer
	// Input size : 4x4
	// Input depth : 3
	// Kernel : 2x2
	// Stride : 2
	mpl := NewMaxPoolingLayer(4, 3, 2, 2)

	// Sample input (4x4 depth=3)
	input := [][][]float32{
		{{1.0, 2.0, 3.0, 4.0}, {5.0, 6.0, 7.0, 8.0}, {9.0, 10.0, 11.0, 12.0}, {13.0, 14.0, 15.0, 16.0}},
		{{17.0, 18.0, 19.0, 20.0}, {21.0, 22.0, 23.0, 24.0}, {25.0, 26.0, 27.0, 28.0}, {29.0, 30.0, 31.0, 32.0}},
		{{33.0, 34.0, 35.0, 36.0}, {37.0, 38.0, 39.0, 40.0}, {41.0, 42.0, 43.0, 44.0}, {45.0, 46.0, 47.0, 48.0}},
	}

	// Expected output (2x2 depth=3)
	expectedOutput := [][][]float32{
		{{6.0, 8.0}, {14.0, 16.0}},
		{{22.0, 24.0}, {30.0, 32.0}},
		{{38.0, 40.0}, {46.0, 48.0}},
	}

	// Call the ForwardPropagate function
	output := mpl.ForwardPropagate(input)

	// Compare the output with the expected result
	if !reflect.DeepEqual(output, expectedOutput) {
		t.Errorf("ForwardPropagate did not produce the expected output. Got %v, expected %v", output, expectedOutput)
	}
}
