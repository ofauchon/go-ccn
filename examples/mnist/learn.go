package main

import (
	"fmt"
	"log"
	"math/rand"
	"net/http"
	_ "net/http/pprof"
	"runtime/pprof"
	"time"

	"github.com/ofauchon/go-cnn/cnn"
	"github.com/petar/GoMNIST"
)

// ConvertRawImageToFloat32 converts a raw MNIST image to [][]float32
func ConvertRawImageToFloat32(rawImage GoMNIST.RawImage) [][][]float32 {
	imgWidth := 28
	imgHeight := 28

	var floatImage [][]float32

	for h := 0; h < imgHeight; h++ {
		var row []float32

		for w := 0; w < imgWidth; w++ {
			// Convert raw byte to float32 and normalize to [0.0, 1.0]
			pixelValue := float32(rawImage[h*imgWidth+w]) / 255.0
			row = append(row, pixelValue)
		}

		floatImage = append(floatImage, row)
	}

	var ret [][][]float32
	ret = append(ret, floatImage)

	return ret
}

// Returns the index of the highest value in the output vector
func highestIndex(output []float32) uint8 {
	var highestIndex uint8
	var highestValue float32

	for i := 0; i < len(output); i++ {
		if output[i] > highestValue {
			highestValue = output[i]
			highestIndex = uint8(i)
		}
	}

	return highestIndex
}

func convertLabelsToUint8(labels []GoMNIST.Label) []uint8 {
	uint8Slice := make([]uint8, len(labels))

	for i, label := range labels {
		uint8Slice[i] = uint8(label)
	}

	return uint8Slice
}

func main() {
	rand.Seed(time.Now().UnixNano())

	// For profiler
	go func() {
		log.Println(http.ListenAndServe("localhost:6060", nil))
	}()

	// Read MNIST dataset
	trainData, testData, err := GoMNIST.Load("datasets/mnist")
	if err != nil {
		fmt.Println("Error loading MNIST dataset:", err)
		return
	}
	fmt.Printf("MNIST OK: TRAIN count:%d dimensions:%dx%d, size:%d\n", trainData.Count(), trainData.NCol, trainData.NCol, len(trainData.Images[0]))
	fmt.Printf("MNIST OK: TEST  count:%d dimensions:%dx%d, size:%d\n", testData.Count(), testData.NCol, testData.NCol, len(testData.Images[0]))

	// Create a new CNN and specify its layers
	fmt.Println("Initializing CNN")
	cnn := cnn.NewCNN()
	cnn.AddConvLayer(28, 1, 6, 5, 1)
	cnn.AddMaxPoolingLayer(24, 6, 2, 2)
	cnn.AddConvLayer(12, 6, 9, 3, 1)
	cnn.AddMaxPoolingLayer(10, 9, 2, 2)
	cnn.AddFullyConnectedLayer(5, 9, 10)

	// Training speed/length
	epochs := int(10)
	batchSize := int(500)

	// Keep track of detection success for statistics
	resultsHistory := []bool{}
	accuracy := float32(0)

	// Iteration of training with whole dataset
	for epoch := 1; epoch <= epochs; epoch++ {

		// Batch processing
		trainLength := trainData.Count()

		for batchStart := 0; batchStart < trainLength; batchStart += batchSize {
			batchEnd := batchStart + batchSize
			if batchEnd > trainData.Count() {
				batchEnd = trainData.Count()
			}
			fmt.Printf("Epoch: %d, Acc: %.2fpct Batch %d-%d \n", epoch, accuracy, batchStart, batchEnd)

			// Get a batch of train data (images/label)
			batch := trainData.Images[batchStart:batchEnd]
			labels := trainData.Labels[batchStart:batchEnd]

			for i := 0; i < len(batch); i++ {

				// Forward pass
				output := cnn.ForwardPropagate(ConvertRawImageToFloat32(batch[i]))

				// Check result and store it in result history
				result := highestIndex(output) == uint8(labels[i])
				resultsHistory = append(resultsHistory, result)

				// Back propagation
				cnn.BackPropagate(int(labels[i]))
			}

			// Compute results stats
			trueCnt := uint(0)
			for _, v := range resultsHistory {
				if v == true {
					trueCnt += 1
				}
			}
			accuracy = (float32(trueCnt) / float32(len(resultsHistory)) * 100)

			//optimizer.update(cnn) // Adjust this based on your optimizer implementation
		}
	}

	pprof.StopCPUProfile()
}
