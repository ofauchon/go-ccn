package main

import (
	"fmt"
	"log"
	"net/http"
	_ "net/http/pprof"
	"os"
	"runtime/pprof"

	"github.com/ofauchon/go-cnn/cnn"
	"github.com/petar/GoMNIST"
)

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

func main() {

	// For profiler
	go func() {
		log.Println(http.ListenAndServe("localhost:6060", nil))
	}()

	// Read MNIST dataset
	_, testData, err := GoMNIST.Load("datasets/mnist")
	if err != nil {
		fmt.Println("Error loading MNIST dataset:", err)
		return
	}
	fmt.Printf("MNIST OK: TEST  count:%d dimensions:%dx%d, size:%d\n", testData.Count(), testData.NCol, testData.NCol, len(testData.Images[0]))

	fn := "/tmp/cnn.json"
	// Create a new CNN and specify its layers
	fmt.Println("Initializing CNN")
	jsonData, err := os.ReadFile(fn)
	if err != nil {
		panic(err)
	}

	cn := cnn.DecodeCNN(jsonData)
	fmt.Println("Initializing CNN OK")

	// Keep track of detection success for statistics
	resultsHistory := []bool{}
	accuracy := float32(0)

	for i := 0; i < len(testData.Images); i++ {
		if i%10 == 0 {
			// Compute results stats
			trueCnt := uint(0)
			for _, v := range resultsHistory {
				if v == true {
					trueCnt += 1
				}
			}
			accuracy = float32(trueCnt) / float32(len(resultsHistory))
			fmt.Printf("Image %d, Accuracy %f\n", i, accuracy*100)
		}

		output := cn.ForwardPropagate(ConvertRawImageToFloat32(testData.Images[i]))
		result := highestIndex(output) == uint8(testData.Labels[i])

		resultsHistory = append(resultsHistory, result)

	}

	pprof.StopCPUProfile()
}
