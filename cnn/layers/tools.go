package layers

func make3D[T any](d1, d2, d3 int) [][][]T {

	x := make([][][]T, d1)
	for i1 := 0; i1 < d1; i1++ {
		x[i1] = make([][]T, d2)
		for i2 := 0; i2 < d2; i2++ {
			x[i1][i2] = make([]T, d3)
		}
	}
	return x
}

func make4D[T any](d1, d2, d3, d4 int) [][][][]T {

	x := make([][][][]T, d1)

	for i1 := 0; i1 < d1; i1++ {
		x[i1] = make3D[T](d2, d3, d4)
	}

	return x
}

// Helper function to clone 3D Matrix
func clone3D(data [][][]float32) [][][]float32 {
	clonedInput := make([][][]float32, len(data))
	for i := range data {
		clonedInput[i] = make([][]float32, len(data[i]))
		for j := range data[i] {
			clonedInput[i][j] = make([]float32, len(data[i][j]))
			copy(clonedInput[i][j], data[i][j])
		}
	}
	return clonedInput
}

// Helper function to clone 4D Matrix
func clone4D(data [][][][]float32) [][][][]float32 {
	clonedKernels := make([][][][]float32, len(data))
	for i := range data {
		clonedKernels[i] = make([][][]float32, len(data[i]))
		for j := range data[i] {
			clonedKernels[i][j] = make([][]float32, len(data[i][j]))
			for k := range data[i][j] {
				clonedKernels[i][j][k] = make([]float32, len(data[i][j][k]))
				copy(clonedKernels[i][j][k], data[i][j][k])
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
