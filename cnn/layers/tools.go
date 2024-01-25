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
