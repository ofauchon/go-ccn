# go-ccn
Experimentations with golang and CCN


## Mnist handwitten digits 

This example code will try learning from dataset/mnist
$ go run examples/mnist/learn.go

## Profiling

go tool pprof  http://localhost:6060/debug/pprof/heap
go tool pprof  http://localhost:6060/debug/pprof/profile?seconds=5
