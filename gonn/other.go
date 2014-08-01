// +build 386 arm

package gonn

func ddot64(X, Y []float64) float64 {
	sum := 0.0
	for i, x := range X {
		sum += x * Y[i]
	}
	return sum
}
