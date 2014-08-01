// +build amd64

package gonn

import (
	"github.com/ziutek/blas"
)

func ddot64(X, Y []float64) float64 {
	return blas.Ddot(len(X), X, 1, Y, 1)
}
