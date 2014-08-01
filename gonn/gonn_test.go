package gonn

import (
	"testing"
)

func TestXOR(t *testing.T) {
	nn := DefaultNetwork(2, []int{2}, 1, true, 1)
	inputs := [][]float64{
		[]float64{0, 0},
		[]float64{0, 1},
		[]float64{1, 0},
		[]float64{1, 1},
	}
	targets := [][]float64{
		[]float64{0}, //0 xor 0 == 0
		[]float64{1}, //0 xor 1 == 1
		[]float64{1}, //1 xor 0 == 1
		[]float64{0}, //1 xor 1 == 0
	}

	nn.Train(inputs, targets, 1E-6)

	for _, p := range inputs {
		out, result := 0, nn.Forward(p)[0]
		if result > .5 {
			out = 1
		}
		if should_be := int(p[0])^int(p[1]); should_be != out {
			t.Errorf("%.1f^%.1f should be %.1f; got %.1f\n", p[0], p[1], float64(should_be), result)
		}
	}
}
