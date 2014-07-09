package main
import (
	"fmt"
	"github.com/pointlander/gonn/gonn"
)

type XorSet struct {
	inputs, targets [][]float64
}

func (s *XorSet) Len() int {
	return len(s.inputs)
}

func (s *XorSet) Fill(inputs, targets []float64, i int) {
	copy(inputs, s.inputs[i])
	copy(targets, s.targets[i])
}

func main(){
	nn := gonn.DefaultNetwork(2,2,1,true)
	set := &XorSet{
		[][]float64{
			[]float64{0,0},
			[]float64{0,1},
			[]float64{1,0},
			[]float64{1,1},
		},
		[][]float64{
			[]float64{0}, //0 xor 0 == 0
			[]float64{1}, //0 xor 1 == 1
			[]float64{1}, //1 xor 0 == 1
			[]float64{0}, //1 xor 1 == 0
		},
	}

	nn.TrainSet(set, 1000)

	for _,p := range set.inputs {
		fmt.Printf("%.1f\n",nn.Forward(p)[0])
	}

}
