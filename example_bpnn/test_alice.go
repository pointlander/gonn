package main
import (
	"flag"
	"fmt"
	"github.com/dcadenas/pagerank"
	"github.com/pointlander/gonn/gonn"
	"io/ioutil"
	"math"
	"runtime"
	//"os"
)

const (
	WIDTH = 16
	LABEL = 0x4000000000000000
	SYMBOL = 0x2000000000000000
)

type AliceSet struct {
	alice []byte
}

func (s *AliceSet) Len() int {
	return len(s.alice) - WIDTH + 1
}

func (s *AliceSet) Fill(inputs, targets []float64, i int) {
	for j := range inputs {
		inputs[j] = 0
	}
	for j := range targets {
		targets[j] = 0
	}
	for j := 0; j < WIDTH; j++ {
		inputs[int(s.alice[i + j]) + j * 256], targets[int(s.alice[i + j]) + j * 256] = 1, 1
	}
}

var (
	train = flag.Bool("train", false, "train the neural network")
	run = flag.Bool("run", false, "run the neural network")
)

func main(){
	runtime.GOMAXPROCS(64)
	flag.Parse()
	size := WIDTH * 256

	if *train {
		nn := gonn.DefaultNetwork(size, []int{256, 128, 64, 32, 16, 8}, size, false, 1)
		data, err := ioutil.ReadFile("alice30.txt")
		if err != nil {
			fmt.Println(err)
			return
		}
		set := &AliceSet{data}

		nn.TrainStackedAutoencoderSet(set, 10)
		gonn.DumpNN("alice.nn", nn)
	}

	if *run {
		nn := gonn.LoadNN("alice.nn")
		/*nn := gonn.DefaultNetwork(size, []int{1024, 512, 256, 128, 64, 32, 16, 8}, size, false)*/
		depth := len(nn.HiddenLayer) - 1
		alice, err := ioutil.ReadFile("alice30.txt")
		if err != nil {
			fmt.Println(err)
			return
		}
		/*labeled, err := os.Create("alice30.l")
		if err != nil {
			fmt.Println(err)
			return
		}
		defer labeled.Close()*/
		length, inputs := len(alice), make([]float64, size)
		graph, graph1, graph2, labels, labels1, labels2 := pagerank.New(), pagerank.New(), pagerank.New(), make([]uint8, length), make([]uint16, length), make([]uint32, length)
		for i := 0; i < length; i++ {
			for j := range inputs {
				inputs[j] = 0
			}
			for j := 0; j < WIDTH; j++ {
				if index := i + j - WIDTH/2; index > -1 && index < length {
					inputs[int(alice[index]) + j * 256] = 1
				}
			}


			getBits := func(depth, count int) (bits uint64) {
				for _, bit := range nn.HiddenLayer[depth][:count] {
					if bit > .5 {
						bits |= 1
					}
					bits <<= 1
				}
				return
			}
			nn.Forward(inputs)
			a, a1, a2 := uint8(getBits(depth, 8)), uint16(getBits(depth-1, 16)), uint32(getBits(depth-2, 32))


			/*for j := WIDTH/2 * 256; j < (WIDTH/2 + 1) * 256; j++ {
				inputs[j] = 0
			}
			nn.Forward(inputs, depth)
			b, b1 := uint8(getBits(depth, 8)), uint16(getBits(depth-1, 16))*/

			labels[i] = a
			labels1[i] = a1
			labels2[i] = a2
			if i > 0 {
				graph.Link(int(labels[i-1]), int(labels[i]))
				graph1.Link(int(labels1[i-1]), int(labels1[i]))
				graph2.Link(int(labels2[i-1]), int(labels2[i]))
			}
		}

		ranks, ranks1, ranks2 := make(map[int] float64), make(map[int] float64), make(map[int] float64)
		graph.Rank(1, 1E-6, func(i int, rank float64) {
			ranks[i] = rank
		})
		graph1.Rank(1, 1E-6, func(i int, rank float64) {
			ranks1[i] = rank
		})
		graph2.Rank(1, 1E-6, func(i int, rank float64) {
			ranks2[i] = rank
		})

		total := make([]float64, length)
		var max float64
		for i := range labels {
			total[i] = (ranks[int(labels[i])] + ranks1[int(labels1[i])] + ranks2[int(labels2[i])])/3
			if total[i] > max {
				max = total[i]
			}
		}

		/*for i := range alice {
			fmt.Printf("%v\n", ranks[i])
		}*/

		for i, symbol := range alice {
			if symbol == '\n' || symbol == '\r' || symbol == ' ' || symbol == '\t' {
				fmt.Printf("%c", symbol)
			} else {
				rank := total[i]
				/*a := 8 * rank / max
				b := a / math.Sqrt(a * a + 1)
				c := 31 + math.Floor(b * 7)*/
				c := 31 + math.Floor(7 * rank / max)
				fmt.Printf("\x1B[%vm%c\x1B[m", c, symbol)
				//fmt.Printf("\x1B[%vm%c\x1B[m", 31 + int(labels[i] % 7), symbol)
			}
		}
	}
}
