package main
import (
	"flag"
	"fmt"
	"github.com/dcadenas/pagerank"
	"github.com/pointlander/gonn/gonn"
	"io/ioutil"
	"math"
	//"os"
)

const (
	WIDTH = 8
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
	flag.Parse()

	if *train {
		nn := gonn.DefaultNetwork(WIDTH * 256, 8, WIDTH * 256, false)
		data, err := ioutil.ReadFile("alice30.txt")
		if err != nil {
			fmt.Println(err)
			return
		}
		set := &AliceSet{data}

		nn.TrainSet(set, 10)
		gonn.DumpNN("alice.nn", nn)
	}

	if *run {
		nn := gonn.LoadNN("alice.nn")
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
		length, inputs := len(alice), make([]float64, WIDTH * 256)
		buffer, graph := make([]uint16, length), pagerank.New()
		for i := 0; i < length; i++ {
			for j := range inputs {
				inputs[j] = 0
			}
			for j := 0; j < WIDTH; j++ {
				if i + j < length {
					inputs[int(alice[i + j]) + j * 256] = 1
				}
			}
			nn.Forward(inputs)
			for _, bit := range nn.HiddenLayer {
				if bit > .5 {
					buffer[i] |= 1
				}
				buffer[i] <<= 1
			}
			buffer[i] <<= 8
			buffer[i] |= uint16(alice[i])
			if i > 0 {
				graph.Link(int(buffer[i - 1]), int(buffer[i]))
			}
		}

		ranks := make(map[uint16] float64)
		var max float64
		var maxSymbol byte
		graph.Rank(0.85, 0.0001, func(identifier int, rank float64) {
			ranks[uint16(identifier)] = rank
			if rank > max {
				max, maxSymbol = rank, byte(identifier & 0xFF)
			}
		})

		/*fmt.Printf("max '%c' %v\n", maxSymbol, max)
		for identifier, rank := range ranks {
			a := 8 * rank / max
			b := a / math.Sqrt(a * a + 1)
			fmt.Printf("'%c' %v\n", byte(identifier & 0xFF), b)
		}*/

		for _, s := range buffer {
			symbol := byte(s & 0xFF)
			if symbol == '\n' || symbol == '\r' || symbol == ' ' || symbol == '\t' {
				fmt.Printf("%c", symbol)
			} else {
				a := 8 * ranks[s] / max
				b := a / math.Sqrt(a * a + 1)
				c := 31 + math.Floor(b * 7)
				fmt.Printf("\x1B[%vm%c\x1B[m", c, symbol)
			}
		}
	}
}
