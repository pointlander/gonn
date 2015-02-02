package gonn

import (
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"os"
	"time"
)

const NEURON_PARALLELISM = 128

type NeuralNetwork struct {
	HiddenLayer      [][]float64
	InputLayer       []float64
	OutputLayer      []float64
	WeightHidden     [][][]float64
	WeightOutput     [][]float64
	ErrOutput        []float64
	ErrHidden        [][]float64
	LastChangeHidden [][][]float64
	LastChangeOutput [][]float64
	Regression       bool
	Rate1            float64 //learning rate
	Rate2            float64
	seed		 int64
	random		 *rand.Rand
}

func sigmoid(X float64) float64 {
	return 1.0 / (1.0 + math.Pow(math.E, -float64(X)))
}

func dsigmoid(Y float64) float64 {
	return Y * (1.0 - Y)
}

func DumpNN(fileName string, nn *NeuralNetwork){
	out_f, err := os.OpenFile(fileName,os.O_CREATE | os.O_RDWR,0777)
	if err!=nil{
		panic("failed to dump the network to " + fileName)
	}
	defer out_f.Close()
	encoder := json.NewEncoder(out_f)
	err = encoder.Encode(nn)
	if err!=nil{
		panic(err)
	}
}

func LoadNN(fileName string) *NeuralNetwork{
	in_f, err := os.Open(fileName)
	if err!=nil{
		panic("failed to load "+fileName)
	}
	defer in_f.Close()
	decoder := json.NewDecoder(in_f)
	nn := &NeuralNetwork{}
	err = decoder.Decode(nn)
	if err!=nil{
		panic(err)
	}
	//fmt.Println(nn)
	return nn
}

func makeMatrix(rows, colums int, value float64) [][]float64 {
	mat := make([][]float64, rows)
	for i := 0; i < rows; i++ {
		mat[i] = make([]float64, colums)
		for j := 0; j < colums; j++ {
			mat[i][j] = value
		}
	}
	return mat
}

func randomMatrix(random *rand.Rand, rows, colums int, lower, upper float64) [][]float64 {
	mat := make([][]float64, rows)
	for i := 0; i < rows; i++ {
		mat[i] = make([]float64, colums)
		for j := 0; j < colums; j++ {
			mat[i][j] = random.Float64()*(upper-lower) + lower
		}
	}
	return mat
}

func DefaultNetwork(iInputCount int, iHiddenCount []int, iOutputCount int, iRegression bool, seed int64) *NeuralNetwork {
	return NewNetwork(iInputCount, iHiddenCount, iOutputCount, iRegression, 0.25, 0.1, seed)
}

func NewNetwork(iInputCount int, iHiddenCount []int, iOutputCount int, iRegression bool, iRate1, iRate2 float64, seed int64) *NeuralNetwork {
	iInputCount += 1
	for i := range iHiddenCount {
		iHiddenCount[i] += 1
	}
	network := &NeuralNetwork{}
	if seed == 0 {
		seed = time.Now().UnixNano()
	}
	network.seed = seed
	network.random = rand.New(rand.NewSource(seed))
	network.Regression = iRegression
	network.Rate1 = iRate1
	network.Rate2 = iRate2
	network.InputLayer = make([]float64, iInputCount)

	network.HiddenLayer, network.ErrHidden = make([][]float64, len(iHiddenCount)), make([][]float64, len(iHiddenCount))
	network.WeightHidden, network.LastChangeHidden = make([][][]float64, len(iHiddenCount)), make([][][]float64, len(iHiddenCount))
	previousSize := iInputCount
	for i, size := range iHiddenCount {
		network.HiddenLayer[i], network.ErrHidden[i] = make([]float64, size), make([]float64, size)
		network.WeightHidden[i], network.LastChangeHidden[i] = randomMatrix(network.random, size, previousSize, -1.0, 1.0), makeMatrix(size, previousSize, 0.0)
		previousSize = size
	}

	network.OutputLayer, network.ErrOutput = make([]float64, iOutputCount), make([]float64, iOutputCount)
	network.WeightOutput, network.LastChangeOutput = randomMatrix(network.random, iOutputCount, previousSize, -1.0, 1.0), makeMatrix(iOutputCount, previousSize, 0.0)

	return network
}

func (n *NeuralNetwork) Copy() *NeuralNetwork {
	c := new(NeuralNetwork)

	c.HiddenLayer = make([][]float64, len(n.HiddenLayer))
	for i, j := range n.HiddenLayer {
		c.HiddenLayer[i] = make([]float64, len(j))
		copy(c.HiddenLayer[i], j)
	}

	c.InputLayer = make([]float64, len(n.InputLayer))
	copy(c.InputLayer, n.InputLayer)

	c.OutputLayer = make([]float64, len(n.OutputLayer))
	copy(c.OutputLayer, n.OutputLayer)

	c.WeightHidden = make([][][]float64, len(n.WeightHidden))
	for x, y := range n.WeightHidden {
		c.WeightHidden[x] = make([][]float64, len(y))
		for i, j := range y {
			c.WeightHidden[x][i] = make([]float64, len(j))
			copy(c.WeightHidden[x][i], j)
		}
	}

	c.WeightOutput = make([][]float64, len(n.WeightOutput))
	for i, j := range n.WeightOutput {
		c.WeightOutput[i] = make([]float64, len(j))
		copy(c.WeightOutput[i], j)
	}

	c.ErrOutput = make([]float64, len(n.ErrOutput))
	copy(c.ErrOutput, n.ErrOutput)

	c.ErrHidden = make([][]float64, len(n.ErrHidden))
	for i, j := range n.ErrHidden {
		c.ErrHidden[i] = make([]float64, len(j))
		copy(c.ErrHidden[i], j)
	}

	c.LastChangeHidden = make([][][]float64, len(n.LastChangeHidden))
	for x, y := range n.LastChangeHidden {
		c.LastChangeHidden[x] = make([][]float64, len(y))
		for i, j := range y {
			c.LastChangeHidden[x][i] = make([]float64, len(j))
			copy(c.LastChangeHidden[x][i], j)
		}
	}

	c.LastChangeOutput = make([][]float64, len(n.LastChangeOutput))
	for i, j := range n.LastChangeOutput {
		c.LastChangeOutput[i] = make([]float64, len(j))
		copy(c.LastChangeOutput[i], j)
	}

	c.Regression = n.Regression
	c.Rate1 = n.Rate1
	c.Rate2 = n.Rate2
	c.seed = n.seed
	c.random = rand.New(rand.NewSource(n.seed))

	return c
}

func (self *NeuralNetwork) forward64(begin, end, layer int, inputs []float64, done chan<- bool) {
	if i := end - begin; i > NEURON_PARALLELISM {
		i = begin + i/2
		a, b := make(chan bool, 1), make(chan bool, 1)
		go self.forward64(begin, i, layer, inputs, a)
		self.forward64(i, end, layer, inputs, b)
		<-a
	} else {
		hidden := self.HiddenLayer[layer]
		for neuron, weights := range self.WeightHidden[layer][begin:end] {
			hidden[begin + neuron] = sigmoid(ddot64(inputs, weights))
		}
	}
	done <- true
}

func (self *NeuralNetwork) forwardOutput64(begin, end int, inputs []float64, done chan<- bool) {
	if i := end - begin; i > NEURON_PARALLELISM {
		i = begin + i/2
		a, b := make(chan bool, 1), make(chan bool, 1)
		go self.forwardOutput64(begin, i, inputs, a)
		self.forwardOutput64(i, end, inputs, b)
		<-a
	} else {
		output := self.OutputLayer
		for neuron, weights := range self.WeightOutput[begin:end] {
			if sum := ddot64(inputs, weights); self.Regression {
				output[begin + neuron] = sum
			} else {
				output[begin + neuron] = sigmoid(sum)
			}
		}
	}
	done <- true
}

func (self *NeuralNetwork) forward(input []float64, depth int) (output []float64) {
	if len(input)+1 != len(self.InputLayer) {
		panic("amount of input variable doesn't match")
	}
	for i, in := range input {
		self.InputLayer[i] = in
	}

	inputs := self.InputLayer
	for layer := range self.WeightHidden[:depth+1] {
		inputs[len(inputs)-1] = 1.0 //bias node for input layer
		self.forward64(0, len(self.WeightHidden[layer])-1, layer, inputs, make(chan bool, 1))
		inputs = self.HiddenLayer[layer]
	}

	inputs[len(inputs)-1] = 1.0 //bias node for hidden layer
	self.forwardOutput64(0, len(self.WeightOutput), inputs, make(chan bool, 1))
	return self.OutputLayer[:]
}

func (self *NeuralNetwork) Forward(input []float64) (output []float64) {
	return self.forward(input, len(self.HiddenLayer) - 1)
}

func (self *NeuralNetwork) Feedback(target []float64, depth int) {
	for i := 0; i < len(self.OutputLayer); i++ {
		self.ErrOutput[i] = self.OutputLayer[i] - target[i]
	}

	for i := 0; i < len(self.HiddenLayer[depth])-1; i++ {
		err := 0.0
		for j := 0; j < len(self.OutputLayer); j++ {
			if self.Regression {
				err += self.ErrOutput[j] * self.WeightOutput[j][i]
			} else {
				err += self.ErrOutput[j] * self.WeightOutput[j][i] * dsigmoid(self.OutputLayer[j])
			}

		}
		self.ErrHidden[depth][i] = err
	}

	for i := 0; i < len(self.OutputLayer); i++ {
		delta := 0.0
		if self.Regression {
			delta = self.ErrOutput[i]
		} else {
			delta = self.ErrOutput[i] * dsigmoid(self.OutputLayer[i])
		}
		for j := 0; j < len(self.HiddenLayer[depth]); j++ {
			change := self.Rate1*delta*self.HiddenLayer[depth][j] + self.Rate2*self.LastChangeOutput[i][j]
			self.WeightOutput[i][j] -= change
			self.LastChangeOutput[i][j] = change

		}
	}

	input := self.InputLayer
	if depth > 0 {
		input = self.HiddenLayer[depth - 1]
	}
	for i := 0; i < len(self.HiddenLayer[depth])-1; i++ {
		delta := self.ErrHidden[depth][i] * dsigmoid(self.HiddenLayer[depth][i])
		for j := 0; j < len(input); j++ {
			change := self.Rate1*delta*input[j] + self.Rate2*self.LastChangeHidden[depth][i][j]
			self.WeightHidden[depth][i][j] -= change
			self.LastChangeHidden[depth][i][j] = change
		}
	}
}

func (self *NeuralNetwork) CalcError(target []float64) float64 {
	errSum := 0.0
	for i := 0; i < len(self.OutputLayer); i++ {
		err := self.OutputLayer[i] - target[i]
		errSum += 0.5 * err * err
	}
	return errSum
}

func genRandomIdx(random *rand.Rand, N int) []int {
	A := make([]int, N)
	for i := 0; i < N; i++ {
		A[i] = i
	}
	//randomize
	for i := 0; i < N; i++ {
		j := i + int(random.Float64()*float64(N-i))
		A[i], A[j] = A[j], A[i]
	}
	return A
}

type TrainingSet interface {
	Len() int
	Fill(inputs, targets []float64, i int)
}

func (self *NeuralNetwork) TrainSet(set TrainingSet, constraint interface{}) {
	length, inputs, targets := set.Len(), make([]float64, len(self.InputLayer) - 1), make([]float64, len(self.OutputLayer))

	iter_flag := -1
	i := 0
	loop: for true {
		idx_ary := genRandomIdx(self.random, length)
		cur_err := 0.0
		for j := 0; j < length; j++ {
			set.Fill(inputs, targets, idx_ary[j])
			self.forward(inputs, 0)
			self.Feedback(targets, 0)
			cur_err += self.CalcError(targets)
			if (j+1)%1000 == 0 {
				if iter_flag != i {
					fmt.Println("")
					iter_flag = i
				}
				fmt.Printf("iteration %vth / progress %.2f %% \r", i+1, float64(j)*100/float64(length))
			}
		}
		i++
		switch c := constraint.(type) {
		case int:
			if (c >= 10 && i%(c/10) == 0) || c < 10 {
				fmt.Printf("\niteration %vth MSE: %.5f", i, cur_err / float64(length))
			}
			if i >= c {
				break loop
			}
		case float64:
			error := cur_err / float64(length)
			if i%100 == 0 {
				fmt.Printf("\niteration %vth MSE: %.5f", i, error)
			}
			if error < c {
				break loop
			}
		}
	}
	fmt.Println("\ndone.")
}

func (self *NeuralNetwork) TrainStackedAutoencoderSet(set TrainingSet, constraint interface{}) {
	length, inputs, targets := set.Len(), make([]float64, len(self.InputLayer) - 1), make([]float64, len(self.OutputLayer))

	for layer := range self.HiddenLayer {
		fmt.Printf("\nlayer %v\n", layer)
		outputSize, hiddenSize := len(self.OutputLayer), len(self.HiddenLayer[layer])
		self.OutputLayer, self.ErrOutput = make([]float64, outputSize), make([]float64, outputSize)
		self.WeightOutput, self.LastChangeOutput = randomMatrix(self.random, outputSize, hiddenSize, -1.0, 1.0), makeMatrix(outputSize, hiddenSize, 0.0)

		iter_flag := -1
		i := 0
		loop: for true {
			idx_ary := genRandomIdx(self.random, length)
			cur_err := 0.0
			for j := 0; j < length; j++ {
				set.Fill(inputs, targets, idx_ary[j])
				self.forward(inputs, layer)
				self.Feedback(targets, layer)
				cur_err += self.CalcError(targets)
				if (j+1)%1000 == 0 {
					if iter_flag != i {
						fmt.Println("")
						iter_flag = i
					}
					fmt.Printf("iteration %vth / progress %.2f %% \r", i+1, float64(j)*100/float64(length))
				}
			}
			i++
			switch c := constraint.(type) {
			case int:
				if (c >= 10 && i%(c/10) == 0) || c < 10 {
					fmt.Printf("\niteration %vth MSE: %.5f", i, cur_err / float64(length))
				}
				if i >= c {
					break loop
				}
			case float64:
				error := cur_err / float64(length)
				if i%100 == 0 {
					fmt.Printf("\niteration %vth MSE: %.5f", i, error)
				}
				if error < c {
					break loop
				}
			}
		}
	}
	fmt.Println("\ndone.")
}

type data struct {
	inputs, targets [][]float64
}

func (d *data) Len() int {
	return len(d.inputs)
}

func (d *data) Fill(inputs, targets []float64, i int) {
	copy(inputs, d.inputs[i])
	copy(targets, d.targets[i])
}

func (self *NeuralNetwork) Train(inputs [][]float64, targets [][]float64, constraint interface{}) {
	if len(inputs[0])+1 != len(self.InputLayer) {
		panic("amount of input variable doesn't match")
	}
	if len(targets[0]) != len(self.OutputLayer) {
		panic("amount of output variable doesn't match")
	}

	self.TrainSet(&data{inputs: inputs, targets: targets}, constraint)
}

/*
func (self *NeuralNetwork) TrainMap(inputs []map[int]float64, targets [][]float64, iteration int) {
	if len(targets[0]) != len(self.OutputLayer) {
		panic("amount of output variable doesn't match")
	}

	iter_flag := -1
	for i := 0; i < iteration; i++ {
		idx_ary := genRandomIdx(self.random, len(inputs))
		cur_err := 0.0
		for j := 0; j < len(inputs); j++ {
			self.ForwardMap(inputs[idx_ary[j]])
			self.FeedbackMap(targets[idx_ary[j]],inputs[idx_ary[j]] )
			cur_err += self.CalcError(targets[idx_ary[j]])
			if (j+1)%1000 == 0 {
				if iter_flag != i {
					fmt.Println("")
					iter_flag = i
				}
				fmt.Printf("iteration %vth / progress %.2f %% \r", i+1, float64(j)*100/float64(len(inputs)))
			}
		}
		if (iteration >= 10 && (i+1)%(iteration/10) == 0) || iteration < 10 {
			fmt.Printf("\niteration %vth MSE: %.5f", i+1, cur_err / float64(len(inputs)))
		}
	}
	fmt.Println("\ndone.")
}


func (self *NeuralNetwork) ForwardMap(input map[int]float64) (output []float64) {
	for k,v := range input {
		self.InputLayer[k] = v
	}
	self.InputLayer[len(self.InputLayer)-1] = 1.0 //bias node for input layer

	for i := 0; i < len(self.HiddenLayer)-1; i++ {
		sum := 0.0
		for j,_ := range input{
			sum += self.InputLayer[j] * self.WeightHidden[i][j]
		}
		self.HiddenLayer[i] = sigmoid(sum)
	}

	self.HiddenLayer[len(self.HiddenLayer)-1] = 1.0 //bias node for hidden layer
	for i := 0; i < len(self.OutputLayer); i++ {
		sum := 0.0
		for j := 0; j < len(self.HiddenLayer); j++ {
			sum += self.HiddenLayer[j] * self.WeightOutput[i][j]
		}
		if self.Regression {
			self.OutputLayer[i] = sum
		} else {
			self.OutputLayer[i] = sigmoid(sum)
		}
	}
	return self.OutputLayer[:]
}

func (self *NeuralNetwork) FeedbackMap(target []float64,input map[int]float64) {
	for i := 0; i < len(self.OutputLayer); i++ {
		self.ErrOutput[i] = self.OutputLayer[i] - target[i]
	}

	for i := 0; i < len(self.HiddenLayer)-1; i++ {
		err := 0.0
		for j := 0; j < len(self.OutputLayer); j++ {
			if self.Regression {
				err += self.ErrOutput[j] * self.WeightOutput[j][i]
			} else {
				err += self.ErrOutput[j] * self.WeightOutput[j][i] * dsigmoid(self.OutputLayer[j])
			}

		}
		self.ErrHidden[i] = err
	}

	for i := 0; i < len(self.OutputLayer); i++ {
		for j := 0; j < len(self.HiddenLayer); j++ {
			change := 0.0
			delta := 0.0
			if self.Regression {
				delta = self.ErrOutput[i]
			} else {
				delta = self.ErrOutput[i] * dsigmoid(self.OutputLayer[i])
			}
			change = self.Rate1*delta*self.HiddenLayer[j] + self.Rate2*self.LastChangeOutput[i][j]
			self.WeightOutput[i][j] -= change
			self.LastChangeOutput[i][j] = change

		}
	}

	for i := 0; i < len(self.HiddenLayer)-1; i++ {
		for j , _ := range input {
			delta := self.ErrHidden[i] * dsigmoid(self.HiddenLayer[i])
			change := self.Rate1*delta*self.InputLayer[j] + self.Rate2*self.LastChangeHidden[i][j]
			self.WeightHidden[i][j] -= change
			self.LastChangeHidden[i][j] = change

		}
	}
}
*/
