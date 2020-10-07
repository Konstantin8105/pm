package pm

import (
	"fmt"
	"math"
	"math/rand"
	"runtime/debug"
	"sort"
	"time"

	"github.com/Konstantin8105/errors"
	"github.com/Konstantin8105/sparse"
)

// Config is default property of power method
type Config struct {
	// Maximal amount iteration
	IterationMax uint64

	// Iteration tolerance
	// TODO (KI) : add comment about for each lambda precasion is lost
	Tolerance float64
}

// Pm power method for approximating eigenvalues.
type Pm struct {
	a      *sparse.Matrix
	ignore []int
	config Config

	// eigenvalue result of calculation
	𝜦 float64

	// eigenvector result of calculation
	𝑿 []float64
}

// zeroize - set 0.0 in each element of slice
func zeroize(x []float64) {
	for i := range x {
		x[i] = 0.0
	}
}

// oneMax - modify slice with max value 1.0
func oneMax(x []float64) {
	// check input data
	if len(x) == 0 {
		return
	}

	// find max value
	max := x[0]
	min := x[0]
	for i := range x {
		if min < x[i] && x[i] < max {
			continue
		}
		if x[i] < min {
			min = x[i]
		}
		if x[i] > max {
			max = x[i]
		}
	}
	if math.Abs(min) > max {
		max = min
	}
	// if max == 1.0 {
	// 	return
	// }
	//
	// if max == 66.66 , then max = 100
	// if max == 0.006 , then max = 0.01
	// if max > 0 {
	// 	max = math.Pow(10.0, float64(int(math.Log10(max))))
	// } else {
	// 	max = -math.Pow(10.0, float64(int(math.Log10(-max))))
	// }

	// modification
	for i := range x {
		// TODO (KI) : need parallel
		x[i] /= max
	}
}

// PM is power method for approximating eigenvalues.
// Find `dominant eigenvalue` of matrix A.
// List `ignore` is list of ignore row and column in calculation.
//
//	Algorith : Power Method
//	x(0) // initial vector
//	k = 1
//	for δ < ɛ {
//		x(k) = A · x(k-1)
//		δ = || x(k) - x(k-1) ||1
//		k = k + 1
//	}
//
// See next articles:
//
//	1. Sepandar D. Kamvar, Taher H. Haveliwala, Christopher D. Manning, Gene H. Golub
//	"Extrapolation Methods for Accelerating PageRank Computations"
//
func (pm *Pm) Factorize(A *sparse.Matrix, config *Config, ignore ...int) (err error) {
	// check input data
	et := errors.New("Function LU.Factorize: check input data")
	if A == nil {
		_ = et.Add(fmt.Errorf("matrix A is nil"))
	}
	if A != nil {
		rows, columns := A.Dims()
		if rows <= 0 {
			_ = et.Add(fmt.Errorf("matrix row A is valid: %v", rows))
		}
		if columns <= 0 {
			_ = et.Add(fmt.Errorf("matrix columns A is valid: %v", columns))
		}
		if A.IsTriplet() {
			_ = et.Add(fmt.Errorf("matrix A is not CSC(Compressed Sparse Column) format"))
		}
	}

	// sort ignore list
	(sort.IntSlice(ignore)).Sort()
	if len(ignore) > 0 && A != nil {
		rows, columns := A.Dims()
		if ignore[0] < 0 {
			_ = et.Add(fmt.Errorf("ignore list have index less zero: %d", ignore[0]))
		}
		if ignore[len(ignore)-1] >= rows || ignore[len(ignore)-1] >= columns {
			_ = et.Add(fmt.Errorf("ignore list have index outside matrix"))
		}
	}

	if et.IsError() {
		err = et
		return
	}

	// panic free. replace to stacktrace
	defer func() {
		if r := recover(); r != nil {
			err = fmt.Errorf("stacktrace from panic: %s\n", debug.Stack())
		}
	}()

	// minimal configuration
	if config == nil {
		config = &Config{
			IterationMax: 500,
			Tolerance:    1e-5,
		}
	}

	// remove duplicate from `ignore` list
	if len(ignore) > 0 {
		list := append([]int{}, ignore[0])
		for i := 1; i < len(ignore); i++ {
			if ignore[i-1] == ignore[i] {
				continue
			}
			list = append(list, ignore[i])
		}
		list, ignore = ignore, list
	}

	// coping matrix A without ignored rows and columns
	C, _ := A.Copy()

	// store
	pm.ignore = ignore
	pm.a = C
	pm.config = *config

	return
}

// TODO (KI) : add research PM from input values
// TODO (KI) : [7 4 1; 4 4 4; 1 4 7]
// TODO (KI) : xo = [1 2 3]T
// TODO (KI) : xo = [0 1 -1]T

// Eigen calculate eigenvalue
func (pm *Pm) Eigen() (err error) {
	// panic free. replace to stacktrace
	defer func() {
		if r := recover(); r != nil {
			err = fmt.Errorf("stacktrace from panic: %s\n", debug.Stack())
		}
	}()

	// workspace
	var (
		rows, cols = pm.a.Dims()
		x          = make([]float64, rows)
		xNext      = make([]float64, rows)
	)
	_ = cols
	x = x[:rows]
	xNext = xNext[:rows]

	rand.Seed(time.Now().UnixNano())

	for i := range x {
		x[i] = rand.Float64() - 0.5
	}

	dlast := 1.0
	oneMax(x)

	// iteration value
	var iter uint64

	// main iteration function
	iteration := func() {
		// x(k) = A*x(k-1) - (∑(𝛌(j)*[x(j)]*[x(j)Transpose])) * x(k-1),
		//        |      |   |                                       |
		//        +------+   +---------------------------------------+
		//          part1               part 2
		// where j = 0 ... k-1
		oneMax(x)
		for _, i := range pm.ignore {
			x[i] = 0.0
		}
		zeroize(xNext)
		for _, i := range pm.ignore {
			xNext[i] = 0.0
		}
		_, _ = sparse.Fkeep(pm.a, func(i, j int, val float64) bool {
			xNext[i] += val * x[j]
			return true
		})

		// value pm.E.X without ignore elements
		EX := make([]float64, rows)
		copy(EX, pm.𝑿)
		for row := 0; row < rows; row++ {
			for col := 0; col < rows; col++ {
				xNext[row] -= pm.𝜦 * EX[row] * EX[col] * x[col]
			}
		}

		// post work
		x, xNext = xNext, x
	}

	// calculation
	for {
		// main iteration function
		oneMax(x)
		iteration()

		// first norm of vector
		// link: https://en.wikipedia.org/wiki/Norm_(mathematics)#Taxicab_norm_or_Manhattan_norm
		oneMax(x)
		oneMax(xNext)
		var d float64
		for i := range x {
			d += math.Abs(x[i] - xNext[i])
		}

		if math.Abs(dlast-d) < pm.config.Tolerance {
			// tolerance breaking
			break
		}
		if iter >= pm.config.IterationMax {
			err = ErrorPm{
				Iteration:    iter,
				IterationMax: pm.config.IterationMax,
				Delta:        dlast - d,
				Tolerance:    pm.config.Tolerance,
				err:          fmt.Errorf("iteration limit"),
			}
			return
		}

		dlast = d
		iter++
	}

	// compute the Rayleigh quotient
	// 𝛌 = (Ax · x) / (x · x)

	// calculation of Ax is ignore and takes value x(k-1)

	// up : Ax · x
	iteration()

	var up float64
	for i := range x {
		up += x[i] * xNext[i]
	}

	// down : x · x
	oneMax(x)
	var down float64
	for i := range x {
		down += x[i] * x[i]
	}

	if math.Abs(down) == 0.0 || math.IsNaN(up) {
		return fmt.Errorf("Not acceptable value")
	}

	// calculation eigenvalue
	𝛌 := up / down

	oneMax(x)

	pm.𝜦 = 𝛌 // eigenvalue
	pm.𝑿 = x // eigenvector

	return
}

// ErrorPm error of power method
type ErrorPm struct {
	Iteration    uint64
	IterationMax uint64
	Delta        float64
	Tolerance    float64
	err          error
}

func (e ErrorPm) Error() string {
	et := errors.New("Power method error")
	_ = et.Add(fmt.Errorf("Iteration: %d", e.Iteration))
	_ = et.Add(fmt.Errorf("Max.Iteration: %d", e.IterationMax))
	_ = et.Add(fmt.Errorf("Delta tolerance: %5e", e.Tolerance))
	_ = et.Add(fmt.Errorf("Tolerance: %5e", e.Tolerance))
	_ = et.Add(e.err)
	return et.Error()
}
