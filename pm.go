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
	// find max value
	max, min := x[0], x[0]
	for i := range x {
		if x[i] < min {
			min = x[i]
		} else if max < x[i] {
			max = x[i]
		}
	}
	if max < math.Abs(min) {
		max = min
	}

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
		ignore = list
	}

	// store
	pm.ignore = ignore
	pm.a = A
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

	rand.Seed(time.Now().UnixNano())

	// prepare x
	for i := range x {
		x[i] = rand.Float64() - 0.5
	}
	oneMax(x)

	// main iteration function
	iteration := func() {
		oneMax(x)
		for _, i := range pm.ignore {
			x[i] = 0.0
		}
		zeroize(xNext)
		_ = sparse.Gaxpy(pm.a, x, xNext, true)
		x, xNext = xNext, x
	}

	// calculation
	dlast := 1.0
	var iter uint64
	for {
		// main iteration function
		iteration()

		// first norm of vector
		// link: https://en.wikipedia.org/wiki/Norm_(mathematics)#Taxicab_norm_or_Manhattan_norm
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
