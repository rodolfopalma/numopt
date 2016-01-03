package numopt

import (
	"flag"
	"fmt"
	"math"
	"os"
	"testing"

	"github.com/rpalmaotero/linalg"
)

var parabola func(linalg.VectorStructure) float64
var banana func(linalg.VectorStructure) float64

func TestMain(main *testing.M) {
	flag.Parse()
	parabola = func(x linalg.VectorStructure) float64 {
		return math.Pow(x[0]*x[1], 2) + math.Pow(x[1], 2)
	}
	banana = func(x linalg.VectorStructure) float64 {
		return math.Pow(1-x[0], 2) + 100*math.Pow(x[1]-math.Pow(x[0], 2), 2)
	}
	os.Exit(main.Run())
}

func TestGradientDescent(t *testing.T) {
	optimal, iter := GradientDescent(parabola, linalg.NewVector([]float64{-10, -10}), 0.01)
	fmt.Println(optimal, iter)
}

func TestNumericalHessian(t *testing.T) {
	hessian := NumericalHessian(parabola, linalg.NewVector([]float64{10, 10}))
	fmt.Println(hessian)
}
