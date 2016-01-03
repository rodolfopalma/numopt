package numopt

import (
	"math"

	"github.com/rpalmaotero/linalg"
)

const (
	CONVERGENCE_EPSILON = 0.00001
	GRADIENT_EPSILON    = 0.00001
)

type Solution struct {
	currentPoint linalg.VectorStructure
	optimal      bool
}

func GradientDescent(f func(linalg.VectorStructure) float64, x0 linalg.VectorStructure, lambda float64) (linalg.VectorStructure, int) {
	i := 1
	x := x0
	for {
		gradient := NumericalGradient(f, x)
		newX := x.Minus(gradient.ScalarMul(lambda))
		if gradient.Norm(2) < CONVERGENCE_EPSILON {
			break
		}
		// fmt.Println(i, gradient, x, f(x))
		x = newX
		i = i + 1
	}
	return x, i
}

func NewtonRaphson(f func(linalg.VectorStructure) float64, x0 linalg.VectorStructure, lambda float64) (linalg.VectorStructure, int) {
	i := 1
	x := x0
	for {
		gradient := NumericalGradient(f, x)
		hessian := NumericalHessian(f, x)
		direction := NumericalHessian.Inverse().Mul(gradient)
		newX := x.Minus(direction.ScalarMul(lambda))
		if gradient.Norm(2) < CONVERGENCE_EPSILON {
			break
		}
		// fmt.Println(i, gradient, x, f(x))
		x = newX
		i = i + 1
	}
	return x, i
}

// TODO: A lot to optimize in this method.
func NumericalGradient(f func(linalg.VectorStructure) float64, x linalg.VectorStructure) linalg.VectorStructure {
	gradientValues := make([]float64, 0)
	for i, value := range x {
		tempEpsilon := x.Remove(i)
		plusEpsilon := tempEpsilon.Insert(i, value+GRADIENT_EPSILON)
		minusEpsilon := tempEpsilon.Insert(i, value-GRADIENT_EPSILON)
		gradientValues = append(gradientValues, (f(plusEpsilon)-f(minusEpsilon))/(2*GRADIENT_EPSILON))
	}
	return linalg.NewVector(gradientValues)
}

func NumericalHessian(f func(linalg.VectorStructure) float64, x linalg.VectorStructure) linalg.MatrixStructure {
	hessianValues := make([][]float64, 0)
	for i, iValue := range x {
		rowValues := make([]float64, 0)
		tempRowVector := x.Remove(i)
		var value float64
		for j, jValue := range x {
			tempRowPlusEpsilon := tempRowVector.Insert(i, iValue+GRADIENT_EPSILON)
			tempRowMinusEpsilon := tempRowVector.Insert(i, iValue-GRADIENT_EPSILON)
			if i == j {
				value = (f(tempRowPlusEpsilon) - 2*f(x) + f(tempRowMinusEpsilon)) / (math.Pow(GRADIENT_EPSILON, 2))
				// fmt.Println(tempRowPlusEpsilon, tempRowMinusEpsilon, value)
			} else {
				tempPlus := tempRowPlusEpsilon.Remove(j)
				tempMinus := tempRowMinusEpsilon.Remove(j)
				tempPlusPlus := tempPlus.Insert(j, jValue+GRADIENT_EPSILON)
				tempPlusMinus := tempPlus.Insert(j, jValue-GRADIENT_EPSILON)
				tempMinusPlus := tempMinus.Insert(j, jValue+GRADIENT_EPSILON)
				tempMinusMinus := tempMinus.Insert(j, jValue-GRADIENT_EPSILON)
				value = (f(tempPlusPlus) - f(tempPlusMinus) - f(tempMinusPlus) + f(tempMinusMinus)) / (4 * math.Pow(GRADIENT_EPSILON, 2))
			}
			rowValues = append(rowValues, value)
		}
		hessianValues = append(hessianValues, rowValues)
	}
	return linalg.NewMatrix(hessianValues)

}
