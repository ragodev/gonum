// Copyright ©2016 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package optimize

import (
	"math"

	"gonum.org/v1/gonum/stat/distmv"
)

// GuessAndCheck is a global optimizer that evaluates the function at random
// locations. Not a good optimizer, but useful for comparison and debugging.
type GuessAndCheck struct {
	Rander distmv.Rander

	bestF float64
	bestX []float64
}

func (g *GuessAndCheck) Needs() struct{ Gradient, Hessian bool } {
	return struct{ Gradient, Hessian bool }{false, false}
}

func (g *GuessAndCheck) InitGlobal(dim, tasks int) int {
	g.bestF = math.Inf(1)
	g.bestX = resize(g.bestX, dim)
	return tasks
}

func (g *GuessAndCheck) sendNewLoc(operation chan<- GlobalTask, task GlobalTask) {
	g.Rander.Rand(task.X)
	task.Operation = FuncEvaluation
	operation <- task
}

func (g *GuessAndCheck) updateMajor(operation chan<- GlobalTask, task GlobalTask) {
	// Update the best value seen so far, and send a MajorIteration.
	if task.F < g.bestF {
		g.bestF = task.F
		copy(g.bestX, task.X)
	} else {
		task.F = g.bestF
		copy(task.X, g.bestX)
	}
	task.Operation = MajorIteration
	operation <- task
}

func (g *GuessAndCheck) RunGlobal(operation chan<- GlobalTask, result <-chan GlobalTask, tasks []GlobalTask) {
	// Send initial tasks to evaluate
	for i, task := range tasks {
		task.Index = i + 1
		g.sendNewLoc(operation, task)
	}

	// Read from the channel until PostIteration is sent.
Outer:
	for {
		task := <-result
		switch task.Operation {
		default:
			panic("unknown operation")
		case PostIteration:
			break Outer
		case MajorIteration:
			g.sendNewLoc(operation, task)
		case FuncEvaluation:
			g.updateMajor(operation, task)
		}
	}

	// PostIteration was sent. Update the best new values.
	for task := range result {
		switch task.Operation {
		default:
			panic("unknown operation")
		case MajorIteration:
		case FuncEvaluation:
			g.updateMajor(operation, task)
		}
	}
	close(operation)
}
