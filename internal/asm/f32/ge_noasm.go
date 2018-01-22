// Copyright ©2017 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//+build !go1.8 !amd64 noasm appengine

package f32

// Ger performs the rank-one operation
//  A += alpha * x * y^T
// where A is an m×n dense matrix, x and y are vectors, and alpha is a scalar.
func Ger(m, n uintptr, alpha float32,
	x []float32, incX uintptr,
	y []float32, incY uintptr,
	a []float32, lda uintptr) {

	if incX == 1 && incY == 1 {
		x = x[:m]
		y = y[:n]
		for i, xv := range x {
			AxpyUnitary(alpha*xv, y, a[uintptr(i)*lda:uintptr(i)*lda+n])
		}
		return
	}

	var ky, kx uintptr
	if int(incY) > 0 {
		ky = 0
	} else {
		ky = -(n - 1) * incY
	}
	if int(incX) > 0 {
		kx = 0
	} else {
		kx = -(m - 1) * incX
	}

	ix := kx
	for i := 0; i < int(m); i++ {
		AxpyInc(alpha*x[ix], y, a[uintptr(i)*lda:uintptr(i)*lda+n], uintptr(n), uintptr(incY), 1, uintptr(ky), 0)
		ix += incX
	}
}
