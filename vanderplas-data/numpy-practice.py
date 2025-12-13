import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    return mo, np


@app.cell
def _(np):
    np.__version__
    return


@app.cell
def _(mo):
    mo.md("""
    ## Data Types in Python
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ### Definitions
    """)
    return


@app.cell
def _(np):
    import array

    L1 = list(range(10))
    L2 = [str(c) for c in L1]
    L3 = [True, "2", 3.0, 4]
    A1 = array.array("i", L1)
    int_arr = np.array([1, 4, 2, 5, 3])
    upcast_arr = np.array([3.14, 4, 2, 3])
    f32_arr = np.array([1, 2, 3, 4], dtype="float32")
    nd_arr = np.array([range(i, i + 3) for i in [2, 4, 6]])

    # array of length 10 filled with zeros
    zero_arr = np.zeros(10, dtype=int)

    # create a 3x5 floating point array filled with ones
    ones_arr = np.ones((3, 5), dtype=float)

    # create a 3x5 filled with 3.14
    full_arr = np.full((3, 5), 3.14)

    # create an array filled with a linear sequence -> start at 0, end at 20, step by 2
    range_arr = np.arange(0, 20, 2)

    # create an array of 5 valkues evenly spaced between 0 and 1
    space_arr = np.linspace(0, 1, 5)

    # create a 3x3 array of uniformly distributed random values between 0 and 1
    random_arr = np.random.random((3, 3))

    # create a 3x3 array of normally distributed random values with mean 0 and standard deviation 1
    normal_random_arr = np.random.normal(0, 1, (3, 3))

    # create a 3x3 array of random integers in the interval [0, 10)
    randint_arr = np.random.randint(0, 10, (3, 3))

    # create a 3x3 identity matrix
    identity_matrix = np.eye(3)

    # create an uninitialized array of 3 integers -> values will be whatever happens to already exist at that memory location
    empty_arr = np.empty(3)
    return (
        A1,
        L1,
        L2,
        L3,
        empty_arr,
        f32_arr,
        full_arr,
        identity_matrix,
        int_arr,
        nd_arr,
        normal_random_arr,
        ones_arr,
        randint_arr,
        random_arr,
        range_arr,
        space_arr,
        upcast_arr,
        zero_arr,
    )


@app.function
def print_arr(dtype, val, showType=False):
    print(dtype)
    print(val)
    if showType:
        print(type(val[0]))


@app.cell
def _(L1):
    print_arr("list", L1, showType=True)
    return


@app.cell
def _(L2):
    # print(L2)
    # print(type(L2[0]))
    print_arr("list", L2, showType=True)
    return


@app.cell
def _(L3):
    print([type(item) for item in L3])
    return


@app.cell
def _(mo):
    mo.md("""
    #### Fixed-Type Arrays in Python
    """)
    return


@app.cell
def _(A1):
    print_arr("fixed-type-array", A1)
    return


@app.cell
def _(mo):
    mo.md("""
    #### Numpy ndarrays
    """)
    return


@app.cell
def _(int_arr):
    print_arr("numpy integer array", int_arr)  # integer array
    return


@app.cell
def _(upcast_arr):
    print_arr("upcast numpy array ", upcast_arr)
    return


@app.cell
def _(f32_arr):
    print_arr("floating point 32 array", f32_arr)
    return


@app.cell
def _(nd_arr):
    print_arr("n-dimensional numpy array", nd_arr)
    return


@app.cell
def _(zero_arr):
    print_arr("zeros", zero_arr)
    return


@app.cell
def _(ones_arr):
    print_arr("ones", ones_arr)
    return


@app.cell
def _(full_arr):
    print_arr("full", full_arr)
    return


@app.cell
def _(range_arr):
    print_arr("arange", range_arr)
    return


@app.cell
def _(space_arr):
    print_arr("linspace", space_arr)
    return


@app.cell
def _(random_arr):
    print_arr("random", random_arr)
    return


@app.cell
def _(normal_random_arr):
    print_arr("random normal", normal_random_arr)
    return


@app.cell
def _(randint_arr):
    print_arr("randint", randint_arr)
    return


@app.cell
def _(identity_matrix):
    print_arr("identity", identity_matrix)
    return


@app.cell
def _(empty_arr):
    print_arr("empty", empty_arr)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Fundamentals of Numpy
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    **Attributes** -> Determining the size, shape, memory consumption, and data types of arrays
    **Indexing** -> Getting and setting the value of individual array elements

    **Slicing** -> Getting and setting smaller subarrays within a larger array

    **Reshaping** -> Changing the shape of a given array

    **Join and Split** -> Combining multiple arrays into one, and splitting one array into many
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ### Defintions
    """)
    return


@app.function
def print_attr(name, val):
    print(
        f"name -> {name}\ndimension -> {val.ndim}\nshape -> {val.shape}\nsize -> {val.size}\ndatatype -> {val.dtype}\nitemsize(array element size) -> {val.itemsize} bytes\nnbytes(total size) -> {val.nbytes} bytes\n{val}"
    )


@app.cell
def _(np):
    np.random.seed(0)
    x1 = np.random.randint(10, size=6)
    x2 = np.random.randint(10, size=(3, 4))
    x3 = np.random.randint(10, size=(3, 4, 5))
    return x1, x2, x3


@app.cell
def _(x1):
    print_attr("x1", x1)
    return


@app.cell
def _(x2):
    print_attr("x2", x2)
    return


@app.cell
def _(x3):
    print_attr("x3", x3)
    return


if __name__ == "__main__":
    app.run()
