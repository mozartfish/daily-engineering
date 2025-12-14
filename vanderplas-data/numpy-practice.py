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

    # create an array of 5 values evenly spaced between 0 and 1
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


@app.cell
def _(np, x2):
    def print_indexing(name, val, nd=False):
        print(f"name -> {name}")
        print(f"val -> {val}")
        if nd:
            val_copy = val.copy()
            print(f"Some values -> {val[0, 0], val[2, 0], val[2, -1]}")
            val_copy[0, 0] = 12
            print(f"Modify val -> {val_copy}")
        else:
            val_copy = val.copy()
            print(f"Some forward values -> {val[0], val[4]}")
            print(f"Some backward values -> {val[-1], val[-2]}")
            val_copy[0] = 3.14159
            print(f"Modify val\n{val_copy}")


    def print_slicing(name, val=x2, nd=False):
        if name == "1d":
            x = np.arange(10)
            print(f"val -> {x}")
            print(f"first five -> {x[:5]}")
            print(f"next five -> {x[5:]}")
            print(f"middle -> {x[4:7]}")
            print(f"every other -> {x[::2]}")
            print(f"start at one, every other -> {x[1::2]}")
            print(f"reverse -> {x[::-1]}")
            print(f"reverse every other starting from index 5 -> {x[5::-2]}")
        else:
            val_copy = val.copy()
            print(f"val -> {val}")
            print(f"two rows three columns -> {val_copy[:2, :3]}")
            print(f"all rows very other column -> {val_copy[:3, ::2]}")
            print(f"reverse subdimensions -> {val_copy[::-1, ::-1]}")
            print(f"first column of x2 -> {val_copy[:, 0]}")
            print(f"first row of x2 -> {val_copy[0, :]}")
            print(f"compact syntax -> {val_copy[0]}")


    def print_reshape():
        grid = np.arange(1, 10).reshape((3, 3))
        print(f"Grid -> {grid}")
        x = np.array([1, 2, 3])
        y = x.copy()
        z = x.copy()
        print(f"row vector via reshape -> {y.reshape((1, 3))}")
        print(f"row vector via newaxis -> {y[np.newaxis, :]}")
        print(f"column vector via reshape -> {z.reshape((3, 1))}")
        print(f"column vector via newaxis -> {z[:, np.newaxis]}")


    def print_concat():
        x = np.array([1, 2, 3])
        y = np.array([3, 2, 1])
        print(f"concat {x} {y} -> {np.concatenate([x, y])}")
        z = [99, 99, 99]
        print(f"concatenate again -> {np.concatenate([x, y, z])}")
        grid1 = np.arange(1, 7).reshape(2, 3)
        grid2 = grid1.copy()
        print(
            f"concatenate grid along first axis -> {np.concatenate([grid1, grid1])}"
        )
        print(
            f"concatenate grid along second axis -> {np.concatenate([grid2, grid2], axis=1)}"
        )
        x1 = np.array([1, 2, 3])
        grid3 = np.array([[9, 8, 7], [6, 5, 4]])
        print(f"vertically stack arrays -> {np.vstack([x, grid3])}")
        y2 = np.array([[99], [99]])
        print(f"horizontally stack arrays -> {np.hstack([grid3, y2])}")


    def print_split():
        x = [1, 2, 3, 99, 99, 3, 2, 1]
        x1, x2, x3 = np.split(x, [3, 5])
        print(f"x1 -> {x1}\nx2 -> {x2}\nx3 -> {x3}")
        grid = np.arange(16).reshape(4, 4)
        print(f"grid\n{grid}")
        upper, lower = np.vsplit(grid, [2])
        print(
            f"upper grid vertical split -> {upper}\nlower grid vertical split -> {lower}"
        )
        left, right = np.hsplit(grid, [2])
        print(
            f"left grid horizontal split -> {left}\nright grid horizontal split -> {right}"
        )
    return (
        print_concat,
        print_indexing,
        print_reshape,
        print_slicing,
        print_split,
    )


@app.cell
def _(print_indexing, x1):
    print_indexing("1-d array", x1)
    return


@app.cell
def _(print_indexing, x2):
    print_indexing("2-d array", x2, nd=True)
    return


@app.cell
def _(print_slicing):
    print_slicing("1d")
    return


@app.cell
def _(print_slicing):
    print_slicing("2d-array", nd=True)
    return


@app.cell
def _(print_reshape):
    print_reshape()
    return


@app.cell
def _(print_concat):
    print_concat()
    return


@app.cell
def _(print_split):
    print_split()
    return


@app.cell
def _(mo):
    mo.md("""
    ## Computation on Numpy Arrays
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
