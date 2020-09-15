from union_find import *

def test(name):
    file_name = "datasets/"+name+".csv"
    X = np.genfromtxt(file_name, delimiter=",")
    res =all_together(X, 2, d_min=1)

if __name__ == '__main__':
    import timeit
    print("IRIS")
    print(timeit.timeit("test(\"IRIS\")",
                        setup="from __main__ import test",
                        number=100))
    print("DIABETES")
    print(timeit.timeit("test(\"DIABETES\")",
                        setup="from __main__ import test",
                        number=10))
    print("MICE")
    print(timeit.timeit("test(\"MICE\")",
                        setup="from __main__ import test",
                        number=4))
