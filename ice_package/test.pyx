from cython cimport view
cdef class A:
    cdef int i, j



cdef A ee
cdef A ees = A()
cdef view.array my_array = view.array(shape=(10), itemsize=sizeof(A))
my_array[0] = ees
print A
