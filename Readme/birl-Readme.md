This file has two functions that call the appropriate functions in llh.py file to perform MAP or MMAP using BIRL inference.

It samples weights using the sampleNewWeight function from utils.py.

Then it uses a scipy optimizing function minimize to find the most appropriate gradient and posterior that fit the evidence.

Records the time taken and returns all the above as results.
