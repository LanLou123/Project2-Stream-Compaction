CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Lan Lou
* Tested on:  Windows 10, i7-6700HQ @ 2.6GHz 16GB, GTX 1070 8GB (Personal Laptop)


## Questions:
- *Roughly optimize the block sizes of each of your implementations for minimal run time on your GPU.*

blocksize/ops' time(ms)|	naive scan(ms)|	efficient scan(ms)
-----|-----|----
128 |	1.911 |	0.541
256 |	1.908 |	0.533
512 |	1.92 |	0.545
1024 |	1.915 |	0.559

- the diagram bellow explicitly shows how variation of block size impact on both the naive and the optimized solutions for scanning, although it's not very apparent, I think block size == 256 is best suited for me, in terms of my own machine condition, algorithms, etc...

![](https://github.com/LanLou123/Project2-Stream-Compaction/raw/master/blocksizechart.JPG)

- *Compare all of these GPU Scan implementations (Naive, Work-Efficient, and Thrust) to the serial CPU version of Scan. Plot a graph of the comparison (with array size on the independent axis).*

Array Size(2^(n))/ops' time(ms)|	Naïve |	Work Efficient(optimized)|	Work Efficient(not optimized) |	Thrust
----|----|----|----|----
3	| 0.032 |	0.029 |	0.093 |	0.085
6	| 0.049	| 0.046	|0.058|	0.065
9	| 0.076	| 0.075	| 0.068	|0.249
12	| 0.09	| 0.137	| 0.135|	0.255
15	| 0.128	| 0.131 |	0.118|	0.263
18	| 0.306	| 0.471 |	0.167	|0.36
21	| 3.89	| 2.311	| 1.008	|0.37
24	| 35.408	| 19.495	|7.925|	0.932
27	| 317.784	| 141.549	|61.03|	5.704

- the corresponding graph are:

![](https://github.com/LanLou123/Project2-Stream-Compaction/raw/master/arraysize.JPG)


## Results

### I got this result using the following settings:

- ```blocksize``` = 1024
- ```SIZE``` (in main.cpp) =  1 << 25

```
****************
** SCAN TESTS **
****************
    [  18  25   1  39  31  37   5  10  42  11   4  11  12 ...  37   0 ]
==== cpu scan, power-of-two ====
   elapsed time: 221.761ms    (std::chrono Measured)
    [   0  18  43  44  83 114 151 156 166 208 219 223 234 ... 821806793 821806830 ]
==== cpu scan, non-power-of-two ====
   elapsed time: 77.1578ms    (std::chrono Measured)
    [   0  18  43  44  83 114 151 156 166 208 219 223 234 ... 821806711 821806713 ]
    passed
==== naive scan, power-of-two ====
   elapsed time: 73.3082ms    (CUDA Measured)
    [   0  18  43  44  83 114 151 156 166 208 219 223 234 ... 821806793 821806830 ]
    passed
==== naive scan, non-power-of-two ====
   elapsed time: 75.7463ms    (CUDA Measured)
    [   0  18  43  44  83 114 151 156 166 208 219 223 234 ...   0   0 ]
    passed
==== work-efficient scan, power-of-two ====
   elapsed time: 15.5116ms    (CUDA Measured)
    [   0  18  43  44  83 114 151 156 166 208 219 223 234 ... 821806793 821806830 ]
    passed
==== work-efficient scan, non-power-of-two ====
   elapsed time: 15.3897ms    (CUDA Measured)
    [   0  18  43  44  83 114 151 156 166 208 219 223 234 ... 821806711 821806713 ]
    passed
==== thrust scan, power-of-two ====
   elapsed time: 1.63021ms    (CUDA Measured)
    [   0  18  43  44  83 114 151 156 166 208 219 223 234 ... 821806793 821806830 ]
    passed
==== thrust scan, non-power-of-two ====
   elapsed time: 1.63738ms    (CUDA Measured)
    [   0  18  43  44  83 114 151 156 166 208 219 223 234 ... 821806711 821806713 ]
    passed

*****************************
** STREAM COMPACTION TESTS **
*****************************
    [   0   0   3   1   2   3   0   0   3   1   0   2   2 ...   1   0 ]
==== cpu compact without scan, power-of-two ====
   elapsed time: 121.362ms    (std::chrono Measured)
    [   3   1   2   3   3   1   2   2   1   1   2   2   1 ...   3   1 ]
    passed
==== cpu compact without scan, non-power-of-two ====
   elapsed time: 121.729ms    (std::chrono Measured)
    [   3   1   2   3   3   1   2   2   1   1   2   2   1 ...   2   3 ]
    passed
==== cpu compact with scan ====
   elapsed time: 293.673ms    (std::chrono Measured)
    [   3   1   2   3   3   1   2   2   1   1   2   2   1 ...   3   1 ]
    passed
==== work-efficient compact, power-of-two ====
   elapsed time: 15.202ms    (CUDA Measured)
    [   3   1   2   3   3   1   2   2   1   1   2   2   1 ...   3   1 ]
    passed
==== work-efficient compact, non-power-of-two ====
   elapsed time: 15.2422ms    (CUDA Measured)
    [   3   1   2   3   3   1   2   2   1   1   2   2   1 ...   2   3 ]
    passed
```
