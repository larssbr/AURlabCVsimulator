import cProfile
import simulation
import slicSuperpixel_lbp_method
#cProfile.run('simulation.main()')
#cProfile.run('slicSuperpixel_lbp_method.main()')

# 1 call the profiling code
#  python -m cProfile -o output_slicSuperpixel_lbp_method.txt slicSuperpixel_lbp_method.py

# python -m cProfile -o output_simulation.txt simulation.py

# 2 run this code
import pstats
p = pstats.Stats("output.txt")
p.strip_dirs().sort_stats(-1).print_stats()




# python -m timeit -s 'import sum' 'slicSuperpixel_lbp_method.main()'


# snakeviz output_simulation.txt
