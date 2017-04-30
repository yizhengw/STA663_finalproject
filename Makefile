all: regular regularreal cyth cythreal ji jireal te pro

regular:
  cd IBP_regular; IBP_mainsolver_simualtion_regular.py

regularreal:
  cd IBP_regular; IBP_realdata_regular.py
  
cyth:
  cd IBP_optimize_Cython; IBP_mainsolver_simulation_cython.py
  
cythreal:
  cd IBP_optimize_Cython; IBP_realdata_cython.py
  
ji:
  cd IBP_Optimize; IBP_mainsolver_simulation_jit.py

jireal:
  cd IBP_Optimize; IBP_realdata_jit.py
  
te:
  Test.py

pro:
  profile_and_optimization.py
